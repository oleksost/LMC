import copy
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import ctrl
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from matplotlib import pyplot
from numpy.lib.arraysetops import isin
from simple_parsing import ArgumentParser, choice
from torch import nn
from torch.utils.data import DataLoader

import wandb
from main_transfer import (ArgsGenerator, bn_warmup, create_dataloader_ctrl,
                           get_accs_for_tasks, loss_function, set_seed, test,
                           test_with_bn)
from Methods.lerners.lllcst_learner import CompositionalDynamicFM
from Methods.models.cnn_soft_gated_lifelong_dynamic import \
    CNNSoftGatedLLDynamic
from Methods.replay import Buffer
from Utils.ctrl.ctrl.tasks.task_generator import TaskGenerator
from Utils.logging_utils import log_wandb
from Utils.utils import construct_name_ctrl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class ArgsGenerator(ArgsGenerator):       
    n_tasks:int = 6 #-
    gating: str = choice('soft_gating_pool', 'sof_gating', default='soft_gating_pool')
    component_update_frequency: int = 100 #-
    use_single_controller: int = 0 #-
    seperate_pool_per_layer: int = 0 #-
    padd_input: bool = 0 #-
    measure_transfer: bool = 0 #-
    single_oh_controller: bool = 0 #-
    def __post_init__(self):
        super().__post_init__()
        if self.debug:
            self.epochs=2
            self.component_update_frequency=1
            # self.hidden_size=64

def init_model(args:ArgsGenerator, gating='locspec', n_classes=10, i_size=28):
    from Methods import ModelOptions
    model_options = ModelOptions()
    # model_options.Module.activation_structural='relu'         
    model_options.Module.use_backup_system=args.use_backup_system
    model_options.Module.structure_inv=args.structure_inv
    model_options.Module.maxpool_kernel=2
    model_options.Module.padding=2
    model_options.Module.use_bn=args.use_bn   
    model_options.Module.use_structural=args.use_structural
    model_options.Module.activation_structural=args.activation_structural       
    model_options.Module.use_backup_system_structural=args.use_backup_system_structural
    #ae
    model_options.Module.use_bn_decoder=args.use_bn_decoder
    model_options.Module.momentum_bn_decoder=args.momentum_bn_decoder
    model_options.Module.activation_target_decoder=args.activation_target_decoder

    model_options.Module.running_stats_steps= args.running_stats_steps if args.running_stats_steps>0 else 100
    model_options.Module.momentum_bn=args.momentum_bn
    model_options.Module.track_running_stats_bn=args.track_running_stats_bn
    model_options.Module.kernel_size = 3
    model_options.Module.keep_bn_in_eval_after_freeze=args.keep_bn_in_eval_after_freeze

    model_options.Module.normalize_oh=args.normalize_oh
    model_options.Module.projection_layer_oh=args.projection_layer_oh
    model_options.Module.structure_inv_oh = args.structure_inv_oh   
    model_options.Module.use_bn_decoder_oh = args.use_bn_decoder_oh   
    model_options.Module.activate_after_str_oh = args.activate_after_str_oh
    if gating == 'soft_gating_pool':
        model_options.SGNet.lr=args.lr      
        model_options.SGNet.wdecay=args.wdecay
        model_options.SGNet.regime='normal' 
        model_options.SGNet.depth=args.depth   
        model_options.SGNet.padd_input=args.padd_input
        model_options.SGNet.module_type=args.module_type
        model_options.SGNet.single_oh_controller=args.single_oh_controller
        model_options.SGNet.use_single_controller=args.use_single_controller
        model_options.SGNet.seperate_pool_per_layer=args.seperate_pool_per_layer

        model_options.SGNet.num_tasks = args.n_tasks    
        model_options.SGNet.init_ordering_mode = 'random'
        model_options.SGNet.keep_bn_in_eval_after_freeze = args.keep_bn_in_eval_after_freeze

        model = CNNSoftGatedLLDynamic(model_options.SGNet, 
                                model_options.Module, 
                                i_size =i_size, 
                                channels=3,
                                hidden_size=args.hidden_size, 
                                num_classes=n_classes).to(device)
    return model


def get_accs_for_tasks(learner, args:ArgsGenerator, loaders:List[DataLoader], accs_past: List[float]=None, task_agnostic_test: bool=False)-> Tuple[List,List,List]:
    accs=[]        
    Fs = []
    masks=[] 
    task_oh_selection_accs=[]                    
    #make sure we test the same model for each task, since we do batchnorm warm-up, this is needed here
    state_dict=copy.deepcopy(learner.net.state_dict())
    for ti, test_loader in enumerate(loaders):    
        learner.net.load_state_dict(state_dict, strict=True)             
        #dont warm up batch norm on the last task, as it just trained on it anyways   
        # no warm up for the last loader, if no batch norm is used, if gating=='experts' 
        steps_bn_warmup = 200*int(args.use_bn)*int(args.gating!='experts')*(1-(int(ti==(len(loaders)-1))*int(not task_agnostic_test)))*(1-int(args.keep_bn_in_eval_after_freeze))
        print('steps_bn_warmup', steps_bn_warmup)
        print(ti)
        acc, info, mask = test_with_bn(learner, test_loader, task_id=ti if not task_agnostic_test else None, bn_warmup_steps=steps_bn_warmup )
        acc = acc#.cpu().item()
        accs.append(acc)
        masks.append(mask)
        if info is not None and len(info['task_head_selection'])>0:
            task_oh_selection_accs.append(sum(info['task_head_selection']==ti)/len(info['task_head_selection']))
        else:
            task_oh_selection_accs.append(1.)
        if accs_past is not None:
            Fs.append(acc-accs_past[ti])
    #         log_wandb({f'F_test_{ti}':acc-loaders[ti]}, prefix='test/')
    learner.net.load_state_dict(state_dict, strict=True)
    return accs,Fs,masks,task_oh_selection_accs
       
def train_on_task(learner:CompositionalDynamicFM, args:ArgsGenerator, train_loader, valid_loader, test_loader, train_loader_unlabeled=None, epochs=400, task_id=None, **kwargs):
    assert task_id is not None
    testloaders = {task_id: test_loader}
    learner.train(train_loader, 
                task_id,                 
                valloader=valid_loader,
                component_update_freq=args.component_update_frequency,
                num_epochs=epochs,
                testloaders=testloaders,
                save_freq=200)

def test_with_bn(learner, test_loader, task_id, bn_warmup_steps=100):          
    #warm up the batchnorms
    model = bn_warmup(learner, test_loader, task_id, bn_warmup_steps)
    acc = learner.evaluate_task(test_loader,task_id, eval_no_update=False)[1]
    return acc, None, None

def bn_warmup(learner, loader:DataLoader, task_id=None, bn_warmup_steps=100):
    was_training=learner.net.training
    learner.net.train()     
    automated_module_addition_before=1#model.args.automated_module_addition
    if bn_warmup_steps>0:   
        for i, (x,_) in enumerate(loader):
            learner.net(x.to(device), task_id)
            if i>=bn_warmup_steps:
                break
    if not was_training:
        learner.net.eval()


def main(args:ArgsGenerator, task_gen:TaskGenerator):    
    if args.task_sequence_test is not None and args.debug:     
        task_gen_test = ctrl.get_stream(args_generator.task_sequence_test, seed=args_generator.stream_seed)

        for i, t in enumerate(task_gen_test):    
            # model.load_state_dict(state_dict_learned)
            print(f'Task {i}, Classes: {t.concepts}')     
            print(t.transformation.trans_descr)
            print(f"Task id {i}")
            dl = create_dataloader_ctrl(task_gen, t, args,0,num_batches=64)
            a = next(iter(dl))
            pyplot.imshow(a[0][0].permute(1,2,0))
            pyplot.savefig('test.png')
            pass
                
    t = task_gen.add_task()   

    ##############################
    #Replay Buffer
    if args.replay_capacity>0: 
        rng = np.random.RandomState(args.seed)
        er_buffer=Buffer(args.replay_capacity,
                        input_shape=t.x_dim,   
                        extra_buffers={"t": torch.LongTensor},
                        rng=rng)
    else:
        er_buffer = None
    ##############################
    model=init_model(args, args.gating, n_classes=t.n_classes.item(),  i_size=t.x_dim[-1]) 
    print(model.components)
    learner = CompositionalDynamicFM(model, lr=args.lr, wdecay=args.wdecay) #, results_dir=results_dir)
             
    try:
        wandb.watch(model)
    except:
        pass 
    n_tasks=args.n_tasks

    test_loaders=[]
    valid_loaders=[]      
    test_accuracies_past = []
    valid_accuracies_past = [] 
    for i in range(n_tasks):        
        print('==='*10)
        print(f'Task train {i}, Classes: {t.concepts}')   
        try:    
            print(t.transformation.trans_descr)   
        except:
            pass
        print('==='*10)
        fim_prev=[]                  
        train_loader_unlabeled = None                                                                                    
        train_loader_current, valid_dataloader, test_loader_current = create_dataloader_ctrl(task_gen, t, args,0,batch_size=args.batch_size, regime='labeled', task_n=i), create_dataloader_ctrl(task_gen, t, args,1,args.batch_size, regime='labeled', shuffle_test=('ood' in args.task_sequence), task_n=i), create_dataloader_ctrl(task_gen, t, args,2,args.batch_size, regime='labeled', shuffle_test=('ood' in args.task_sequence), task_n=i) 
        if args.task_sequence=='s_minus_unsup':
            #try to create a unlabeled loader of the same length as the labeled one
            train_loader_unlabeled = create_dataloader_ctrl(task_gen, t, args,0,num_batches=len(train_loader_current), regime=None, shuffle_test=('ood' in args.task_sequence)) #gets all data, labeled and unlabeled 


        if i>0:
            if args.warmup_bn_bf_training: 
                #warup batchnorms before training on task
                steps_bn_warmup = 200*int(args.use_bn)*int(args.gating!='experts')
                model = bn_warmup(model, train_loader_current, None, steps_bn_warmup)
        if args.running_stats_steps==0:
            model.module_options.running_stats_steps=len(train_loader_current)
        
        train_on_task(learner, args, train_loader_current, valid_dataloader, test_loader_current, train_loader_unlabeled, epochs=args.epochs, anneal=args.anneal, str_anneal=args.str_anneal, task_id=i, reg_factor=args.reg_factor)

        # model_p=copy.deepcopy(model)
        model.last_learned_task = i
        test_acc = learner.evaluate_task(test_loader_current,i, eval_no_update=False)[1] #test(model, None, test_loader_current, model.min_temp, model.min_str_prior_temp, task_id=i if not args.task_agnostic_test else None)[0].cpu().item()
        valid_acc = learner.evaluate_task(valid_dataloader,i, eval_no_update=False)[1] #test(model, None, valid_dataloader, model.min_temp, model.min_str_prior_temp, task_id=i if not args.task_agnostic_test else None)[0].cpu().item()

        test_accuracies_past.append(test_acc)
        valid_accuracies_past.append(valid_acc)
        ####################
        #Logging
        ####################
        #Current accuracy     
        log_wandb({f'test/test_acc_{i}':test_acc}) 
        log_wandb({f'valid/valid_acc_{i}':valid_acc})
        n_modules = torch.tensor(learner.net.num_components).cpu().numpy()     
        log_wandb({'total_modules': np.sum(np.array(n_modules))}, prefix='model/')
        ####################
        test_loaders.append(test_loader_current)
        valid_loaders.append(valid_dataloader)

        # accs, _, _,_ = get_accs_for_tasks(learner, args, test_loaders, task_agnostic_test=args.task_agnostic_test)
        
        #Get new task
        try:
            t = task_gen.add_task()
        except:
            print(i)
            break                  
        #fix previous output head          
        if isinstance(model.decoder, nn.ModuleList):   
            if hasattr(model.decoder[i],'weight'):
                print(torch.sum(model.decoder[i].weight))
    if isinstance(model.decoder, nn.ModuleList):
        for d in model.decoder:
            if hasattr(d,'weight'):   
                print(torch.sum(d.weight))
    #########################
    accs_test, Fs, masks_test, task_selection_accs = get_accs_for_tasks(learner, args, test_loaders, test_accuracies_past, task_agnostic_test=args.task_agnostic_test)
    for ti, (acc, Frg, task_selection_acc) in enumerate(zip(accs_test, Fs, task_selection_accs)):
        log_wandb({f'test_acc_{ti}':acc}, prefix='test/')
        #Forgetting (test)
        log_wandb({f'F_test_{ti}':Frg}, prefix='test/')           
        #Task selection accuracy (only relevant in not ask id is geven at test time) (test)
        log_wandb({f'Task_selection_acc{ti}':task_selection_acc}, prefix='test/')    
    ####################
    #Average accuracy (test) at the end of the sequence 
    print(accs_test)
    print(task_selection_accs)
    log_wandb({"mean_test_acc":np.mean(accs_test)})#, prefix='test/')
    #Average forgetting (test)
    log_wandb({"mean_test_F":np.mean(Fs)})#, prefix='test/')
    ####################
    #Masks / Module usage
    if len(masks_test)>0 and args.gating=='locspec':         
        fig, axs = pyplot.subplots(1,len(test_loaders),figsize=(15,15))
        for i, ax in enumerate(axs):
            im = ax.imshow(masks_test[i].cpu().T, cmap='Blues')
            # ax.set_yticks([0,1,2])  
            # ax.set_yticklabels(['module 0', 'module 1', 'module 2'],rotation=45,fontsize=15)
            ax.set_xticks([0,1,2,3])
        log_wandb({f"module usage": wandb.Image(fig)})
    ####################
    accs_valid, Fs_valid, _, task_selection_accs = get_accs_for_tasks(learner, args, valid_loaders, valid_accuracies_past, task_agnostic_test=args.task_agnostic_test)        
    for ti, (acc, Frg, task_selection_acc) in enumerate(zip(accs_valid, Fs_valid, task_selection_accs)):
        log_wandb({f'valid_acc_{ti}':acc}, prefix='valid/')
        #Forgetting (valid)
        log_wandb({f'F_valid_{ti}':Frg}, prefix='valid/') 
        #Task selection accuracy (only relevant in not ask id is geven at test time)(valid)
        log_wandb({f'Task_selection_acc{ti}':task_selection_acc}, prefix='valid/')        
    ####################
    #Average accuracy (valid) at the end of the sequence 
    log_wandb({"mean_valid_acc":np.mean(accs_valid)})#, prefix='valid/')
    #Average forgetting (valid)
    log_wandb({"mean_valid_F":np.mean(Fs_valid)})#, prefix='test/')
    #Metric to maximize in wandb sweeps      
    log_wandb({"valid_acc_module_usage": np.mean(accs_valid)/np.sum(np.array(n_modules))})#, prefix='test/')
    ####################
    #Transfer: need to train a seperate expert model on each task from scratch (we can also calculate it from wandb taking the expert baseline's accuracies)
    if args.measure_transfer:
        for i,t in enumerate(task_gen.task_pool):                                                                           
            train_loader_current, valid_dataloader, test_loader_current = create_dataloader_ctrl(task_gen, t, args,0,args.batch_size, regime='labeled', task_n=i), create_dataloader_ctrl(task_gen, t, args,1,args.batch_size, regime='labeled', shuffle_test=('ood' in args.task_sequence), task_n=i), create_dataloader_ctrl(task_gen, t, args,2,args.batch_size, regime='labeled', shuffle_test=('ood' in args.task_sequence), task_n=i) 
            expert = model=init_model(args, 'experts', n_classes=t.n_classes.item(),  i_size=t.x_dim[-1]) 
            expert=train_on_task(expert, args, train_loader_current, valid_dataloader, test_loader_current, epochs=args.epochs, anneal=0, str_anneal=0, task_id=0)
            test_acc_expert = test(expert, None, test_loader_current, model.min_temp, model.min_str_prior_temp, task_id=0)[0].cpu().item()
            log_wandb({"Transfer":accs_test[i]-test_acc_expert})#, prefix='test/')
    
         
    if args.task_sequence_test is not None and 'ood' in args.task_sequence: # and args.debug:
        state_dict_learned=model.state_dict()
        task_gen_test = ctrl.get_stream(args_generator.task_sequence_test, seed=args_generator.stream_seed)
        classes=[]
        transformations=[]
        task_id = -1
        accuracies=[]    
        accuracies_valid=[]
        masks_test=[]
            
        for i, t in enumerate(task_gen_test):    
            model.load_state_dict(state_dict_learned)
            classes_name = str([int(s) for s in str(t.concepts).split() if s.isdigit()])
            if len(classes)==0 or classes[-1]!=classes_name:
                #task witched
                task_id+=1
            print(f'Task {i}, Classes: {t.concepts}')     
            print(t.transformation.trans_descr)
            print(f"Task id {task_id}")
            classes.append(classes_name)      
            descr=t.transformation.trans_descr.split('->')[-1]
            name=construct_name_ctrl(descr)
            transformations.append(name)#t.transformation.trans_descr.split('->')[-1])
            loader_valid, loader_test = create_dataloader_ctrl(task_gen, t, args,1, batch_size=args.batch_size, regime='labeled', task_n=i), create_dataloader_ctrl(task_gen, t, args,2, batch_size=args.batch_size, regime='labeled', task_n=i)       
            test_acc, _, mask = test_with_bn(model, None, loader_test, model.min_temp, model.min_str_prior_temp, task_id=task_id if not args.task_agnostic_test else None, bn_warmup_steps=200)
            test_acc=test_acc.cpu().item()
            try:
                masks_test.append(mask.detach())
            except:
                masks_test.append(mask)
            valid_acc = test_with_bn(model, None, loader_valid, model.min_temp, model.min_str_prior_temp, task_id=task_id if not args.task_agnostic_test else None, bn_warmup_steps=100)[0].cpu().item()
            accuracies.append(test_acc)
            accuracies_valid.append(valid_acc)
        
        log_wandb({f"mean_test_ood": np.mean(accuracies)}) 
        log_wandb({f"mean_valid_ood": np.mean(accuracies_valid)})
        array=[]
        indexes = np.unique(transformations, return_index=True)[1]
        unique_transformations = [transformations[index] for index in sorted(indexes)]
        for tr in unique_transformations:
            results_for_transform=[]         
            for i, tr2 in enumerate(transformations):
                if tr==tr2:
                    results_for_transform.append(accuracies[i])
            array.append(results_for_transform)
        ####################
        #Masks / Module usage ood  
        if len(masks_test)>0 and args.gating=='locspec':  
            fig, axs = pyplot.subplots(len(unique_transformations),len(np.unique(classes)),figsize=(20,2*len(unique_transformations)))
            for row, ax_row in enumerate(axs):
                for column, ax in enumerate(ax_row):
                    im = ax.imshow(masks_test[column*len(axs)+row].cpu().T, cmap='Blues')
                    ax.set_title(unique_transformations[row])
                    # ax.set_yticks([0,1,2])  
                    # ax.set_yticklabels(['module 0', 'module 1', 'module 2'],rotation=45,fontsize=15)
                    # ax.set_xticks([0,1,2,3])
            log_wandb({f"ood/module_usage": wandb.Image(fig)})
        ####################
        #create a confusion matrix
        df_cm = pd.DataFrame(array, index = unique_transformations,columns = np.unique(classes))
        fig = pyplot.figure(figsize = (15,15))
        sn.set(font_scale=1.0)
        sn.heatmap(df_cm, annot=True,vmin=0, vmax=1, fmt=".2%")
        #confusion matrix
        log_wandb({f"confusion_matrix": wandb.Image(fig)})
        


if __name__== "__main__":                                  
    parser = ArgumentParser()     
    parser.add_arguments(ArgsGenerator, dest="Global")
    args = parser.parse_args()
    args_generator = args.Global

    pr_name=f'lmc' if args_generator.pr_name is None else args_generator.pr_name
    for i in range(args_generator.n_runs):                
        task_gen = ctrl.get_stream(args_generator.task_sequence_train, seed=args_generator.stream_seed)  
        if args_generator.debug:
            pr_name='test'
        if not args_generator.debug:
            run = wandb.init(project=pr_name, notes=args_generator.wand_notes, settings=wandb.Settings(start_method="fork"))
        if args_generator.regenerate_seed:
            args_generator.generate_seed()
        if not args_generator.debug:      
            wandb.config.update(args_generator, allow_val_change=False)  
        set_seed(manualSeed=args_generator.seed)
        main(args_generator, task_gen)
        if not args_generator.debug:
            run.finish()
