import copy
from dataclasses import dataclass
from typing import List, Optional

import ctrl
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from simple_parsing import ArgumentParser, choice
from torch import nn
from torch.utils.data import DataLoader

import wandb
from Utils.utils import set_seed
from main_transfer import ArgsGenerator, create_dataloader_ctrl
from Methods.models.hat_appr import Appr as Hat
from Methods.models.hat_appr import Hat_Network
from Methods.models.LMC import LMC_net
from Utils.ctrl.ctrl.tasks.task_generator import TaskGenerator
from Utils.logging_utils import log_wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

loss_function = nn.CrossEntropyLoss()

@dataclass
class ArgsGeneratorHat(ArgsGenerator):       
    #HAT parameters
    smax: int = 400 #-
    gating:str=choice("hat", default="hat")#- 
    per_task_bn: bool = 1 #-

def init_model(args:ArgsGeneratorHat, gating='hat', n_classes=10, i_size=28):
    multihead=args.multihead
    from Methods import ModelOptions
    from Methods.models.LMC import LMC_net
    model_options = ModelOptions()     
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
          
    net=Hat_Network((3,i_size),[n_classes]*args.n_tasks, args.hidden_size, use_bn=args.use_bn, per_task_bn=args.per_task_bn).to(device)
    appr=Hat(net,nepochs=args.epochs,lr=args.lr, smax=args.smax, lamb=args.reg_factor, wdecay=args.wdecay)
    return appr

def test(model, test_loader, task_id, *args):
    xtest, ytest = test_loader.dataset.tensors[0].to(device).to(device), test_loader.dataset.tensors[1].to(device).to(device)
    test_loss,test_acc=model.eval(task_id,xtest,ytest)
    print(f'>>> Test on task {task_id} loss={test_loss}, acc={100*test_acc}% <<<')
    return test_acc#, None, None

def bn_warmup(model, loader:DataLoader, task_id=None, bn_warmup_steps=100):
    """ warms up batchnorms by running several forward passes on the model in training mode """
    was_training=model.model.training
    model.model.train()
    if not was_training:
        model.model.eval()
    return model

def test_with_bn(model, classes, test_loader, temp, str_temp, task_id=None, bn_warmup_steps=100):          
    """ test mode with batchnomr warmup """
    model = bn_warmup(model, test_loader, task_id, bn_warmup_steps)  
    return test(model, test_loader, task_id=task_id)

def get_accs_for_tasks(model, args:ArgsGeneratorHat, loaders:List[DataLoader], accs_past: List[float]=None, task_agnostic_test: bool=False):
    accs=[]        
    Fs = []
    masks=[]                
    task_oh_selection_accs=[]                    
    #make sure we test the same model for each task, since we may do batchnorm warm-up, this is needed here
    state_dict=copy.deepcopy(model.model.state_dict())
    for ti, test_loader in enumerate(loaders):    
        model.model.load_state_dict(state_dict, strict=True)             
        #dont warm up batch norm on the last task, as it just trained on it anyways   
        # no warm up for the last loader, if no batch norm is used
        steps_bn_warmup = 200*int(args.use_bn)*int(args.gating!='experts')*(1-(int(ti==(len(loaders)-1))*int(not task_agnostic_test)))*(1-int(args.keep_bn_in_eval_after_freeze))
        #make this explicit here
        if args.keep_bn_in_eval_after_freeze:
            steps_bn_warmup=0
        print('steps_bn_warmup', steps_bn_warmup)
        print(ti)
        acc = test_with_bn(model, None, test_loader, None, None, task_id=ti if not task_agnostic_test else None, bn_warmup_steps=steps_bn_warmup )
        acc = acc
        accs.append(acc)
        task_oh_selection_accs.append(1.)
    #     ####################
        if accs_past is not None:
            Fs.append(acc-accs_past[ti])
    model.model.load_state_dict(state_dict, strict=True)   
    return accs,Fs,masks,task_oh_selection_accs

def train(args:ArgsGeneratorHat, model, task_idx, train_loader_current, test_loader_current, valid_dataloader, train_loader_unlabeled, fim_prev,er_buffer):
    epochs=args.epochs
    
    best_valid_acc, best_model = None, None
    xtrain,ytrain = train_loader_current.dataset.tensors[0].to(device),train_loader_current.dataset.tensors[1].to(device)
    xvalid,yvalid=valid_dataloader.dataset.tensors[0].to(device),valid_dataloader.dataset.tensors[1].to(device)
    model.train(task_idx,xtrain,ytrain,xvalid,yvalid)
    
    test_acc = test(model, test_loader_current, task_idx)
    if best_valid_acc is None:
        valid_acc = test(model, valid_dataloader, task_id=task_idx)
    else:
        valid_acc=best_valid_acc
    return model,test_acc,valid_acc,None

def main(args:ArgsGeneratorHat, task_gen:TaskGenerator):          
    t = task_gen.add_task()  
    model=init_model(args, args.gating, n_classes=t.n_classes.item(),  i_size=t.x_dim[-1]) 
    model.model.to(device)
    
    ##############################
    er_buffer = None
    ##############################
             
    try:
        wandb.watch(model)
    except:
        pass 
    n_tasks=args.n_tasks
    train_loaders=[]
    test_loaders=[]
    valid_loaders=[]      
    test_accuracies_past = []
    valid_accuracies_past = [] 
    fim_prev=[]
    for i in range(n_tasks):                     
        print('==='*10)
        print(f'Task train {i}, Classes: {t.concepts}')   
        try:    
            print(t.transformation.trans_descr)   
        except:
            pass
        print('==='*10)
        train_loader_unlabeled = None                                                                                               
        train_loader_current, valid_dataloader, test_loader_current = create_dataloader_ctrl(task_gen, t, args,0, batch_size=args.batch_size, labeled=True, task_n=i), create_dataloader_ctrl(task_gen, t, args,1,args.batch_size, labeled=True, shuffle_test=('ood' in args.task_sequence), task_n=i), create_dataloader_ctrl(task_gen, t, args,2,args.batch_size, labeled=True, shuffle_test=('ood' in args.task_sequence), task_n=i) 
        if args.regime=='cl':
            
            model,test_acc,valid_acc,fim_prev = train(args,model,i,train_loader_current,test_loader_current,valid_dataloader,train_loader_unlabeled,fim_prev,er_buffer)
            
            test_accuracies_past.append(test_acc)
            valid_accuracies_past.append(valid_acc)
            test_loaders.append(test_loader_current)
            valid_loaders.append(valid_dataloader)
            ####################
            #Logging
            ####################
            #Current accuracy     
            log_wandb({f'test/test_acc_{i}':test_acc})
            log_wandb({f'valid/valid_acc_{i}':valid_acc})
            #Avv acc sofar (A)
            if args.log_avv_acc:
                accs, _, _,_ = get_accs_for_tasks(model, args, test_loaders, task_agnostic_test=args.task_agnostic_test)
                log_wandb({f'test/avv_test_acc_sofar':np.mean(accs+[test_acc])})
                accs_valid, _, _,_ = get_accs_for_tasks(model, args, valid_loaders, task_agnostic_test=args.task_agnostic_test)
                log_wandb({f'test/avv_test_acc_sofar':np.mean(accs_valid+[valid_acc])})
        elif args.regime=='multitask':
                #collect data first
                train_loaders.append(train_loader_current)
                test_loaders.append(test_loader_current)
                valid_loaders.append(valid_dataloader)
        #Model
        ####################
        #Get new task
        try:
            t = task_gen.add_task()
        except:
            print(i)
            break 
        if args.task_sequence=='s_long30' and i==30:
            print(i)
            break 
        #fix previous output head          
        if isinstance(model, LMC_net):
            if isinstance(model.decoder, nn.ModuleList):   
                if hasattr(model.decoder[i],'weight'):
                    print(torch.sum(model.decoder[i].weight))
    if args.regime=='multitask':
        #train
        train_set = torch.utils.data.ConcatDataset([dl.dataset for dl in train_loaders])
        test_set = torch.utils.data.ConcatDataset([dl.dataset for dl in test_loaders])
        valid_set = torch.utils.data.ConcatDataset([dl.dataset for dl in valid_loaders])
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=1)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=1)
        valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=1)
        model,test_acc,valid_acc,_ = train(args,model,0,train_loader,test_loader,valid_loader,None,None,None)     
        test_accuracies_past=None       
        valid_accuracies_past=None

    #########################
    # this is for debugging     
    if isinstance(model, LMC_net):
        if isinstance(model.decoder, nn.ModuleList):
            for d in model.decoder:
                if hasattr(d,'weight'):   
                    print(torch.sum(d.weight))
    #########################
    accs_test, Fs, masks_test, task_selection_accs = get_accs_for_tasks(model, args, test_loaders, test_accuracies_past, task_agnostic_test=args.task_agnostic_test)
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
        pyplot.clf()
        fig, axs = pyplot.subplots(1,len(test_loaders),figsize=(15,4))
        for i, ax in enumerate(axs):
            im = sns.heatmap(F.normalize(masks_test[i].cpu().T, p=1, dim=0), vmin=0, vmax=1, cmap='Blues', cbar=False, ax=ax, xticklabels=[0,1,2,3])
            ax.set_title(f'Task {i}')
            for _, spine in im.spines.items():
                spine.set_visible(True)
        pyplot.setp(axs[:], xlabel=f'layer')
        pyplot.setp(axs[0], ylabel='module')
        log_wandb({f"module usage": wandb.Image(fig)})
        if args.save_figures:
            for i in range(len(masks_test)):
                print(masks_test[i].cpu().T)
            for i in range(len(masks_test)):
                print(F.normalize(masks_test[i].cpu().T, p=1, dim=0))     
            fig.savefig(f'module_selection_{args.task_sequence}.pdf', format='pdf', dpi=300)
    ####################
    accs_valid, Fs_valid, _, task_selection_accs = get_accs_for_tasks(model, args, valid_loaders, valid_accuracies_past, task_agnostic_test=args.task_agnostic_test)        
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
    # log_wandb({"valid_acc_module_usage": np.sqrt((np.mean(accs_valid)**2)-(np.sum(np.array(n_modules)**2)))})#, prefix='test/')
    ####################  
    # 

if __name__== "__main__":         
    parser = ArgumentParser()     
    parser.add_arguments(ArgsGeneratorHat, dest="Global")
    args = parser.parse_args()
    args_generator = args.Global
    dfs=[]
    pr_name=f'lmc' if args_generator.pr_name is None else args_generator.pr_name
    for r in range(args_generator.n_runs):
        if args_generator.regenerate_seed:
            args_generator.generate_seed()             
        task_gen = ctrl.get_stream(args_generator.task_sequence_train, seed=args_generator.stream_seed)  
        if args_generator.debug:
            pr_name='test'
        # if not args_generator.debug:
        run = wandb.init(project=pr_name, notes=args_generator.wand_notes, settings=wandb.Settings(start_method="fork"), reinit=(args_generator.n_runs>1))
        if not args_generator.debug:      
            wandb.config.update(args_generator, allow_val_change=False)  
        set_seed(manualSeed=args_generator.seed)
        df= main(args_generator, task_gen)
        if df is not None:
            dfs.append(df)
        if not args_generator.debug:
            if not r==(args_generator.n_runs-1):
                try:
                    run.finish()
                except:
                    pass
