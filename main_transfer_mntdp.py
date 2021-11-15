
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List

import ctrl
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from matplotlib import pyplot
from simple_parsing import ArgumentParser, choice
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

import wandb
from main_transfer import (ArgsGenerator, construct_name_ctrl,
                           create_dataloader_ctrl, get_accs_for_tasks,
                           loss_function, plot_confusion, set_seed)
from Methods.models.cnn_independent_experts import ExpertMixture
from Methods.models.LMC import LMC_net
from Methods.models.mntdp import MNTDP_net
from Utils.logging_utils import log_wandb
from Utils.utils import cosine_rampdown

device = 'cuda' if torch.cuda.is_available() else 'cpu'
@dataclass#(eq=True, frozen=False)
class ArgsGenerator(ArgsGenerator):            
    gating: str = choice('experts', 'locspec', 'MNTDP', default='MNTDP')
    
    n_runs:int = 1 # - 
    optmize_structure_only_free_modules: int = 1

    log_avv_acc: int = 0 # if 'True' calculates the average accuracy over tasks sofar after each task
    measure_transfer: int = 0 # if 'True' calculates the transfer by training a seperate expert in isolation on each task at the end of the sequence
    wdecay: float = 1e-4 #weight decay [0,1e-4, 1e-5]
    lr: float = 1e-3 # learning rate             
    copy_batchstats: int = 1 #if 'True' copies batchstats of new tasks into old (reused) modules
    keep_bn_in_eval_after_freeze: bool = 1 # -

    #if 'True' uses entropy to select path with the lowest walue in case of task agnostic testing
    entropy_task_inf: int = 0

    searchspace: str = choice('topdown', 'bottomup', default='topdown') #-

    def generate_random_args(self):
        super.generate_random_args()
    
def create_mask(mask, label):   
            max_dim = max(list(map(lambda x: x.size(0), mask)))
            mask = list(map(lambda x: x[:,:].mean(1), mask))
            return list(map(lambda x: torch.cat((x.cpu(),torch.zeros((max_dim-x.size(0))))) if x.size(0)<max_dim else x.cpu(), mask))   

def init_model(args:ArgsGenerator, gating='MNTDP', n_classes=10, n_modules=1, multihead='usual', i_size=28):
    from Methods import ModelOptions
    model_options = ModelOptions()
    model_options.Module.use_backup_system=args.use_backup_system
    model_options.Module.structure_inv=args.structure_inv
    model_options.Module.maxpool_kernel=2
    model_options.Module.padding=2
    model_options.Module.use_bn=args.use_bn   
    model_options.Module.keep_bn_in_eval_after_freeze=args.keep_bn_in_eval_after_freeze   

    model_options.Module.activation_structural=args.activation_structural
    model_options.Module.use_backup_system_structural=args.use_backup_system_structural
    
    model_options.Module.use_bn_decoder=args.use_bn_decoder
    model_options.Module.momentum_bn_decoder=args.momentum_bn_decoder
    model_options.Module.activation_target_decoder=args.activation_target_decoder

    model_options.Module.running_stats_steps=args.running_stats_steps
    model_options.Module.momentum_bn=args.momentum_bn  
    model_options.Module.track_running_stats_bn=args.track_running_stats_bn
    model_options.Module.kernel_size = 3
    if gating=='MNTDP':
        model_options.MNTDP.lr=args.lr     
        model_options.MNTDP.wdecay=args.wdecay
        model_options.MNTDP.regime='normal' 
        model_options.MNTDP.depth=args.depth   
        model_options.MNTDP.multihead=multihead 
        model_options.MNTDP.net_arch=args.net_arch
        model_options.MNTDP.searchspace=args.searchspace
        model_options.MNTDP.module_type=args.module_type
        model_options.MNTDP.entropy_task_inf=args.entropy_task_inf
        model = MNTDP_net(model_options.MNTDP,   
                                model_options.Module, 
                                i_size =i_size, 
                                channels=3,
                                hidden_size=args.hidden_size, 
                                num_classes=n_classes).to(device)
        return model
    else:
        raise NotImplementedError
       
def test(model, classes, test_loader, temp, task_id=None):
    model.eval()
    result = defaultdict(lambda: 0)
    acc_test = 0                   
    mask = []      
    task_head_selection=[]
    for i, (x,y) in enumerate(test_loader):
        i+=1  
        x,y = x.to(device), y.to(device)           
        forward_out = model(x, inner_loop=False, task_id=task_id, temp=temp)
        logit = forward_out.logit    
        logit = logit.squeeze()
        acc_test += torch.sum(logit.max(1)[1] == y).float()/len(y)
        if task_id is None and 'selected_decoder' in forward_out.info.keys():
            task_head_selection.append(forward_out.info['selected_decoder'])
        if isinstance(model, LMC_net):
            mask.append(torch.stack(create_mask(forward_out.mask, 0)))
            if classes is not None:     
                dev_mask = list(map(lambda x: x.T.detach().cpu().numpy().mean(0),  forward_out.info['deviation_mask']))
                str_loss_per_module = list(map(lambda x: x.T.detach().cpu().numpy().mean(0), forward_out.mask_bf_act))
                z_score_per_module = list(map(lambda x: x.T.detach().cpu().numpy().mean(0), forward_out.info['outlier_signal']))
                for l, ms in enumerate(dev_mask):
                    for m, v in enumerate(ms):   
                        result['deviation_mask/'+f'l_{l}_m_{m}'] += (v - result['deviation_mask/'+f'l_{l}_m_{m}']) /i 
                        result['loss_str/'+f'l_{l}_m_{m}'] += (str_loss_per_module[l][m] - result['loss_str/'+f'l_{l}_m_{m}'])/i                        
                        result['z_score/'+f'l_{l}_m_{m}'] += (z_score_per_module[l][m] - result['z_score/'+f'l_{l}_m_{m}'])/i
    result['task_head_selection']=np.array(task_head_selection)
    if len(mask)>0:
        mask=torch.stack(mask).mean(0)     
    return acc_test/len(test_loader), result, mask

def train_on_task(model:nn.Module, args:ArgsGenerator, train_loader, valid_loader, test_loader, epochs=400, temp=1, anneal=False, task_id=None, epochs_str_only=0, str_only=False, classes=range(10), ewc=0., fim=None, train_str=True, reg_factor=1, patience=0):
    
    current_epoch=0
    #     for e in range(current_epoch, epochs): 
    e=0
    n_modules_model = copy.deepcopy(model.n_modules)
    best_model = None #copy.deepcopy(model.state_dict())
    best_val=0.
    epochs_overfitting=0
    _epochs_str_only = 0
    while e<epochs: 
        len_loader = len(train_loader)      
        loader = train_loader
        model.train()
        acc=0
        reg = 0            
        for batch in loader:
            x,y = batch[0].to(device), batch[1].to(device)

            model.zero_grad()                            
            temp_e = torch.tensor(temp) if not anneal else torch.tensor(temp) * cosine_rampdown(e, epochs+10)
            forward_out = model(x, inner_loop=False, task_id=task_id, temp=temp_e, record_stats=True)

            if not isinstance(model, ExpertMixture):     
                if torch.sum(model.n_modules) > torch.sum(n_modules_model):      
                    best_model=None        
                    best_val=0.     
                    if not model.n_modules[-1] > n_modules_model[-1]:
                        #if it was added not on the last layer
                        _epochs_str_only = e+epochs_str_only #max(_epochs_str_only, e+epochs_str_only)
                        epochs=max(epochs, _epochs_str_only+int(model.args.projection_phase_length/len_loader+10))
                        # print(epochs)
                    else:
                        #module was added on the last layer
                        epochs=max(epochs, e+10) 
                n_modules_model = copy.deepcopy(model.n_modules)
            logit = forward_out.logit
            logit=logit.squeeze()                
            logit = logit[:len(y)]
            if not str_only and e>=_epochs_str_only:
                outer_loss = loss_function(logit, y)
                # print(outer_loss)
            else:
                outer_loss = torch.tensor(0.).to(device)

            if forward_out.regularizer is not None and train_str:
                regularizer = forward_out.regularizer #/ (model.depth * model.n_modules)
                reg+=regularizer.detach()
                outer_loss+= reg_factor*regularizer    
            
            if outer_loss.requires_grad:
                outer_loss.backward()
                model.optimizer.step()     
                if model.optimizer_structure is not None:     
                    model.optimizer_structure.step()
            acc += torch.sum(logit.max(1)[1] == y).float()/len(y)
        print('train acc: ',acc/len_loader, 'epoch: ',e, 'reg: ', reg/len_loader)

        
        # keep track of the best model as measured on the validation set
        #############################################
        if args.keep_best_model:
            if e>=_epochs_str_only:  
                validate=False
                if hasattr(model, 'projection_phase'):
                    if not model.projection_phase: #should not be in ht eprojection phase # and patience>0:
                        validate=True
                else:
                    validate=True
                if validate:
                    model.eval()  
                    acc_valid, _, _ = test(model, classes, valid_loader, temp=temp_e, task_id=task_id) 
                    log_wandb({f'task_{task_id}/valid_acc':acc_valid})
                    if best_val < acc_valid:
                        epochs_overfitting = 0
                        best_val = acc_valid
                        best_model = copy.deepcopy(model.state_dict())
                
        #############################################

        if e %5 == 0:
            model.eval()         
            acc_test, result, _ = test(model, classes, test_loader, temp=temp_e, task_id=task_id)           
            # acc_test_out, result_out = test(classes_out,test_loader_out, temp=temp_e)
            log_wandb(result, prefix=f'result_{task_id}/')    
            # log_wandb(result_out)   
            # log_wandb(dict(filter(lambda v: ('_buffer' in v[0]), model.state_dict().items())), prefix='module_stats/')  
            print('test acc: ', acc_test, ' epoch ', e)
            log_wandb({f'task_{task_id}/test_acc':acc_test})
        e+=1  
    current_epoch = e
    if best_model is not None:          
        if args.use_backup_system and args.gating=='locspec':
            output_heads = [head.out_features for head in model.decoder]
            model = init_model(args, args.gating, n_classes=output_heads[0], multihead='usual', i_size=model.i_size) 
            #add missing output heads
            if len(output_heads)>1:
                for head_classes in output_heads[1:]:
                    model.add_output_head(head_classes)
        model.load_state_dict(best_model, strict=True)
    return model

def bn_warmup(model, task_id, test_loader, bn_warmup_steps, **kwargs):
    model.train()
    if bn_warmup_steps>0:   
        for i, (x,y) in enumerate(test_loader):
            model(x.to(device), record_stats=False, task_id=task_id, inner_loop=False, **kwargs)
            if i>=bn_warmup_steps:
                break
    return model

def test_with_bn(model, classes, test_loader, temp, task_id=None, bn_warmup_steps=100):
    model.train()                  
    automated_module_addition_before=1#model.args.automated_module_addition
    model.args.automated_module_addition=0
    #warm up the batchnorms
    model = bn_warmup(model, task_id, test_loader, bn_warmup_steps)
    model.args.automated_module_addition=automated_module_addition_before
    return test(model, classes, test_loader, temp, task_id)

def get_accs_for_tasks(model:nn.Module, args:ArgsGenerator, loaders:List[DataLoader], accs_past: List[float]=None):
    accs=[]        
    Fs = []
    masks=[]    
    task_oh_selection_accs=[]       
    cf_matrixs=np.zeros((len(model.structure_pool),len(model.structure_pool)))
    #make sure we test the same model for each task, since we do batchnorm warm-up, this is needed here
    state_dict=copy.deepcopy(model.state_dict())
    for ti, test_loader in enumerate(loaders):    
        model.load_state_dict(state_dict, strict=True)             
        #dont warm up batch norm on the last task, as it just trained on it anyways   
        # no warm up for the last loader, if no batch norm is used, if gating=='experts'     
        steps_bn_warmup = 200*max(1-int(ti==(len(loaders)-1)), 1-int(args.copy_batchstats))*int(args.use_bn)*(1-int(args.keep_bn_in_eval_after_freeze))  #*int(args.gating=='locspec')
        if args.warmup_bn_bf_training:
            steps_bn_warmup=200
        print('steps_bn_warmup', steps_bn_warmup)
        print(ti)
        print('structure', model.structure_pool[ti])
        acc, info ,mask = test_with_bn(model, None, test_loader, model.min_temp, task_id=ti if not args.task_agnostic_test else None, bn_warmup_steps=steps_bn_warmup)
        acc = acc.cpu().item()
        accs.append(acc)
        masks.append(mask)
        if info is not None and len(info['task_head_selection'])>0:
            task_oh_selection_accs.append(sum(info['task_head_selection']==ti)/len(info['task_head_selection']))
            cf_matrix=confusion_matrix([ti]*len(info['task_head_selection']), info['task_head_selection'], labels=list(range(len(model.structure_pool))))
            cf_matrixs+=cf_matrix#.append(cf_matrix)
        else:
            task_oh_selection_accs.append(1.)
        if accs_past is not None:
            Fs.append(acc-accs_past[ti])
    model.load_state_dict(state_dict, strict=True)   
    return accs, Fs, masks, task_oh_selection_accs, cf_matrixs

@torch.no_grad()  
def prepare_xy(model:MNTDP_net,args:ArgsGenerator, task_id:int, structure:List, loader:DataLoader):
    steps_bn_warmup = 100*int(args.use_bn)*(1-int(args.keep_bn_in_eval_after_freeze)) #*int(args.gating=='locspec')
    if args.warmup_bn_bf_training:
        steps_bn_warmup=100
    if steps_bn_warmup>0:
        model=bn_warmup(model,task_id,loader, steps_bn_warmup, strucure=structure)
    model.eval()
    loss = correct = 0
    embeddings = []
    targets = []

    for data, target in loader:
        data, target = data.to(device), target.to(device) 
        embeddings.append(model(data, task_id=task_id, structure=structure).info['embeddings'])
        targets.append(target)
    X = np.concatenate([x.cpu() for x in embeddings])
    y = np.concatenate([y.cpu() for y in targets])
    return X, y

def evaluate_knn(model,args,task_id:int,structure:List,train_loader:DataLoader,valid_loader:DataLoader,k=5):
    X, y = prepare_xy(model, args, task_id, structure, train_loader)
    Xt, yt = prepare_xy(model, args, task_id, structure,valid_loader)
    scaler = StandardScaler()
    n=X.shape[0]
    X = scaler.fit_transform(X.reshape(n,-1))
    n=Xt.shape[0]
    Xt = scaler.transform(Xt.reshape(n,-1))
    clf = KNeighborsClassifier(n_neighbors=k, metric='cosine').fit(X, y)
    train_acc = np.mean(clf.predict(Xt) == yt)
    test_acc = np.mean(clf.predict(X) == y)
    return train_acc, test_acc

def main(args:ArgsGenerator, task_gen):    
    t = task_gen.add_task() 
    model_main=init_model(args, args.gating, n_classes=t.n_classes.item(), multihead='usual', i_size=t.x_dim[-1]) 
    
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

        train_loader_current, valid_dataloader, test_loader_current = create_dataloader_ctrl(task_gen, t, args, 0,batch_size=args.batch_size, regime='labeled'), create_dataloader_ctrl(task_gen, t, args,1,args.batch_size, regime='labeled',normalize=args.normalize_dst, shuffle_test=('ood' in args.task_sequence)), create_dataloader_ctrl(task_gen, t, args,2,args.batch_size, regime='labeled', shuffle_test=('ood' in args.task_sequence)) 

        ####################################
        #1. Define the search space
        #1a. Most likeliy structure sofar
        best_structure_knn=None   
        if len(model_main.structure_pool)>1:
            accs_knn = []
            for task_id, structure in enumerate(model_main.structure_pool): 
                _, acc_valid = evaluate_knn(model_main, args, task_id, structure, train_loader_current, valid_dataloader, k=5)
                accs_knn.append(acc_valid)
            best_structure_knn = model_main.structure_pool[np.argmax(accs_knn)]
        
        if args.warmup_bn_bf_training and i>0:
            model_main=bn_warmup(model_main,i,train_loader_current,200,force_structure=True,strucure=best_structure_knn if best_structure_knn is not None else model_main.structure_pool[0])
        #1b. Create the search space
        search_space = model_main.create_search_space(best_structure_knn)
        ##########################################

        #2. Search for the bast model on the given task
        best_valid_acc, best_model, best_structure, best_idx = None, None, None, None
        for _m, (model, structure) in enumerate(search_space):   
            model.optimizer, _ = model.get_optimizers()       
            model=train_on_task(model, args, train_loader_current, valid_dataloader, test_loader_current, epochs=args.epochs, task_id=i, epochs_str_only=0)
            # model_p=copy.deepcopy(model)
            valid_acc = test(model, None, valid_dataloader, None, task_id=i)[0].cpu().item()
            if best_valid_acc is None or best_valid_acc<valid_acc:
                best_valid_acc = copy.deepcopy(valid_acc)
                best_model = copy.deepcopy(model)
                best_structure = copy.deepcopy(structure)
                best_idx=_m

        ##########################################
        #3. Add new modules to the model_main if needed   
        print(f"Best structure selected task{i}", best_structure)
        model_main.add_structure_to_pool(best_structure)
        for l, module_idx in enumerate(best_structure):
            if module_idx>len(model_main.components[l])-1:          
                model_main.add_modules(at_layer=l, state_dict=best_model.components[l][-1].state_dict())
            elif args.copy_batchstats or not model_main.components[l][module_idx].module_learned:
                model_main.components[l][module_idx].load_state_dict(best_model.components[l][-1].state_dict())
            model_main.components[l][module_idx].freeze_module()
        

        assert isinstance(model_main.decoder, Iterable)
        assert isinstance(best_model.decoder, Iterable)
        if len(model_main.decoder)-1==i:
            model_main.decoder[-1].load_state_dict(best_model.decoder[-1].state_dict())
        elif len(model_main.decoder)-1<i:
            model_main.add_output_head(t.n_classes.item(),state_dict=best_model.decoder[-1].state_dict())
        else:
            raise NotImplementedError
        ##########################################
        test_acc = test(model_main, None, test_loader_current, None, task_id=i if not args.task_agnostic_test else None)[0].cpu().item()
        test_accuracies_past.append(test_acc)
        valid_accuracies_past.append(best_valid_acc)
        ####################
        #Logging
        ####################
        pyplot.clf()
        mask = torch.zeros(model_main.depth,max([len(model_main.components[k]) for k in range(len(model_main.components))]))
        for m, z in zip(mask, best_structure):
            m[z]=1
        im = pyplot.imshow(mask.T, cmap='Blues')
        log_wandb({f'selected_path{i}':wandb.Image(im)})
        #Current accuracy     
        log_wandb({f'test/test_acc_{i}':test_acc})
        log_wandb({f'valid/valid_acc_{i}':valid_acc})
        #Avv acc sofar (A)
        if args.log_avv_acc:
            accs, _, _, _, _ = get_accs_for_tasks(model_main, args, test_loaders)
            log_wandb({f'test/avv_test_acc_sofar':np.mean(accs+[test_acc])})    
            accs_valid, _, _, _, _ = get_accs_for_tasks(model_main, args, valid_loaders)
            log_wandb({f'test/avv_test_acc_sofar':np.mean(accs_valid+[valid_acc])})
        #Model
        n_modules = torch.tensor(model_main.n_modules).cpu().numpy()     
        log_wandb({'total_modules': np.sum(np.array(n_modules))}, prefix='model/')
        ####################
        test_loaders.append(test_loader_current)
        valid_loaders.append(valid_dataloader)
        
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
        if isinstance(model_main, LMC_net) or isinstance(model_main, MNTDP_net):
            print(f'Output head sum {torch.sum(model_main.decoder[i].weight)}')
            model_main.fix_oh(i)
        model_main.add_output_head(t.n_classes.item())

    if isinstance(model_main.decoder, Iterable):
        for d in model_main.decoder:
            print(torch.sum(d.weight))

    accs_test, Fs, masks_test, task_selection_accs, cf_matrix_task_selection = get_accs_for_tasks(model_main, args, test_loaders, test_accuracies_past)
    for ti, (acc, Frg, task_selection_acc) in enumerate(zip(accs_test, Fs, task_selection_accs)):
        log_wandb({f'test_acc_{ti}':acc}, prefix='test/')
        #Forgetting (test)
        log_wandb({f'F_test_{ti}':Frg}, prefix='test/')    
        #Task selection accuracy (only relevant in not ask id is geven at test time) (test)
        log_wandb({f'Task_selection_acc{ti}':task_selection_acc}, prefix='test/')  
    #create a confusion matrix
    df_cm = pd.DataFrame(cf_matrix_task_selection).astype(int)
    fig = pyplot.figure(figsize = (5,5))
    sn.set(font_scale=0.5)   
    sn.heatmap(df_cm, annot=True, fmt='d')
    #confusion matrix
    log_wandb({f"confusion_matrix_task_selection": wandb.Image(fig)})  
    matplotlib.rc_file_defaults()
    ####################
    #Average accuracy (test) at the end of the sequence 
    print(accs_test)
    print('Average accuracy (test) at the end of the sequence:',np.mean(accs_test))
    log_wandb({"mean_test_acc":np.mean(accs_test)})#, prefix='test/')
    #Average forgetting (test)
    log_wandb({"mean_test_F":np.mean(Fs)})#, prefix='test/')
    ####################
    #Masks / Module usage
    if len(masks_test)>0 and args.gating=='locspec':
        fig, axs = pyplot.subplots(1,len(test_loaders),figsize=(15,15))
        for i, ax in enumerate(axs):
            im = ax.imshow(masks_test[i].cpu().T, cmap='Blues')
            ax.set_xticks([0,1,2,3])
        log_wandb({f"module usage": wandb.Image(fig)})
    ####################
    accs_valid, Fs_valid, _, _, _ = get_accs_for_tasks(model_main, args, valid_loaders, valid_accuracies_past)        
    for ti, (acc, Frg) in enumerate(zip(accs_valid, Fs_valid)):
        log_wandb({f'valid_acc_{ti}':acc}, prefix='valid/')
        #Forgetting (test)
        log_wandb({f'F_valid_{ti}':Frg}, prefix='valid/')  
    ####################
    #Average accuracy (valid) at the end of the sequence
    print('Average accuracy (valid) at the end of the sequence:',np.mean(accs_valid))
    log_wandb({"mean_valid_acc":np.mean(accs_valid)})#, prefix='valid/')
    #Average forgetting (valid)
    log_wandb({"mean_valid_F":np.mean(Fs_valid)})#, prefix='test/')
    ####################    
         
    if args.task_sequence_test is not None and 'ood' in args.task_sequence: # and args.debug:
        state_dict_learned=model_main.state_dict()
        task_gen_test = ctrl.get_stream(args_generator.task_sequence_test, seed=args_generator.stream_seed)
        classes=[]
        transformations=[]
        task_id = -1
        accuracies=[]    
        accuracies_valid=[]
        masks_test=[]
            
        for i, t in enumerate(task_gen_test):         
            model_main.load_state_dict(state_dict_learned)
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
            test_acc, _, mask = test_with_bn(model_main, None, loader_test, model_main.min_temp, task_id=task_id if not args.task_agnostic_test else None, bn_warmup_steps=200)
            test_acc=test_acc.cpu().item()
            try:
                masks_test.append(mask.detach())
            except:
                masks_test.append(mask)
            valid_acc = test_with_bn(model_main, None, loader_valid, model_main.min_temp, task_id=task_id if not args.task_agnostic_test else None, bn_warmup_steps=100)[0].cpu().item()
            accuracies.append(test_acc)
            accuracies_valid.append(valid_acc)
        array=[]
        array_valid=[]
        indexes = np.unique(transformations, return_index=True)[1]
        unique_transformations = [transformations[index] for index in sorted(indexes)]
        for tr in unique_transformations:
            results_for_transform=[]  
            results_for_transform_valid=[]         
            for i, tr2 in enumerate(transformations):
                if tr==tr2:
                    results_for_transform.append(accuracies[i])
                    results_for_transform_valid.append(accuracies_valid[i])
            array.append(results_for_transform)
            array_valid.append(results_for_transform_valid)
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
        col = np.unique(classes)
        df_cm = pd.DataFrame(array[:len(col)], index = unique_transformations[:len(col)],columns = np.unique(classes))

        log_wandb({f"mean_test_ood": np.mean(array[:len(col)])}) 
        log_wandb({f"mean_valid_ood": np.mean(array_valid[:len(col)])})


        fig = pyplot.figure(figsize = (15,15))
        sn.set(font_scale=1.0)   
        sn.heatmap(df_cm, annot=True,vmin=0, vmax=1, fmt=".2%")
        #confusion matrix
        log_wandb({f"confusion_matrix": wandb.Image(fig)})
        matplotlib.rc_file_defaults()
        return df_cm
    return None


if __name__== "__main__":                                   
    parser = ArgumentParser()     
    parser.add_arguments(ArgsGenerator, dest="Global")
    args = parser.parse_args()
    args_generator = args.Global
    dfs=[]
    pr_name=f'lmc' if args_generator.pr_name is None else args_generator.pr_name
    for r in range(args_generator.n_runs):                                    
        task_gen = ctrl.get_stream(args_generator.task_sequence_train, seed=args_generator.stream_seed)  
        if args_generator.debug:
            pr_name='test'
        # if not args_generator.debug:
        run = wandb.init(project=pr_name, notes=args_generator.wand_notes,settings=wandb.Settings(start_method='fork'), reinit=(args_generator.n_runs>1))
        if args_generator.regenerate_seed:
            args_generator.generate_seed()
        if not args_generator.debug:      
            wandb.config.update(args_generator, allow_val_change=False)  
        set_seed(manualSeed=args_generator.seed)
        df = main(args_generator, task_gen)
        if df is not None:
            dfs.append(df)
        if not args_generator.debug:
            if not r==(args_generator.n_runs-1):
                try:
                    run.finish()
                except:
                    pass
    if len(dfs)>1:
        df_concat = pd.concat(dfs)   
        mean=df_concat.groupby(df_concat.index, sort=False).mean()
        std=df_concat.groupby(df_concat.index, sort=False).std()
        lables=[]
        for i_r in range(mean.shape[0]):
            l_row=[]
            for i_c in range(mean.shape[0]):
                m_formated="{:.1f}".format(100*mean.iloc[i_r,i_c])
                std_formated="{:.1f}".format(100*std.iloc[i_r,i_c])
                pm=u"\u00B1" #'+/-'
                l_row.append(f"{m_formated}\n{pm}{std_formated}")
            lables.append(l_row)
        plot_confusion(mean, wandb_tag='confusion_matrix_final', save_dir=f'confusion_final_{pr_name}_{args_generator.gating}_{args_generator.ewc}_ood.pdf', labels=lables)
