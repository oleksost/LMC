import copy
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional

import ctrl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from simple_parsing import ArgumentParser, choice
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor

import wandb
from Data.Utils import TensorDataset     
from Methods.models.cnn_independent_experts import ExpertMixture
from Methods.models.LMC import LMC_net
from Methods.replay import BalancedBuffer, Buffer
from Utils.ctrl.ctrl.tasks.task_generator import TaskGenerator
from Utils.logging_utils import log_wandb
from Utils.nngeometry.nngeometry.metrics import FIM
from Utils.nngeometry.nngeometry.object import PMatDiag, PVector
from Utils.nngeometry.nngeometry.object.pspace import PMatAbstract
from Utils.utils import construct_name_ctrl, cosine_rampdown, set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

@dataclass#(eq=True, frozen=False)
class ArgsGenerator():       
    ##############################
    #learning process related
    projection_phase_length: int = 5 # length projection phase in epochs           
    fix_layers_below_on_addition: int = 0 #if 1 also layers below the layer where module is added are frozen during the projection phase
    deviation_threshold: float = 4 # random.choice([1,2,4,8]) 
    reg_factor: float = 1. #random.choice([1,2,5,10]) 
    temp: float = 1 #random.choice([0.1,1.,0.01]) 
    anneal: int = 0 #random.choice([0,1])                     
    lr_structural: float = 0.001 #-     
    regime: str = choice('multitask','cl', default='cl') # multitask regime = ofline trainiogn on all tasks (use single head for it, i.e. multihead=none)
    module_init: str = choice('none','mean','previously_active', 'most_likely', default='previously_active') # new module innitialization strategy  
    mask_str_loss: int = 1 # if 'True', the structural components of active modules are weighted as well
    structure_inv: str = choice('linear_no_act', 'pool_only_large_lin_no_act', 'linear_act', 'ae', default='linear_no_act')
    use_backup_system: int = 0 #whether to use backup system for modules (only used for continual-meta experiments)
    use_backup_system_structural: int = 0 #whether to use backup system for the structural component (only used for continual-meta experiments)
    running_stats_steps: int = 1000 #size of the running mean and variance interval for the modules
    str_prior_factor: float = 1. # str_prior_factor
    str_prior_temp: float = 1. #tempretur of structural prior --- the smaller the more batch-level selection is prioritized
    str_anneal: int = 0 # if 'True', anneal structural tempreture
    concat: str = choice('beam', 'sum', default='sum') # -
    beam_width: int = 1 # beam width parameter of beam search strategy (only used if concat='beam')
    catch_outliers_old: bool = 1 # if 'True', only use free (non forzen) module if old modules report outliers
    momentum_bn: float = 0.1 # momentum_bn  
    track_running_stats_bn: int = 1 # if 'True' tracks running stats of the batch nroms
    keep_bn_in_eval_after_freeze: bool = 1 # if 'True', keep batch nroms of the frozen modules in eval mode
    detach_structural:bool = 1 # if 'True' structural loss in not backpropagated into the module's functional component
    init_runingstats_on_addition: int = 1 # if 'True', apply module_init strategy also to running statistics of new module, otherwise running stats are initialized from scratch for new modules
    optmize_structure_only_free_modules: int = 1 # if True the structural component is only optimized for modules which are 'free' - not yet frozen/learned
    ################
    
    #model related
    hidden_size: int = 64 # hidden size of modules
    module_type: str = 'conv' #'resnet_block'          
    gating: str = choice('experts', 'locspec', default='locspec')
    num_modules: int = 1 # Number of modules per layer
    net_arch: int = choice('none', default='none') # -
    activation_structural: str = choice('sigmoid', 'relu', 'tanh', default='relu') #structural activation
    depth: int = 4 #network depth
    use_bn: int = 1 #whether to use batchnorm in the modules (for Alexnet architecture should be 0)
    use_structural: int = 1 # if 0 no structural components are used at all (model becomes nonmodular if num_modules =1 )
    ################
    
    #output module related (output leayer)
    multihead: str = choice('usual', 'gated_linear', 'gated_conv', 'none', default='usual') #multihead type, if 'none' uses single head
    normalize_oh: bool = 0 # -
    projection_layer_oh: bool = 0 # -
    structure_inv_oh: str =choice('ae', 'linear_no_act', 'linear_act', default='linear_no_act') # -
    use_bn_decoder_oh: int = 0 # -
    activate_after_str_oh: int = 0 # -
    init_oh: str = choice('mean', 'none', 'most_likely', default='none') # -
    ################

    #unfreezing of modules
    active_unfreezing: int = 0 #if 'True', modules can be unfrozen
    unfreeze_structural: int = 0 #if 'True', also structural component will be unfrozen whenever a mdule is unfrozen
    treat_unfreezing_as_addition: int = 0 # if 'True' performs projection phase on unfreesing as well
    ################
    
    #########
    # ae          
    use_bn_decoder:int = 1 #wether to use batchnorm in the decoder of structural (ae)
    momentum_bn_decoder: float = 0.1 #momentum of the structural decoder
    activation_target_decoder: str = choice('sigmoid', 'relu', 'tanh', 'None', default='None') #activation for the decoders' target (output of previous layer)

    task_sequence_train: Optional[str]=None
    task_sequence_test: Optional[str]=None

    ##############################
    #Optimization
    wdecay: float = 1e-4 #weight decay [0,1e-4, 1e-5]
    lr: float = 1e-3 # learning rate
    ##############################
    #Logging
    pr_name: Optional[str]=None #wandb project name
    wand_notes: str = '' #wandb notes
    log_avv_acc: int = 0 # if 'True' calculates the average accuracy over tasks sofar after each task
    ##############################

    ##############################  
    #Data generation                             
    stream_seed: int = 180 # seed of the ctrl stream
    n_tasks: int = 6 # n_tasks       
    task_sequence: str = choice('s_minus', 's_pl','s_plus', 's_mnist_svhn', 's_pnp_comp', 's_pnp_tr', 's_pnp', 's_in', 's_out', 's_long', 's_long30', 's_ood', default='s_minus') #task sequence from ctrl  
    batch_size: int = 64 #bacths sizes
    normalize_data: int=0 #if 1 apply nromalization transform to data
    ##############################
    #Hparams tuning & training
    regenerate_seed: int = 0 #wether to regenerate seet ad each run
    n_runs:int = 1 # - 
    seed: int = 180 #seed
    debug: int = 0 #debug mode
    early_stop_complete: bool = 0 # it 'True' resets best model to None every time a new module was added during learning a task
    warmup_bn_bf_training: int = 0 # -
    task_agnostic_test: int = 0 #if 'True' (1) no task_id is given at test time
    keep_best_model: int = 1 # if 'True' keeps bestvalidation model
    num_epochs_added_after_projection:int = 10 #least number training epochs run after a projection phase 
    epochs_str_only_after_addition: int = 0 # number of epochs after module addition during which only the structural loss is used
    epochs_structure_only_at_start: int = 0 # number of epochs during which only structural loss will be used ad the beginning of training on each task
    epochs: int = 20 # number of epochs to train on each task
    shuffle_test: int = 0 #if 'True', shuffls test and validation sets (might give better performance when using batchnorm warmup)
    ##############################
    #EWC
    ewc_online: bool = 0. # online LMC consolidates the FIMs
    ewc: float = 0. #if >0 EWC regularization is used
    ##############################
    #Replay
    replay_capacity: int = 0 #if > 0 uses replay buffer, if -1 calculates the replay size automatically to match the max size of the LMC in case of linear growth
    ##############################

    #ablation
    no_projection_phase:int = 0 #-
    save_figures: int = 0 #-
    n_heads_decoder: int = 1 #-
    ##############################
    
    def __post_init__(self):   
        if self.task_sequence == 's_ood':
            self.task_sequence_train ='s_ood_train'
            self.task_sequence_test ='s_ood_test'
        else:
            self.task_sequence_train = self.task_sequence

        if not self.use_backup_system:
            self.use_backup_system_structural=0
        if self.debug:  
            self.epochs=1      
            self.regenerate_seed=0
            self.generate_args=0
            self.hidden_size=8         

    def generate_seed(self):
        self.seed=random.randint(1, 2021)

loss_function = nn.CrossEntropyLoss()

def create_dataloader_ctrl(task_gen:TaskGenerator, task, args:ArgsGenerator, split=0, 
                            batch_size=64, num_batches=None, labeled=True, normalize=False, **kwargs):
    single_head=(args.multihead=='none')
    normalize=args.normalize_data
    y = task.get_labels(split=split, prop=0)
    x = task.get_data(split=split)
    if labeled:
        idx = torch.where(y!=-1)
        y = y[idx]
        x = x[idx]
    if num_batches is not None:       
        batch_size=int(len(y)//num_batches)
    transform=None
    
    if x.shape[1]<task.x_dim[-1] and args.task_sequence=='s_mnist_svhn':
        transform = transforms.Compose([ transforms.ToPILImage(),transforms.Resize((task.x_dim[-1],task.x_dim[-1])), ToTensor()])
    if normalize:
        if min(task.statistics['mean'])>0 and 'mnist' in str(task.concepts) and 'ood' in args.task_sequence: 
            #if no dimention is completely zeros we use statistics of the complete MNIST dataset (for simplisity) - will be used for task sequence s_ood_bkgrnd_white_digits
            if transform is None:
                transform = transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))
            else:
                transform.append(transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)))
        else:
            #we leave dimentions with only 0s to stay only 0s
            if transform is None:    
                transform = transforms.Normalize(task.statistics['mean'], [s if s>0 else s+1 for s in task.statistics['std']])
            else:
                transform.append(transforms.Normalize(task.statistics['mean'], [s if s>0 else s+1  for s in task.statistics['std']]))    

    if single_head:
        # adjust class labels for the single head regime
        adjust_y=0
        for t,old_t in enumerate(task_gen.task_pool):
            if str(old_t.concepts)==str(task.concepts):
                break
            else:
                adjust_y+=old_t.info()['n_classes'][0]           
        y+=adjust_y 
    if args.shuffle_test and split!=0:
        idx = torch.randperm(x.size(0))
        x=x[idx]
        y=y[idx]
    
    dataset = TensorDataset([x,y], transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split==0)) #or shuffle_test))

def init_model(args:ArgsGenerator, gating='locspec', n_classes=10, i_size=28):
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
    if gating=='locspec':
        model_options.Module.detach_structural=args.detach_structural
        model_options.LMC.no_projection_phase=args.no_projection_phase
        model_options.LMC.init_stats=args.init_runingstats_on_addition
        model_options.LMC.regime='normal'
        model_options.LMC.lr=args.lr
        model_options.LMC.wdecay=args.wdecay
        model_options.LMC.depth=args.depth  
        model_options.LMC.lr_structural=args.lr_structural   

        model_options.LMC.net_arch=args.net_arch
        model_options.LMC.n_modules=args.num_modules       
        model_options.LMC.temp=args.temp
        model_options.LMC.str_prior_temp=args.str_prior_temp
        model_options.Module.n_heads_decoder=args.n_heads_decoder
        model_options.LMC.fix_layers_below_on_addition=args.fix_layers_below_on_addition
        model_options.LMC.module_type=args.module_type
        model_options.LMC.str_prior_factor=args.str_prior_factor
        model_options.LMC.concat=args.concat
        model_options.LMC.beam_width=args.beam_width
        model_options.LMC.catch_outliers_old=args.catch_outliers_old
        model_options.LMC.module_init=args.module_init
        model_options.LMC.multihead=multihead
        model_options.LMC.deviation_threshold=args.deviation_threshold
        model_options.LMC.mask_str_loss=args.mask_str_loss
        model_options.LMC.projection_phase_length=args.projection_phase_length                  
        model_options.LMC.optmize_structure_only_free_modules=args.optmize_structure_only_free_modules
        model_options.LMC.automated_module_addition=1
        model_options.LMC.active_unfreezing=args.active_unfreezing
        model_options.LMC.unfreeze_structural=args.unfreeze_structural
        model_options.LMC.treat_unfreezing_as_addition=args.treat_unfreezing_as_addition

        model = LMC_net(model_options.LMC, 
                                    model_options.Module, 
                                    i_size =i_size, 
                                    channels=3,
                                    hidden_size=args.hidden_size, 
                                    num_classes=n_classes).to(device)     
    elif gating == 'experts':    
        model_options.Experts.lr=args.lr     
        model_options.Experts.wdecay=args.wdecay
        model_options.Experts.regime='normal' 
        model_options.Experts.depth=args.depth   
        model_options.Experts.net_arch=args.net_arch
        model_options.Experts.n_modules=args.num_modules
        model_options.Experts.module_type=args.module_type
        model = ExpertMixture(model_options.Experts, 
                                model_options.Module, 
                                i_size =i_size, 
                                channels=3,
                                hidden_size=args.hidden_size, 
                                num_classes=n_classes).to(device)
    
    return model

def test(model, classes, test_loader, temp, str_prior_temp, task_id=None):
    model.eval()
    result = defaultdict(lambda: 0)
    acc_test = 0                   
    mask = []   
    task_head_selection=[]
    for i, (x,y) in enumerate(test_loader):
        i+=1  
        x,y = x.to(device), y.to(device)           
        forward_out = model(x, inner_loop=False, task_id=task_id, temp=temp, str_prior_temp=str_prior_temp)
        logit = forward_out.logit    
        logit = logit.squeeze()
        if task_id is None:
            task_head_selection.append(forward_out.info['selected_decoder'])
        acc_test += torch.sum(logit.max(1)[1] == y).float()/len(y)
        if isinstance(model, LMC_net):
            mask.append(forward_out.mask)
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

def consolidate_fim(fim_previous, fim_new, task):
    # consolidate the fim_new into fim_previous in place
    if isinstance(fim_new, PMatDiag):
        fim_previous.data = (
            copy.deepcopy(fim_new.data) + fim_previous.data * (task)
        ) / (task + 1)
    else:
        raise NotImplemented
    return fim_previous  

def train_on_task(model:nn.Module, args:ArgsGenerator, train_loader, valid_loader, test_loader, epochs=400, 
                    str_temp=1, anneal=False, str_anneal=False,  task_id=None, str_only=False, classes=range(10), fim=None, fims=[], train_str=True, reg_factor=1, patience=0, er_buffer:Buffer=None):
    if isinstance(fims, PMatAbstract):
        fim=fims
    temp=args.temp
    str_temp=args.str_prior_temp
    # these are set to 0 in all experiments
    epochs_str_only_after_addition =(int((task_id>0))*args.epochs_str_only_after_addition)
    epochs_structure_only_at_start=(int((task_id>0))*args.epochs_structure_only_at_start)
    ############
    _epochs_str_only_after_addition = 0
    ewc=args.ewc
    s=None
    if ewc>0:
        anchor = PVector.from_model(model.components).clone().detach()
    
    e=0
    n_modules_model = copy.deepcopy(model.n_modules)
    best_model = None
    best_val=0. 
    while e<epochs: 
        len_loader = len(train_loader)    
        loader = train_loader
        model.train()
        acc=0
        reg = 0               
        for bi, batch in enumerate(loader):
            x,y = batch[0].to(device), batch[1].to(device)
            ##################################################
            # Add to ER Buffer only during the first epoch
            if er_buffer is not None and e==0:
                er_buffer.add_reservoir({"x": x, "y": y, "t": task_id})
            ##################################################
            model.zero_grad()  
            temp_e = torch.tensor(temp) if not anneal else torch.tensor(temp) * cosine_rampdown(e, epochs+10)
            str_temp_e = torch.tensor(str_temp) if not str_anneal else torch.tensor(str_temp) * cosine_rampdown(e, epochs+10)

            forward_out = model(x, inner_loop=False, task_id=task_id, temp=temp_e, str_prior_temp=str_temp_e, record_running_stats=True, detach_head=(e<_epochs_str_only_after_addition and e<epochs_structure_only_at_start), s=s)

            if not any(map(lambda x: isinstance(model, x), [ExpertMixture])):     
                if torch.sum(model.n_modules) > torch.sum(n_modules_model):     
                    #A new module was added at this iteration 
                    if args.early_stop_complete:
                        #discard the best model found sofar
                        best_model=None        
                        best_val=0.      
                    if not model.n_modules[-1] > n_modules_model[-1]:
                        #if it was added not on the last layer
                        _epochs_str_only_after_addition = e+epochs_str_only_after_addition # use only structural loss for epochs_str_only_after_addition epochs
                        #model.args.projection_phase_length/len_loader = projection phase length in epochs
                        #train at least for args.projection_phase_length +  args.num_epochs_added_after_projection epochs more
                        epochs=max(epochs, e+int(model.args.projection_phase_length/len_loader+args.num_epochs_added_after_projection))
                        
                    else:
                        #module was added on the last layer = no projection phase
                        #train at elast for 10 epochs more
                        epochs=max(epochs, e+10) 
                n_modules_model = copy.deepcopy(model.n_modules)
            
            logit = forward_out.logit
            logit=logit.squeeze()                
            logit = logit[:len(y)]
            outer_loss = loss_function(logit, y)

            if forward_out.regularizer is not None and train_str and not torch.isnan(forward_out.regularizer):
                regularizer = forward_out.regularizer
                reg+=regularizer.detach()
                # assert not torch.isnan(regularizer)
                outer_loss+= reg_factor*regularizer   
            # print(forward_out.regularizer)
            ##############
            ###  EWC  ##  
            if ewc>0:
                if fim is not None:
                    v_current = PVector.from_model(model.components)
                    regularizer=(fim.vTMv(v_current - anchor))
                    outer_loss += ewc*regularizer
                    reg+=regularizer.detach()
                elif len(fims)>0:
                    regularizer=0
                    v_current = PVector.from_model(model.components)
                    for f in fims:
                        regularizer+=(f.vTMv(v_current - anchor))                    
                    outer_loss += ewc*regularizer
                    reg+=regularizer.detach()
            ##############
            ###  REPLAY ##   
            if task_id > 0 and er_buffer:
                x_buffer = []
                y_buffer = []
                if args.multihead=='none':
                    #single head
                    for past_t in range(task_id):
                        replay_bs=x.size(0)
                        b_samples = er_buffer.sample(replay_bs,only_task=past_t)
                        x_buffer.append(b_samples['x'])
                        y_buffer.append(b_samples['y'])
                    x_buffer=torch.cat(x_buffer)
                    y_buffer=torch.cat(y_buffer)

                    b_logits = forward_out = model(x_buffer.to(device), inner_loop=False, task_id=past_t, temp=temp_e, str_prior_temp=str_temp_e, record_running_stats=True, 
                                                        detach_head=(e<_epochs_str_only_after_addition and e<epochs_structure_only_at_start)).logit
                    loss_replay = loss_function(b_logits, y_buffer.to(device))
                    outer_loss += loss_replay
                else:
                    for past_t in range(task_id):
                        replay_bs=x.size(0)
                        b_samples = er_buffer.sample(replay_bs,only_task=past_t)
                        b_logits = forward_out = model(b_samples['x'].to(device), inner_loop=False, task_id=past_t, temp=temp_e, str_prior_temp=str_temp_e, record_running_stats=True, 
                                                            detach_head=(e<_epochs_str_only_after_addition and e<epochs_structure_only_at_start)).logit
                        loss_replay = loss_function(b_logits, b_samples["y"].to(device))
                        outer_loss += loss_replay
                    outer_loss/=task_id+1
            ##############
            if outer_loss.requires_grad:
                outer_loss.backward()
                model.optimizer.step()     
                if model.optimizer_structure is not None:     
                    model.optimizer_structure.step()


            acc += torch.sum(logit.max(1)[1] == y).float()/len(y)
        print('train acc: ',acc/len_loader, 'epoch: ',e, 'reg: ', reg/len_loader)
        
        # keep track of the best model as measured on the validation set
        if args.keep_best_model:
            if e>=_epochs_str_only_after_addition:  
                validate=False
                if hasattr(model, 'projection_phase'):
                    if not model.projection_phase: 
                        #should not be in the eprojection phase when validating
                        validate=True
                else:
                    validate=True
                if validate:
                    model.eval()  
                    acc_valid, _, _ = test(model, classes, valid_loader, temp=temp_e, str_prior_temp=str_temp_e, task_id=task_id) 
                    if best_val < acc_valid:
                        best_val = acc_valid
                        best_model = copy.deepcopy(model.state_dict())
                
        if e %5 == 0:
            #test on the test set
            model.eval()         
            acc_test, result, _ = test(model, classes, test_loader, temp=temp_e, str_prior_temp=str_temp_e, task_id=task_id if not args.task_agnostic_test else None)           
            if 's_long' not in args.task_sequence:
                log_wandb(result, prefix=f'result_{task_id}/')      
            print('test acc: ', acc_test, ' epoch ', e)
            log_wandb({f'task_{task_id}/test_acc':acc_test})
        e+=1  

    if best_model is not None:          
        if args.gating=='locspec': 
            # make sure the model returned has same number of modules as the best_model
            # potentially, best model can have less modules than the current model
            modules_best_model=best_model['_n_modules']
            for l, n_mod_at_layer in enumerate(modules_best_model):
                if n_mod_at_layer<model.n_modules[l]:
                    model.remove_module(at_layer=l)
        model.load_state_dict(best_model, strict=True)
    if ewc>0:  
        #calculate FIM 
        model.eval()
        def function(*d):                    
            return model(d[0].to(device)).logit
        fim = FIM(model=model.components,
                    function=function,
                    loader=train_loader,
                    representation=PMatDiag,
                    n_output=model.num_classes,
                    variant='classif_logits',
                    device=device)
        return model, fim
    return model

def bn_warmup(model, loader:DataLoader, task_id=None, bn_warmup_steps=100):
    """ warms up batchnorms by running several forward passes on the model in training mode """
    was_training=model.training
    model.train()     
    automated_module_addition_before=1#model.args.automated_module_addition
    model.args.automated_module_addition=0
    if bn_warmup_steps>0:   
        for i, (x,_) in enumerate(loader):
            model(x.to(device), record_running_stats=False, task_id=task_id if task_id is not None else -1, inner_loop=False) #temp=temp, str_prior_temp=str_temp,
            if i>=bn_warmup_steps:
                break
    model.args.automated_module_addition=automated_module_addition_before
    if not was_training:
        model.eval()
    return model

def test_with_bn(model, classes, test_loader, temp, str_temp, task_id=None, bn_warmup_steps=100):          
    """ test mode with batchnomr warmup """
    model = bn_warmup(model, test_loader, task_id, bn_warmup_steps)  
    return test(model, classes, test_loader, temp, str_temp, task_id=task_id)

def get_accs_for_tasks(model, args:ArgsGenerator, loaders:List[DataLoader], accs_past: List[float]=None, task_agnostic_test: bool=False):
    accs=[]        
    Fs = []
    masks=[]               
    task_oh_selection_accs=[]                    
    #make sure we test the same model for each task, since we may do batchnorm warm-up, this is needed here
    state_dict=copy.deepcopy(model.state_dict())
    for ti, test_loader in enumerate(loaders):    
        model.load_state_dict(state_dict, strict=True)             
        #dont warm up batch norm on the last task, as it just trained on it anyways   
        # no warm up for the last loader, if no batch norm is used
        steps_bn_warmup = 200*int(args.use_bn)*int(args.gating!='experts')*(1-(int(ti==(len(loaders)-1))*int(not task_agnostic_test)))*(1-int(args.keep_bn_in_eval_after_freeze))
        #make this explicit here
        if args.keep_bn_in_eval_after_freeze:
            steps_bn_warmup=0
        print('steps_bn_warmup', steps_bn_warmup)
        print(ti)
        acc, info, mask = test_with_bn(model, None, test_loader, model.min_temp, model.min_str_prior_temp, task_id=ti if not task_agnostic_test else None, bn_warmup_steps=steps_bn_warmup )
        acc = acc.cpu().item()
        accs.append(acc)
        masks.append(mask)
        if info is not None and len(info['task_head_selection'])>0:
            task_oh_selection_accs.append(sum(info['task_head_selection']==ti)/len(info['task_head_selection']))
        else:
            task_oh_selection_accs.append(1.)
    #     ####################
        if accs_past is not None:
            Fs.append(acc-accs_past[ti])
    model.load_state_dict(state_dict, strict=True)   
    return accs,Fs,masks,task_oh_selection_accs

def get_oh_init_idx(model, dataloader:DataLoader, args:ArgsGenerator):
    if args.init_oh=='most_likely':
        selected_head=[]
        for x,_ in dataloader:
            x = x.to(device)
            selected_head.append(model(x).info['selected_decoder'])
        return Counter(selected_head).most_common(1)[0][0]
    else:
        return None
    pass

def train(args:ArgsGenerator, model, task_idx, train_loader_current, test_loader_current, valid_dataloader, fim_prev,er_buffer):
    #args.projection_phase_length*len(train_loader_current) = prpojection phase length in number of iterations (batch updates)
    model.args.projection_phase_length = args.projection_phase_length*len(train_loader_current)
    if task_idx>0:  
        if args.warmup_bn_bf_training: 
            #warup batchnorms before training on task
            steps_bn_warmup = 200*int(args.use_bn)*int(args.gating!='experts')
            model = bn_warmup(model, train_loader_current, None, steps_bn_warmup)
        #make sure module addition is allowed from the begnning of training on task   
        model._steps_since_last_addition=torch.tensor(model.args.projection_phase_length)

    if args.running_stats_steps==0:
        model.module_options.running_stats_steps=len(train_loader_current)

    epochs=args.epochs
    best_valid_acc, best_model = None, None
    model=train_on_task(model, args, train_loader_current, valid_dataloader, test_loader_current, epochs=epochs, anneal=args.anneal, str_anneal=args.str_anneal, task_id=task_idx, reg_factor=args.reg_factor, fims=fim_prev, er_buffer=er_buffer)
    
    if args.ewc>0:  
        model, fim = model
        if args.ewc_online:
            if not isinstance(fim_prev, PMatAbstract):
                fim_prev=fim
            else:
                fim_prev=consolidate_fim(fim_previous=fim_prev ,fim_new=fim, task=task_idx)
        else:
            fim_prev.append(fim)
    # model_p=copy.deepcopy(model)
    test_acc = test(model, None, test_loader_current, model.min_temp, model.min_str_prior_temp, task_id=task_idx if not args.task_agnostic_test else None)[0].cpu().item()
    if best_valid_acc is None:
        valid_acc = test(model, None, valid_dataloader, model.min_temp, model.min_str_prior_temp, task_id=task_idx if not args.task_agnostic_test else None)[0].cpu().item()
    else:
        valid_acc=best_valid_acc
    return model,test_acc,valid_acc,fim_prev

def main(args:ArgsGenerator, task_gen:TaskGenerator):              
    t = task_gen.add_task()  
    model=init_model(args, args.gating, n_classes=t.n_classes.item(),  i_size=t.x_dim[-1]) 

    ##############################
    #Replay Buffer                 
    if args.replay_capacity!=0:
        rng = np.random.RandomState(args.seed)
        if args.replay_capacity<0:
            #automatically calculating the replay capacity to match the maximal LMC size in case of linear growth
            # memory of a float (24 bytes) x number of parameters in LMC with 1 module per layer x number of tasks = bytes of the worst case LMC
            net_size = 24 * sum([np.prod(p.size()) for p in model.parameters()]) * args.n_tasks
            # we assume that 1 pixel can be stored using 1 byte of memory
            args.replay_capacity = net_size // np.prod(t.x_dim)
        er_buffer=BalancedBuffer(args.replay_capacity,
                        input_shape=t.x_dim,   
                        extra_buffers={"t": torch.LongTensor},
                        rng=rng).to(device)
    else:
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
        print('==='*10)                                                                                         
        train_loader_current, valid_dataloader, test_loader_current = create_dataloader_ctrl(task_gen, t, args,0, batch_size=args.batch_size, labeled=True, task_n=i), create_dataloader_ctrl(task_gen, t, args,1,args.batch_size, labeled=True, shuffle_test=('ood' in args.task_sequence), task_n=i), create_dataloader_ctrl(task_gen, t, args,2,args.batch_size, labeled=True, shuffle_test=('ood' in args.task_sequence), task_n=i) 
        if args.regime=='cl':
            model,test_acc,valid_acc,fim_prev = train(args,model,i,train_loader_current,test_loader_current,valid_dataloader,fim_prev,er_buffer)
            
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
        n_modules = torch.tensor(model.n_modules).cpu().numpy()     
        log_wandb({'total_modules': np.sum(np.array(n_modules))}, prefix='model/')
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
                
        if args.multihead!='none':
            model.fix_oh(i)   
            init_idx=get_oh_init_idx(model, create_dataloader_ctrl(task_gen, t, args,0,batch_size=args.batch_size, labeled=True, task_n=i), args)
            print('init_idx', init_idx)        
            model.add_output_head(t.n_classes.item(), init_idx=init_idx)
        else:
            #single head mode: create new, larger head
            model.add_output_head(model.decoder.out_features+t.n_classes.item(), state_dict=model.decoder.state_dict())

        if args.gating not in ['experts']:
            for l in range(len(n_modules)):
                log_wandb({f'total_modules_l{l}': n_modules[l]}, prefix='model/')
            if args.use_structural:      
                if args.use_backup_system:
                    model.freeze_permanently_structure()
                else:
                    for l,layer in enumerate(model.components):   
                        for m in layer:                          
                            m.freeze_functional(inner_loop_free=False)
                            m.freeze_structural()     
                            m.module_learned=torch.tensor(1.)
                            # model.add_modules(at_layer=l)
            model.optimizer, model.optimizer_structure = model.get_optimizers()
    
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
    print('Average accuracy (test) at the end of the sequence:',np.mean(accs_test))
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
    print('Average accuracy (valid) at the end of the sequence:',np.mean(accs_valid))
    #Average accuracy (valid) at the end of the sequence 
    log_wandb({"mean_valid_acc":np.mean(accs_valid)})#, prefix='valid/')
    #Average forgetting (valid)
    log_wandb({"mean_valid_F":np.mean(Fs_valid)})#, prefix='test/')
    ####################    
         
    if args.task_sequence_test is not None and 'ood' in args.task_sequence:
        #test on all combinations of features and classes
        state_dict_learned=model.state_dict()
        task_gen_test = ctrl.get_stream(args.task_sequence_test, seed=args.stream_seed)
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
            loader_valid, loader_test = create_dataloader_ctrl(task_gen, t, args,1, batch_size=args.batch_size, labeled=True, task_n=i), create_dataloader_ctrl(task_gen, t, args,2, batch_size=args.batch_size, labeled=True, task_n=i)       
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
            fig, axs = pyplot.subplots(len(unique_transformations),len(np.unique(classes)),figsize=(10,2*len(unique_transformations)))
            fig.tight_layout(pad=2.5)
            for row, ax_row in enumerate(axs):
                for column, ax in enumerate(ax_row):
                    im = ax.imshow(masks_test[column*len(axs)+row].cpu().T, cmap='Blues')
                    ax.set_title(unique_transformations[row].replace('\n', ''))
                    ax.set_yticks([0,1,2,3,4])  
                    ax.set_xticks([0,1,2,3])
                    if row == column:
                        for spine in ax.spines.values(): 
                            spine.set_edgecolor('red')#, linewidth=2)
            # set labels
            for i,cl in enumerate(np.unique(classes)):
                plt.setp(axs[-1, i], xlabel=f'layer\nClasses {cl}')
            plt.setp(axs[:, 0], ylabel='module')
            pyplot.savefig('module_selection.pdf', format='pdf',dpi=300, bbox_inches='tight')

            log_wandb({f"ood/module_usage": wandb.Image(fig)})
        
        col = np.unique(classes)
        df_cm = pd.DataFrame(array[:len(col)], index = unique_transformations[:len(col)],columns = np.unique(classes))

        log_wandb({f"mean_test_ood": np.mean(array[:len(col)])}) 
        log_wandb({f"mean_valid_ood": np.mean(array_valid[:len(col)])})
        plot_confusion(df_cm, wandb_tag='confusion_matrix')
        return df_cm
    return None
                        
def plot_confusion(df_cm, wandb_tag=None, save_dir=None, labels=None):    
    #################### 
    #create a confusion matrix/
    fig = pyplot.figure(figsize = (15.5,15))
    sn.set(font_scale=2.0)   
    if labels is not None:
        hm=sn.heatmap(df_cm, annot=labels, vmin=0, vmax=1, fmt="", annot_kws={"size":28})
    else:
        hm=sn.heatmap(df_cm, annot=True, vmin=0, vmax=1, fmt=".2%", annot_kws={"size":28})
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=30, va="center")
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=30)
    
    #confusion matrix
    if wandb_tag is not None:
        log_wandb({f"{wandb_tag}": wandb.Image(fig)})
    if save_dir is not None:
        fig.savefig(save_dir, format='pdf', dpi=300, bbox_inches = 'tight',pad_inches = 0)
    
    matplotlib.rc_file_defaults()
        

if __name__== "__main__":                                     
    parser = ArgumentParser()     
    parser.add_arguments(ArgsGenerator, dest="Global")
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
    #for ood experiments, plot the confusion matrix with the standard deviations
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
