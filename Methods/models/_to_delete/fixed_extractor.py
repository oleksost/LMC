import copy
from logging import error
import torch
import random
from .base_modular import bn_eval
from torch.nn.parameter import Parameter    
import torchvision   
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F  
from torchvision import transforms as T
from typing import List, Tuple, Union, Dict,Optional, Iterable
from dataclasses import dataclass
from Utils.utils import match_dims, ordered_dict_mean
from .base_modular import ModularBaseNet
from Methods.metalearners.utils import ChannelPool, convtransp_output_shape, create_mask
from .LMC import LMC_net
from .LMC_components import ComponentList, LMC_conv_block, GatedOh, LMC_conv_block_BYOL
from simple_parsing import choice
device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# augmentation utils


class OvAInn(ModularBaseNet):
    @dataclass
    class Options(ModularBaseNet.Options):  
        fixed_encoder: bool = 1 #-
        bn_eval: bool = 0 #-

    def __init__(self, options:Options = Options(), module_options:LMC_conv_block.Options=LMC_conv_block.Options(), i_size: int = 28, channels:int = 1, hidden_size=64, num_classes:int=5):
        super(OvAInn, self).__init__(options, i_size, channels, hidden_size, num_classes)
        
        self.args: OvAInn.Options = copy.copy(options)
        self.bn = not module_options.batchnorm_inside_of_module
        self.deviation_threshold=self.args.deviation_threshold
        self.module_options=module_options
        self.meta_lr_structure=self.args.meta_lr_structure         
        self.catch_outliers_for_old_modules=self.args.catch_outliers_old
        ##############################
        #n_modules - might change over the runtime
        self.register_buffer('_n_modules', torch.tensor([float(self.n_modules)]*self.depth))    
        self.register_buffer('_steps_since_last_addition', torch.tensor(0.))   
        self.register_buffer('min_str_prior_temp', torch.tensor(float(self.args.str_prior_temp)))
        ##############################
        self.init_modules()    
        self.modules_to_unfreeze_func=[]

        self.register_buffer('outlier_batch_counter_buffer', torch.tensor([0. for _ in range(self.depth)]))
    
    def train(self, *args, **kwargs):      
            r = super().train(*args, **kwargs)
            if self.args.fixed_encoder and self.args.bn_eval:
                bn_eval(self.encoder)
            if self.args.fixed_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False
            return r  
    @property    
    def total_number_of_modules(self):
        return torch.sum(self._n_modules).item()
    
    @property
    def n_modules(self):
        return self.n_modules_per_layer
    
    @property
    def n_modules_per_layer(self):
        return self._n_modules#.cpu().numpy()
    
    @property
    def optimizers_update_needed(self): 
        return any(l.optimizer_update_needed for l in self.components)
        
    def n_modules_at_layer(self, at_layer):
        return int(self._n_modules[at_layer].item())

    def maybe_add_modules(self, outlier_signal, fixed_modules, layer, bottom_up=True, module_i=None, init_stats=False):
        add_modules_layers = False  
        unfreeze_modules = torch.zeros(len(self.components[layer]))       
        freeze_modules_functional = torch.zeros(len(self.components[layer]))
        if self.training:      
            _outlier_ = outlier_signal>=self.deviation_threshold 
            # print(outlier_signal) 
            criterion_1 = (_outlier_.sum(0)==self.n_modules[layer]) #if all modules report outliers
            criterion_2 = (fixed_modules.sum(0)==self.n_modules[layer])# if all modules are fixed
            add_modules_layers = (criterion_1 & criterion_2).item()
            
            if add_modules_layers and self.args.module_addition_batch_number>0:
                #Can only add modules if outlier was detected in several batche sin a row (should be more robust)
                self.outlier_batch_counter_buffer[layer]=self.outlier_batch_counter_buffer[layer]+1
                if not self.outlier_batch_counter_buffer[layer]>=self.args.module_addition_batch_number:
                    add_modules_layers=False                    
            elif not add_modules_layers and self.args.module_addition_batch_number>0:
                self.outlier_batch_counter_buffer[layer]*=0

            #freeze and unfreeze modules
            unfreeze_modules = (fixed_modules==1) & (_outlier_==0) 
            #we dont want to unfreeze modules in layers where new modules have been already added
            unfreeze_modules = unfreeze_modules & (criterion_2)

            #newly added modules will not be frozen here
            freeze_modules_functional = (fixed_modules==1) & (_outlier_==1)

        new_m_idx=[]
        _params_changed=0      
        #prefer unfreezing over addition
        if add_modules_layers and (not self.args.active_unfreezing or (self.args.active_unfreezing and not torch.sum(unfreeze_modules)>0)):
            _params_changed=True  
            print('adding at layers ', layer)
            self.add_modules(at_layer=layer, module_i=module_i, init_stats=init_stats)
            new_m_idx.append(-1)
        else:
            if self.args.active_unfreezing: 
                if torch.sum(unfreeze_modules)>0:     
                    if self.args.treat_unfreezing_as_addition:  
                                    for c in self.components[layer]:
                                       c.on_module_addition_at_my_layer(inner_loop_free=(self.args.regime=='meta'))    
                    for i_m, m in enumerate(unfreeze_modules):
                            if m:     
                                unfreeze_completed = self.components[layer][i_m].unfreeze_learned_module(self.args.unfreeze_structural)
                                _params_changed = max( _params_changed, unfreeze_completed)
                                self.components[layer][i_m].detach_structural=True
                                # if self.args.treat_unfreezing_as_addition:  
                                self.components[layer][i_m].module_learned=torch.tensor(0.)
                                new_m_idx.append(i_m)
                                break
            
                # if torch.sum(freeze_modules_functional)>0:
                #     for i_m, m in enumerate(freeze_modules_functional):
                #             if m:    
                #                 _params_changed = max( _params_changed, self.components[layer][i_m].freeze_learned_module())
        return add_modules_layers, _params_changed, new_m_idx

    def init_modules(self):
        if self.args.encoder:           
            self.encoder = torchvision.models.resnet18(pretrained=True)
            # self.encoder.layer4=nn.Identity()
            
            self.encoder.fc = nn.Identity()
            self.args.module_type='linear'
            self.channels = 1
            self.i_size = 512
            if self.args.fixed_encoder:
                # def set_bns(model:nn.Module):
                # for layer in model.children():
                #     if isinstance(layer, nn.BatchNorm2d):
                #         layer.momentum=1.
                #         layer.track_running_stats=False
                #     elif isinstance(layer.children(), Iterable):
                #         set_bns(layer)
                # et_bns(self.encoder)
                if self.args.bn_eval:
                    bn_eval(self.encoder)
                for p in self.encoder.parameters():
                    p.requires_grad = False
            # self.i_size = 6
            # self.channels = 256
        else:
            self.encoder = None

        channels_in = self.channels
        if self.args.module_type=='linear':
            channels_in = self.i_size * self.channels  

        out_h=self.i_size  
        self.str_priors=nn.ModuleList()
        self.bnorms=nn.ModuleList()
        hidden_size=self.hidden_size
        self.channels_in = channels_in        
        if self.args.module_type=='resnet_block':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            out_h=1
        else:
            self.avgpool=nn.Identity()

        if self.args.module_type=='linear':
            out_h=1                         

        if self.args.multihead != 'modulewise':  
            if self.args.multihead=='gated_linear':
                self.representation_dim = self.channels_in*out_h*out_h      
                self.decoder = nn.ModuleList([GatedOh(out_h, self.representation_dim, self.representation_dim, self.num_classes, initial_inv_block_lr=self.meta_lr_structure, module_type='linear', options=self.module_options, name='oh_0')])
            elif self.args.multihead=='gated_conv':
                self.representation_dim = self.channels_in
                self.decoder = nn.ModuleList([GatedOh(out_h, self.channels_in, hidden_size, self.num_classes, initial_inv_block_lr=self.meta_lr_structure, module_type='conv', options=self.module_options, name='oh_0')])
            else: #self.args.multihead=='usual':
                self.representation_dim = self.channels_in*out_h*out_h      
                self.decoder = nn.Linear(self.representation_dim, self.num_classes) if self.args.multihead=='none' else nn.ModuleList([nn.Linear(self.representation_dim, self.num_classes)])
        
        else:
            self.decoder = nn.Identity()

        self.ssl_pred = torch.nn.Identity()        
        self.softmax = nn.Softmax(dim=0)  
        if self.args.regime=='normal':
            self.optimizer, self.optimizer_structure = self.get_optimizers()
        self._reset_references_to_learnable_params()
    
    def fix_oh(self, i):              
        for p in self.decoder[i].parameters():
            p.requires_grad = False 
        try:
            if self.args.regime=='normal':
                self.optimizer, self.optimizer_structure = self.get_optimizers()
        except ValueError:
            #will throw Value error if all modules are frozen
            pass
                
    def add_output_head(self, num_classes=None, init_idx=None, state_dict=None):            
        num_classes = self.num_classes if num_classes is None else num_classes
        # if self.args.multihead=='usual':   
        # if self.args.gate_oh:
        if self.args.multihead=='gated_linear':   
            out_h = self.decoder[-1].in_h                      
            l=GatedOh(out_h, self.representation_dim, self.representation_dim, num_classes, initial_inv_block_lr=self.meta_lr_structure, module_type='linear', options=self.module_options, name=f'oh_{len(self.decoder)}')
            if init_idx is not None:
                l.load_state_dict(copy.deepcopy(self.decoder[init_idx].state_dict()))
            self.decoder.append(l.to(device))
        elif self.args.multihead=='gated_conv':
            out_h = self.decoder[-1].in_h
            l=GatedOh(out_h, self.channels_in, self.hidden_size, num_classes, initial_inv_block_lr=self.meta_lr_structure, module_type='conv', options=self.module_options, name=f'oh_{len(self.decoder)}')
            if init_idx is not None:
                l.load_state_dict(copy.deepcopy(self.decoder[init_idx].state_dict()))
            self.decoder.append(l.to(device))

        elif self.args.multihead=='usual':     
            l=nn.Linear(self.representation_dim, num_classes)
            if init_idx is not None:
                l.load_state_dict(copy.deepcopy(self.decoder[init_idx].state_dict()))
            self.decoder.append(l.to(device))
        
        else:
            #single head 
            self.decoder=nn.Linear(self.representation_dim, num_classes).to(device)
            if state_dict is not None:
                for k in state_dict:
                    if hasattr(self.decoder, k):
                        if len(getattr(self.decoder, k).shape)==2:
                            getattr(self.decoder, k).data[:state_dict[k].size(0),:state_dict[k].size(1)] = state_dict[k].data
                        elif len(getattr(self.decoder, k).shape)==1:
                            getattr(self.decoder, k).data[:state_dict[k].size(0)]=state_dict[k].data
                        else:
                            raise NotImplementedError
               #self.decoder.load_state_dict(state_dict, srict=False)

        
        # try:          
        if self.args.regime=='normal':  
            self.optimizer, self.optimizer_structure = self.get_optimizers()
        # except ValueError:
        #     #will throw Value error if all modules are frozen
        #     pass

    # def reinit_decoder(self):
    #     import math  
    #     stdv = 1. / math.sqrt(self.decoder.weight.size(1))
    #     self.decoder.weight.data.uniform_(-stdv, stdv)
    #     if self.decoder.bias is not None:
    #         self.decoder.bias.data.uniform_(-stdv, stdv)

    @property
    def projection_phase(self):
        return False

    def get_mask(self, temp=None, train_inputs=None, task_id=0, params=None):
        temp = temp if temp is not None else self.args.temp
        if self.mask_activation=='softmax':
            return self.softmax(self.structure[task_id]/temp)
        elif self.mask_activation=='sigmoid':
            return torch.sigmoid(self.structure[task_id]*temp)
        else:
            raise NotImplementedError
           
    def freeze_permanently_functional(self, free_layer=None, inner_loop_free=True):
        for l, layer in enumerate(self.components):
            if l!=free_layer:
                for c, component in enumerate(layer):
                    component.freeze_functional(inner_loop_free)
        if self.args.regime=='normal':
            self.optimizer, self.optimizer_structure = self.get_optimizers()
        self._reset_references_to_learnable_params()
    
    def freeze_permanently_structure(self, free_layer=None, clear_backups=True):
        for l, layer in enumerate(self.components):
            if l!=free_layer:
                for c, component in enumerate(layer):
                    if component.active:
                        component._reset_backups()  

        if self.optimizers_update_needed:
            if self.args.regime=='normal':
                self.optimizer, self.optimizer_structure = self.get_optimizers()
            self._reset_references_to_learnable_params()
    
    def remove_module(self, at_layer):      
        self.components[at_layer]=self.components[at_layer][:-1]
        self._n_modules[at_layer]+=-1
        
    def add_modules(self, block_constructor=None, at_layer=None, strict=False, module_i=None, verbose=True, init_stats=False):
        #if at_layer == None add to all layers
        if block_constructor is None:            
            block_constructor = self.block_constructor      
        for l, layer in enumerate(self.components):  
            ################################################
            if l != at_layer and at_layer!=None:
                if self.args.freeze_module_after_step_limit and self.args.reset_step_count_on_module_addition:
                    for c in layer:
                        if c.active:
                            #reset step count for other modules
                            c.reset_step_count()

            elif l == at_layer or at_layer==None:
                if verbose:
                    print(f'adding modules at layer {l}')
                self._n_modules[l]+=1
                ################################################
                #new module's params
                in_h = layer[0].in_h
                channels_in = layer[0].in_channels
                channels = layer[0].out_channels
                i_size = layer[0].i_size
                deeper = layer[0].deeper
                module_type = layer[0].module_type         
                self.module_options.kernel_size=layer[0].args.kernel_size
                self.module_options.dropout=layer[0].args.dropout
                # out_h = layer[0].out_h       
                ################################################
                init_params = None         
                if self.args.module_init == 'identical':         
                    init_params = self.per_layer_module_initialization[l]
                elif self.args.module_init == 'mean':
                    if verbose:
                        print('adding module with mean innitialization')
                    l_state_dict=layer.state_dict()
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, l_state_dict.keys()))
                        for k in keys_to_pop:
                            l_state_dict.pop(k, None)
                    init_params = copy.deepcopy(Ordered_dict_mean(l_state_dict))
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, init_params.keys()))
                        for k in keys_to_pop:
                            init_params.pop(k, None)
                elif self.args.module_init == 'existing':
                    init_params = layer[0].state_dict()
                elif self.args.module_init == 'previously_active':
                    for c in layer:
                        if c.active:
                            init_params = c.state_dict()                                       
                    assert init_params is not None, f'At least one module in the layer {l} should have been active'
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, init_params.keys()))
                        for k in keys_to_pop:
                            init_params.pop(k, None)
                elif self.args.module_init == 'most_likely' and module_i is not None:
                    # assert (str_prior is not None)
                    # module_i = str_prior.argmax().item()
                    init_params=layer[module_i].state_dict()
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, init_params.keys()))
                        for k in keys_to_pop:
                            init_params.pop(k, None)

                for c in layer:
                    if not c.args.anneal_structural_lr:
                        c.on_module_addition_at_my_layer(inner_loop_free=(self.args.regime=='meta'))    
                
                # if self.args.module_type=='resnet_block':
                #     if l==0:
                #         module_type='conv'
                #     else:
                #         module_type='resnet_block'
                # else:
                #     module_type=self.args.module_type    
                new_module = block_constructor(in_h, channels_in, channels, i_size, initial_inv_block_lr=self.meta_lr_structure, 
                                                            name=f'components.{l}.{len(layer)}', 
                                                            module_type=module_type, deviation_threshold=self.deviation_threshold, freeze_module_after_step_limit=self.args.freeze_module_after_step_limit, deeper=deeper, options=self.module_options,
                                                            num_classes=self.num_classes if (self.args.multihead=='modulewise' and l==self.depth-1) else 0 )
                if init_params is not None:  
                    if verbose:
                        print('loading state dict new module')     
                    new_module.load_state_dict(init_params, strict)

                new_module.to(device)
                new_module.reset()
                layer.append(new_module)
        self._reset_references_to_learnable_params()        
        if self.args.regime=='normal':  
            self.optimizer, self.optimizer_structure = self.get_optimizers()
             
    def get_optimizers(self):  
        #set seperate optimizer for structure and model
        structure_names = ['inv_block', 'structure']   
        if self.args.fixed_encoder:
            structure_names.append('encoder')
        model_params = [param for n, param in self.named_parameters() if not any(map(n.__contains__, structure_names)) and param.requires_grad]
        # for n, p in self.named_parameters():
        #     print(n, p.is_leaf)
        try:
            optimizer = torch.optim.Adam(model_params, lr=self.args.lr, weight_decay=self.args.wdecay)
        except:
            optimizer=None
        
        structure_param_groups = []  
        #create module specific parameter groups
        if self.args.module_type!='filterwise':
            for layer in self.components:
                for component in layer:   
                    if hasattr(component, 'inv_block'):            
                        params = list(filter(lambda p: p.requires_grad, component.inv_block.parameters()))
                        if len(params)>0:
                            structure_param_groups.append({'params': params, 'lr': component.block_lr_buffer, 'name': component.name})
        else:
            for layer in self.components:
                for component in layer:                
                    pass
                    #structure_param_groups.append({'params': filter(lambda p: p.requires_grad, component.inv_block.parameters()), 'lr': component.block_lr_buffer, 'name': component.name})
        if 'gated' in self.args.multihead:
            for decoder in self.decoder:
                if hasattr(decoder, 'inv_block'):            
                        params = list(filter(lambda p: p.requires_grad, decoder.inv_block.parameters()))
                        if len(params)>0:
                            structure_param_groups.append({'params': params, 'lr': decoder.block_lr_buffer, 'name': decoder.name})

        if len(structure_param_groups)>0:
            optimizer_structure = torch.optim.Adam(structure_param_groups,weight_decay=self.args.wdecay) # lr=self.args.meta_lr_structure)
        else:
            optimizer_structure=None   
        return optimizer, optimizer_structure

    def forward(self, X, task_id=None, params=None, ssl=False, temp=None, inner_loop=False, decoder = None, record_stats=False, env_assignments=None, info='', detach_head=False, str_prior_temp=None, **kwargs):
        #start = time.time()     
        X = self.encoder(X) 
        X=X.view(X.size(0), -1) 
        decoder_idx=None
        inv_total=0   
        if self.args.multihead=='none':
            logit=self.decoder(X)         
        elif 'gated' in self.args.multihead:     
            if task_id is not None and task_id<=(len(self.decoder)-1):
                logit, str_act = self.decoder[task_id](X)
                if self.training:       
                    inv_total+=str_act.mean()
            else:
                #dont use task label, instead, select most lilely head
                assert not self.training
                logit, best_str_act, decoder_idx=None, None, None
                for d, decoder in enumerate(self.decoder):
                    logit_d, str_act = decoder(X)
                    if best_str_act is None:
                        best_str_act=str_act.mean()
                        decoder_idx=d
                        logit=logit_d
                    elif str_act.mean()<best_str_act: 
                        best_str_act=str_act.mean()
                        decoder_idx=d
                        logit=logit_d
        else:
            raise NotImplementedError
        if self.training and self.args.regime=='normal':
            self.handle_outer_loop_finished(True)

        return self.forward_template(logit=logit, regularizer=inv_total,   
        info={'selected_decoder': decoder_idx})#, 'modules_lr': modules_lr}) 'add_modules_layers': add_modules_at_leayer, 

    def handle_outer_loop_finished(self, finished_outer_loop: bool):
        for l in self.components:
            for c in l:
                c.on_finished_outer_loop(finished=finished_outer_loop)
    def on_before_zero_grad(self):
        pass
     
    def load_state_dict(self, state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], strict: bool=True):
        #TODO: test this
        return super().load_state_dict(state_dict,strict)


def LocalSpecializationConvOmniglot(args_model, args_module, out_features, hidden_size=64, i_size=28):       
    return LMC_net(args_model, args_module, i_size =i_size, channels=1, hidden_size=hidden_size, num_classes=out_features)
def LocalSpecializationConvImnet(args_model, args_module, out_features, hidden_size=32, i_size=64):
    return LMC_net(args_model, args_module, i_size =i_size, channels=3, hidden_size=hidden_size, num_classes=out_features)