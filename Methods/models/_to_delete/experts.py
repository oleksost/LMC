import copy
from dataclasses import dataclass
import time
import math
import torch
import numpy as np
import torch.nn as nn 
from collections import deque
from runstats import Statistics
import torch.nn.functional as F
from collections import OrderedDict

from Utils.utils import RunningStats, DequeStats, cosine_rampdown, ordered_dict_mean
from .base_modular import ModularBaseNet, conv_block_base

 
device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ExpertMixture(ModularBaseNet):

    def __init__(self, options: ModularBaseNet.Options = ModularBaseNet.Options(),        
                       module_options: conv_block_base.Options = conv_block_base.Options(), 
                       i_size: int = 28, channels:int = 1, hidden_size = 64, num_classes: int = 5):
                       
        super(ExpertMixture, self).__init__(options, i_size, channels, hidden_size, num_classes)        
        #n_modules - might change over the runtime      
        ###########
        self.module_options = module_options
        self.n_experts = self.n_modules
        ###########
        self.min_str_prior_temp=None
        self.init_modules()

    def init_modules(self):
        self.components = nn.ModuleList()
        for _ in range(self.n_experts+1):      
            expert = Expert(self.depth, self.args.net_arch, self.i_size, self.channels, self.hidden_size, self.num_classes, self.args.module_type, module_options=self.module_options)      
            self.components.append(expert)
            
        self.components[-1].load_state_dict(copy.deepcopy(self.get_average_parameters(self.components[:-1])))
        
        self.softmax = nn.Softmax(dim=0)  
        if self.args.regime=='normal':
            self.optimizer, self.optimizer_structure = self.get_optimizers()
        
    def update_global_prior(self, weights=None):
        print('Updating global prior')     
        self.components[-1].load_state_dict(copy.deepcopy(self.get_average_parameters(self.components[:-1], weights)))
    
    @staticmethod 
    def get_average_parameters(experts:nn.Module, weights=None):        
        return Ordered_dict_mean(experts.state_dict(), weights)
              
    def freeze_permanently_functional(self, free_layer=None):
        for l, expert in enumerate(self.components):            
                expert.freeze_functional()   
                #self.frozen_components.add(f'components.{l}.{c}')
        if self.args.regime=='normal':
            self.optimizer, self.optimizer_structure = self.get_optimizers()
    
    def freeze_permanently_structure(self, free_layer=None):
        raise Exception('no structural learning component is defined')
    
    def add_modules(self, num_classes=None, **kwargs):
        self.n_experts+=1; self.n_modules=self.n_modules + 1
        #add one new expert
        num_classes = self.num_classes if num_classes is None else num_classes    
        new_expert = Expert(self.depth, self.args.net_arch, self.i_size, self.channels, self.hidden_size, num_classes, self.args.module_type, module_options=self.module_options)        
        ##################################################
        ###Initialize from global prior###  
        if self.args.regime!='normal':
            new_expert.load_state_dict(copy.deepcopy(self.components[-1].state_dict())) #this is the new global prior
        ##################################################
        self.components.append(new_expert.to(device))        
        if self.args.regime=='normal':
            self.optimizer, self.optimizer_structure = self.get_optimizers()
    
    def add_output_head(self,num_classes=None, *args, **kwargs):
        return self.add_modules(num_classes)

    def get_optimizers(self):                                                    
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr, weight_decay=self.args.wdecay) #, betas=[self.args.optimizer_adam_b_1, 0.999])
        optimizer_structure = None
        return optimizer, optimizer_structure
     
    def accept_new_expert(self, freeze_previous=False):
        if freeze_previous:
            for e in self.components[:-1]:
                e.freeze_functional()
        self.add_modules()
    
    def fix_oh(self, i):     
        self.components[i].freeze_functional(requieres_grad=self.args.regime!='normal')
           
    def forward(self, X, task_id=None, *args, **kwargs):
        # if self.module_type=='linear':
        #         X=X.view(X.size(0), -1) 
        X_out = []
        if task_id is not None:
            _, logit = self.components[task_id](X)
            logit = logit.unsqueeze(0)
        else:
            for expert in self.components:                                                                                               
                _, X_c = expert(X)   
                X_out.append(X_c)
            logit = X_out
        return self.forward_template(logit=logit)

def MixtureofExperts(args_model, args_module, out_features, hidden_size=64, i_size=28):
    return ExpertMixture(args_model, args_module, i_size =i_size, channels=1, hidden_size=hidden_size, num_classes=out_features)

def MixtureofExpertsImnet(args_model, args_module, out_features, hidden_size=64, i_size=64):
    return ExpertMixture(args_model, args_module, i_size =i_size, channels=3, hidden_size=hidden_size, num_classes=out_features)