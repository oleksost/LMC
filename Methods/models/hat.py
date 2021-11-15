import copy
from dataclasses import dataclass
from ssl import Options
import time
import math
from simple_parsing.helpers.fields import choice
import torch
import numpy as np
import torch.nn as nn 
from collections import deque
from runstats import Statistics
import torch.nn.functional as F
from collections import OrderedDict

from Utils.utils import RunningStats, DequeStats, cosine_rampdown, ordered_dict_mean
from .base_modular import ModularBaseNet, Expert, conv_block_base, Conv_Module


from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
 
device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Expert_HAT(Expert):
    def __init__(self, depth: int, net_arch: int, i_size: tuple, channels, hidden_size, num_classes, module_type, module_options: conv_block_base.Options, Module):
        super().__init__(depth, net_arch, i_size, channels, hidden_size, num_classes, module_type, module_options=module_options, Module=Module)
    def forward(self, x, t, s):
        # n = x.size(0)   
        masks=[]        
        reg=0  
        t = torch.autograd.Variable(torch.LongTensor([t]).to(device), volatile=False)
        for module in self.components: 
            x, mask = module(x=x, t=t, s=s)
            masks.append(mask)
            r=module.criterion(mask)
            assert not torch.isnan(r)
            reg+=r
        return x, masks, reg
    def on_task_learned(self, **kwargs):
        prev_component=None
        for i,c in enumerate(self.components): 
            c.on_task_learned(prev_component=prev_component,**kwargs)
            prev_component = c
    def restrict_grads(self, **kwargs):
        for c in self.components:
            c.restrict_grads(**kwargs)
    # def criterion(self, masks):
    #     reg = 0        
    #     for i,c in enumerate(self.components):
    #         reg+=c.criterion(masks[i])
    #     return reg
    def constraint_embeddings(self, **kwargs):
        for c in self.components:
            c.constraint_embeddings(**kwargs)


class Conv_ModuleHAT(Conv_Module):                   
    def __init__(self, in_channels, out_channels, kernel_size, track_running_stats, pooling_kernel, n_tasks, pooling_stride=None, pooling_padding=0, affine_bn =True, momentum=1, decode=False, use_bn=True, hidden_size=64, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, track_running_stats, pooling_kernel, pooling_stride=pooling_stride, pooling_padding=pooling_padding, affine_bn=affine_bn, momentum=momentum, decode=decode, use_bn=use_bn, **kwargs)
        self.gate=nn.Sigmoid()    
        self.ec1=torch.nn.Embedding(n_tasks, hidden_size)

    def forward(self, x, t, s=1):
        gc1=self.mask(t,s=s)
        h = self.module(x)
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        return h, gc1

    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))        
        return gc1
    
    def get_view_for(self, n, gc2, gc1=None):
        if n=='module.conv.weight':
            # return gc1.data.view(-1,1,1,1).expand_as(self.module.conv.weight)
            if gc1 is None:
                return gc2.data.view(-1,1,1,1).expand_as(self.module.conv.weight)
            post=gc2.data.view(-1,1,1,1).expand_as(self.module.conv.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.module.conv.weight)
            return torch.min(post,pre)
        elif n=='module.conv.bias':
            return gc2.data.view(-1)
        return None
    
class conv_block_hat(conv_block_base):
    def __init__(self, in_channels, out_channels, i_size, name, module_type, n_layers, stride, bias=True, deeper=False, options: Options=conv_block_base.Options(), n_classes=None, **kwargs):
        
        assert options.n_tasks is not None
        super().__init__(in_channels, out_channels, i_size, name=name, module_type=module_type, n_layers=n_layers, stride=stride, bias=bias, deeper=deeper, options=options, n_classes=n_classes, **kwargs)   
        if module_type == 'conv':             
            if not deeper:
                self.module = Conv_ModuleHAT(self.in_channels, self.out_channels, self.args.kernel_size, self.args.track_running_stats_bn, self.args.maxpool_kernel, pooling_stride=self.args.maxpool_stride, 
                                        pooling_padding=self.args.maxpool_padding, padding=self.args.padding, affine_bn=self.args.affine_bn, momentum=self.args.momentum_bn, use_bn=self.args.use_bn, stride=stride, bias=bias, n_tasks=self.args.n_tasks, hidden_size=options.hidden_size, **self.module_kwargs).to(device)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
        self.mask_pre=None
        self.mask_back=None

    def forward(self, x, t, s):
        return self.module(x,t=t,s=s) 
    def on_task_learned(self,prev_component, t, smax): 
        task = torch.autograd.Variable(torch.LongTensor([t]).to(device),volatile=False)
        mask = self.module.mask(task,s=smax)
        # mask=[None,mask]
        mask_pre_prev=None
        if prev_component is not None:  
            mask_pre_prev = copy.deepcopy(prev_component.mask_pre)
            # mask[0]=mask_pre_prev
            
        # for i in range(len(mask)):
            # if mask[i] is not None: 
        mask=torch.autograd.Variable(mask.data.clone(),requires_grad=False)
        # mask = torch.autograd.Variable(mask.data.clone(),requires_grad=False)
        # for i in range(len(mask)):
        #     mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)


        if task==0:
            self.mask_pre=mask
        else:
            # for i in range(len(self.mask_pre)):
            # if mask[i] is not None:
            self.mask_pre=torch.max(self.mask_pre,mask)

        # Weights mask
        self.mask_back={} 
        for n, _ in self.module.named_parameters():
            # n='module.'+n
            vals=self.module.get_view_for(n,self.mask_pre,mask_pre_prev)
            n='module.'+n
            if vals is not None: 
                self.mask_back[n]=1-vals
        # self.mask_pre = self.mask_pre[1]

    def restrict_grads(self, t, s, smax, thres_cosh, clipgrad):
        # Restrict layer gradients in backprop
        if t>0:
            for n,p in self.named_parameters():
                if n in self.mask_back:  
                    p.grad.data *= self.mask_back[n]

        # Compensate embedding gradients
        for n,p in self.module.named_parameters():
            if n.startswith('e'):
                num = torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                den = torch.cosh(p.data)+1
                p.grad.data *= smax/s*num/den
        
        torch.nn.utils.clip_grad_norm_(self.module.parameters(),clipgrad)
    
    def criterion(self, masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        if count>0:
            reg/=count
        assert not torch.isnan(reg)
        return reg
    
    def constraint_embeddings(self, thres_emb):
        for n,p in self.module.named_parameters():
                if n.startswith('e'):
                    p.data=torch.clamp(p.data,-thres_emb,thres_emb)

class HAT(ModularBaseNet):
    @dataclass                         
    class Options(ModularBaseNet.Options):
        smax: int = 400 # smax
        thres_cosh: float = 6 #thres_cosh
        # lamb: float = 0.75 # lamb
        clipgrad: float = 10000. #clipgrad 
        thres_emb: float = 6. #thres_emb
        optimizer: str = choice('adam', 'sgd', default='sgd')
        
    def __init__(self, options: Options = Options(),     
                       module_options: conv_block_hat.Options = conv_block_hat.Options(), 
                       i_size: int = 28, channels:int = 1, hidden_size = 64, num_classes: int = 5):
                       
        super(HAT, self).__init__(options, i_size, channels, hidden_size, num_classes)
        ###########
        self.module_options = module_options
        self.n_experts = self.n_modules
        self.module_options.hidden_size=hidden_size
        ###########
        self.min_temp, self.min_str_prior_temp = None, None
        self.init_modules()

    def init_modules(self):
        self.components = nn.ModuleList()     
        model = Expert_HAT(self.depth, self.args.net_arch, self.i_size, self.channels, self.hidden_size, self.num_classes, self.args.module_type, module_options=self.module_options, Module=conv_block_hat)      
        self.representation_dim=model.decoder.in_features
        model.decoder = nn.Identity()
        self.components.append(model)
        self.softmax = nn.Softmax(dim=0)
        if self.args.multihead=='usual':
            self.decoder = nn.ModuleList([nn.Linear(self.representation_dim, self.num_classes)])
        else:
            raise NotImplementedError

        if self.args.regime=='normal': 
            self.optimizer, self.optimizer_structure = self.get_optimizers()    

    @staticmethod 
    def get_average_parameters(experts:nn.Module, weights=None):        
        return ordered_dict_mean(experts.state_dict(), weights)
              
    def freeze_permanently_functional(self, free_layer=None):
        for l, expert in enumerate(self.components):            
                expert.freeze_functional()   
                #self.frozen_components.add(f'components.{l}.{c}')
        if self.args.regime=='normal':
            self.optimizer, self.optimizer_structure = self.get_optimizers()
    
    def freeze_permanently_structure(self, free_layer=None):
        raise Exception('no structural learning component is defined')

    def add_output_head(self,num_classes=None, init_idx=None, **kwargs):             
        num_classes = self.num_classes if num_classes is None else num_classes            
        l=nn.Linear(self.representation_dim, num_classes)
        if init_idx is not None:
            l.load_state_dict(copy.deepcopy(self.decoder[init_idx].state_dict()))
        self.decoder.append(l.to(device)) 
        # try:          
        if self.args.regime=='normal':        
            self.optimizer, self.optimizer_structure = self.get_optimizers()

    def get_optimizers(self):   
        if self.args.optimizer=='sgd':                              
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr, weight_decay=self.args.wdecay)#original implemetntion uses SGD, runing with Adam might be problematic due to momentum.
        elif self.args.optimizer=='adam':
            optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr, weight_decay=self.args.wdecay)
        optimizer_structure = None
        return optimizer, optimizer_structure
     
           
    def forward(self, X, task_id=None, s=None, *args, **kwargs):
        if s is None:
            s=self.args.smax
        # print(s)
        assert task_id is not None
        n=X.size(0)        
        h, masks, reg = self.components[-1](X, t=task_id, s=s)
        h = h.view(n, -1)
        logit=self.decoder[task_id](h)
        # reg = self.criterion(masks)
        return self.forward_template(logit=logit, regularizer=reg)

    def restrict_grads(self, task_id, s):
        return self.components[-1].restrict_grads(t=task_id, s=s, smax=self.args.smax, thres_cosh=self.args.thres_cosh, clipgrad=self.args.clipgrad)
    def fix_oh(self, i):     
        for p in self.decoder[i].parameters():
            p.requires_grad = False 
        try:
            if self.args.regime=='normal':
                self.optimizer, self.optimizer_structure = self.get_optimizers()
        except ValueError:
            #will throw Value error if all modules are frozen
            pass

    def on_task_learned(self, task_id):
        self.components[-1].on_task_learned(t=task_id, smax=self.args.smax)
    
    def criterion(self,masks):
        return self.components[-1].criterion(masks)      

    def constraint_embeddings(self):     
        self.components[-1].constraint_embeddings(thres_emb=self.args.thres_emb)  
