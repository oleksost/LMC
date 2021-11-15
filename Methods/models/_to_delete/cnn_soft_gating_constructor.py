import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_net_classes import SoftGatedNet
from collections import OrderedDict
from Methods.models.cnn_soft_order_constructor import torch_NN, conv_block

from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)


nn_device = device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_conv_modules_list(depth:int, in_size:int, n_modules:int, channels_in:int, channels:int, conv_kernel:int, padding:int, maxpool_kernel:int, shared_modules:bool=True):
    components = nn.ModuleList()
    out_h = in_size
    if not shared_modules:
        for l in range(depth):
            components_l = nn.ModuleList()
            for _ in range(n_modules):
                conv = conv_block(channels_in, channels, kernel_size=conv_kernel, padding=padding, out_h=out_h, bias=True)
                components_l.append(conv)
            out_h = out_h + 2 * padding - conv_kernel + 1
            out_h = (out_h - maxpool_kernel) // maxpool_kernel + 1
            components.append(components_l)
            channels_in = channels
    else:
        #create a pool of mofules shared across the layers
        for _ in range(n_modules):
            conv = conv_block(channels, channels, kernel_size=conv_kernel, stride=1, padding=padding, out_h=out_h, bias=True)
            components.append(conv)
        for l in range(depth):
            out_h = out_h + 2 * padding - conv_kernel + 1
            out_h = (out_h - maxpool_kernel) // maxpool_kernel + 1
    return components, out_h
        

def get_lin_modules_list(depth:int, in_size:int, n_modules:int, hidden_size:int, shared_modules:bool=True):
    components = nn.ModuleList()
    if not shared_modules:
        for l in range(depth): 
            components_l = nn.ModuleList()
            for _ in range(n_modules):
                module = torch_NN(inp=in_size, out=hidden_size, hidden=[], final_act='relu')
                components_l.append(module)
            components.append(components_l)
            in_size = hidden_size
    else:
        #create a pool of mofules shared across the layers
        for _ in range(n_modules):
            module = torch_NN(inp=hidden_size, out=hidden_size, hidden=[], final_act='relu')
            components.append(module)
    return components, hidden_size


class MN_Controller(nn.Module):
    def __init__(self, insize, out_size, sample_size=5, type='linear') -> None:
        super().__init__()
        self.type=type
        self.insize=insize
        self.out_size=out_size
        self.sample_size=sample_size
        if type=='linear':
            self.module = nn.Sequential(nn.Flatten(), nn.Linear(insize, out_size), nn.Softmax())
        elif type=='conv':
            self.module = nn.Sequential(nn.Flatten(), nn.Linear(insize, out_size), nn.Softmax())
        else:
            raise NotImplementedError            

    def forward(self, x):
        probs = self.module(x)
        samples = torch.distributions.Categorical(probs).sample(self.sample_size)
        return samples
    
class SoftGatingNetConstructor(SoftGatedNet):
    def __init__(self, 
                args,
                depth,
                i_size, 
                n_modules,
                hidden_size=64,
                num_classes_per_task=5,
                module_type = 'conv',
                structure_type='linear',
                shared_modules = False,
                ssl='None',
                mask_activation='softmax'
                ):
        super().__init__(i_size,
            depth,
            num_classes=num_classes_per_task,
            num_tasks=1,
            num_init_tasks=None,       
            init_ordering_mode='uniform')
        
        self.depth = depth
        self.channels = self.hidden_size = hidden_size
        self.shared_modules = shared_modules

        
        #n_modules - might change over the runtime     
        self.register_buffer('_n_modules', torch.tensor([float(n_modules)]*self.depth))           

        self.frozen_components = set()
        self.mask_activation = mask_activation
        self.module_type = module_type
        self.structure_type = structure_type

        if module_type=='conv':
            self.components, out_h = get_conv_modules_list(depth, i_size[1], n_modules, i_size[0], hidden_size, conv_kernel=3, padding=1, maxpool_kernel=2, shared_modules=shared_modules)
            self.decoder = self.classifier = MetaLinear(out_h*out_h*hidden_size, self.num_classes[0])

        elif module_type=='linear':
            self.components, out_h = get_lin_modules_list(depth, np.prod(i_size), n_modules, hidden_size, shared_modules=shared_modules)     
            self.decoder = self.classifier = MetaLinear(out_h, self.num_classes[0])
        self.dropout = nn.Dropout(args.dropout)
        self.temp = args.temp*torch.ones(self.depth).to(device)
        self.ssl = ssl

        if self.ssl == 'rot':           
            self.ssl_pred = MetaLinear(self.channels, 4)
        elif self.ssl == 'kmeans':   
            self.ssl_pred = MetaLinear(self.channels, self.num_classes[0])
        else:
            self.ssl_pred = torch.nn.Identity()        
        
        self.init_ordering()   
        self.softmax = nn.Softmax(dim=0)
     
    def init_ordering(self):
        ##############
        #  STRUCTURE #       
        channels = self.channels 
        self.structure = nn.ModuleList()   
        if self.controller_type=='soft_gating':     
            for t in range(1): 
                channels_in = self.channels
                structure_conv = [] 
                structure_t = nn.ModuleList()
                for i in range(self.depth):                          
                    conv = MetaConv2d(channels_in, channels, 3, padding=1)
                    structure_conv.append(MetaSequential(conv, nn.MaxPool2d(2), nn.ReLU()))
                    structure_t.append(MetaSequential(*structure_conv[::-1], nn.Flatten(), MetaLinear(channels, int(self.n_modules[i].item()))))
                    if i == self.depth-2:
                        channels_in = self.i_size[0]
                    else:
                        channels_in = channels
                self.structure.append(structure_t[::-1])
        elif self.controller_type=='modular_nets':
            in_size = self.channels
            for i in range(self.depth):        
                self.structure.append(MN_Controller(in_size,self.n_modules[i],sample_size=self.sample_size, type=self.controller_type ))
                in_size=self.components[i][0].out_h

        else:
            raise NotImplementedError
        ##############
     
    def get_mask(self, temp=None, train_inputs=None, task_id=0, params=None):
        temp = temp if temp is not None else self.temp
        return self.softmax(self.structure[task_id]/temp)
    
    def freeze_permanently_functional(self, free_layer=None):
        if not self.shared_modules:
            for l, layer in enumerate(self.components):
                if l != free_layer:
                    for c, component in enumerate(layer):
                        self.frozen_components.add(f'components.{l}.{c}')
                        component.freeze()
        else:
            for c, component in enumerate(self.components):
                #NOTE: here the comonent is completely frozen, ndependently from which layer it is being used at
                component.freeze()
                self.frozen_components.add(f'components.{c}')
                for l in range(self.depth):
                    if l !=free_layer:
                        self.frozen_components.add(f'components.{l}.{c}')
    
    def freeze_permanently_structure(self, free_layer=None):
        #dont really need it
        raise NotImplemented  

    @property
    def n_modules(self):
        return self.n_modules_per_layer
    
    @property
    def n_modules_per_layer(self):
        return self._n_modules#.cpu().numpy()
             
    def add_modules(self, at_layer=None):
        raise NotImplementedError
        def ordered_dict_mean(d):
            import copy
            res = OrderedDict()
            def sum_val_for_key(d, key, val):
                val = val.clone()
                n = 1
                for k,v in d.items():
                    kk = k[2:]
                    keyk = key[2:]
                    if k != key and kk==keyk:
                        n+=1
                        val+=v
                return val, n

            for k, v in d.items():
                kk = k[2:]
                if kk not in res.keys():
                    vv, n =sum_val_for_key(d,k,v)
                    res[kk] = vv/n
            return res
        #add one new module to each layer
        if not self.shared_modules:
            #add at each layer
            for l, layer in enumerate(self.components):
                channels_in = layer[0].inp
                channels = layer[0].out
                self._n_modules[l]+=1

                layer_mean_params = ordered_dict_mean(layer.state_dict())
                if self.module_type=='conv':
                    new_module = conv_block(channels_in, channels, kernel_size=3, padding=1).to(device)
                elif self.module_type=='linear':
                    new_module =  torch_NN(inp=channels_in, out=channels, hidden=[], final_act='relu').to(device)
                else:
                    raise NotImplementedError
                #init to be mean over all modules in layer 
                new_module.load_state_dict(layer_mean_params)
                layer.append(new_module)
        else:
            channels_in = self.components[0].inp
            channels = self.components[0].out       
            mean_params = ordered_dict_mean(self.components.state_dict())
            if self.module_type=='conv':
                new_module = conv_block(channels_in, channels, kernel_size=3, padding=1).to(device)
            elif self.module_type=='linear':
                new_module =  torch_NN(inp=channels_in, out=channels, hidden=[], final_act='relu').to(device)
            else:
                raise NotImplementedError

            #init to mean over all modules       
            new_module.load_state_dict(mean_params)
            self.components.append(new_module)
        
        #expand structural mask
        self.structure = nn.ParameterList([nn.Parameter(torch.ones(int(self.n_modules[0].item()), self.depth)) for t in range(self.num_tasks)]).to(device)
           
    def forward(self, X, task_id=0, params=None, ssl=False, temp=None, *args, **kwargs):
        temp = temp if temp is not None else self.temp
        if self.module_type=='linear':
                X=X.view(X.size(0), -1)
        if self.shared_modules:  
            if self.module_type=='conv':
                return self._forward_shared_conv(X, task_id=0, params=None, ssl=False, temp=temp)
            else:
                return self._forward_shared_lin(X, task_id=0, params=None, ssl=False, temp=temp)
        else:
            return self._forward_indep_modules(X, task_id=0, params=None, ssl=False, temp=temp)
               
    def get_structure(self, X, layer, params=None):
        return self.structure[0][layer](X)

    def _forward_indep_modules(self, X, task_id=0, params=None, ssl=False, temp=None):
        n = X.shape[0]
        c = X.shape[1]
        masks = []
        for layer in range(self.depth):
            X_tmp = 0.
            #start = time.time()
            s = F.softmax(self.get_structure(X, layer, params)/self.temp[layer], dim=1) # bs x n_comp - 128 x 4
            for module in range(int(self.n_modules[layer])):
                comp = self.components[layer][module]
                X_tmp += s[:,module].view(-1, 1, 1, 1) * self.dropout(comp(X)) #, params=self.get_subdict(params, f'components.{layer}.{module}')))
            #end = time.time()
            #print(end-start)
            masks.append(s)
            X = X_tmp                        
        X = X.view(n, -1) 
        return self.forward_template(mask=torch.stack(masks).permute(1,2,0), hidden=X, ssl_pred=self.ssl_pred(X), logit=self.classifier(X))
        # if ssl:
        #     return s, X, self.ssl_pred(X), self.classifier(X.detach(), params=self.get_subdict(params, 'classifier'))  
        # return s, X, self.ssl_pred(X), self.classifier(X, params=self.get_subdict(params, 'classifier'))  

    def _forward_shared_conv(self, X, task_id=0, params=None, ssl=False, temp=None):
        n = X.shape[0]
        c = X.shape[1]
        s = self.get_mask(temp, task_id)  
        X = F.pad(X, (0,0, 0,0, 0,self.channels-c, 0,0))

        for layer in range(self.depth):
            X_tmp = 0.
            for module in range(int(self.n_modules.item())):
                conv = self.components[module]      
                X_tmp += s[module, layer] * self.dropout(conv(X, params=self.get_subdict(params, f'components.{layer}.{module}')))
            X = X_tmp                        
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        return self.forward_template(mask=s, hidden=X, ssl_pred=self.ssl_pred(X), logit=self.classifier(X))
        # if ssl:
        #     return s, X, self.ssl_pred(X), self.classifier(X.detach(), params=self.get_subdict(params, 'classifier'))  
        # return s, X, self.ssl_pred(X), self.classifier(X, params=self.get_subdict(params, 'classifier'))  
    
    def _forward_shared_lin(self, X, task_id=0, params=None, ssl=False, temp=None):
        raise NotImplementedError
        n = X.shape[0]
        c = X.shape[1]
        s = self.get_mask(temp, task_id)
        X = X.view(n, -1)
        for layer in range(self.depth):
            X_tmp = 0.
            for module in range(int(self.n_modules.item())):
                lin = self.components[module]        
                X_tmp += s[module, layer] * self.dropout(lin(X, params=self.get_subdict(params, f'components.{layer}.{module}')))
            X = X_tmp                        
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        return self.forward_template(mask=s, hidden=X, ssl_pred=self.ssl_pred(X), logit=self.classifier(X))
        # if ssl:
        #     return s, X, self.ssl_pred(X), self.classifier(X.detach(), params=self.get_subdict(params, 'classifier')), None
        # return s, X, self.ssl_pred(X), self.classifier(X, params=self.get_subdict(params, 'classifier')), None
