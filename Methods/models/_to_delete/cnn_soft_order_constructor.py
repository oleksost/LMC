import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.activation import Sigmoid
from .base_net_classes import SoftOrderingNet
from collections import OrderedDict

from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)


nn_device = device = 'cuda' if torch.cuda.is_available() else 'cpu'


#copied from https://github.com/FerranAlet/modular-metalearning
class torch_NN(nn.Module):
  '''
  Mimic the pytorch-maml/src/ominglot_net.py structure
  '''
  def __init__(self, inp=1, out=1, hidden=[], final_act='affine', loss_fn=None):
    super(torch_NN, self).__init__()
    self.inp = inp
    self.dummy_inp = torch.randn(8, inp, device=nn_device)
    self.out = out
    self.num_layers = len(hidden) + 1
    self.final_act = final_act
    key_words = []
    for i in range(self.num_layers):
      key_words.append('fc'+str(i))
      if i < self.num_layers-1: #final_act may not be a relu
        key_words.append('relu'+str(i))

    def module_from_name(name):
      if self.num_layers >10: raise NotImplementedError
      #TODO: allow more than 10 layers, put '_' and use split
      num = int(name[-1])
      typ = name[:-1]
      if typ == 'fc':
        inp = self.inp if num==0 else hidden[num-1]
        out = self.out if num+1==self.num_layers else hidden[num]
        return nn.Linear(inp, out)
      elif typ=='relu':
        return nn.ReLU() #removed inplace
      else: raise NotImplementedError

    self.add_module('features', nn.Sequential(OrderedDict([
      (name, module_from_name(name)) for name in key_words])))

    if self.final_act == 'sigmoid': self.add_module('fa', nn.Sigmoid())
    elif self.final_act == 'exp': self.add_module('fa', exponential())
    elif self.final_act == 'affine': self.add_module('fa', nn.Sequential())
    elif self.final_act == 'relu': self.add_module('fa', nn.ReLU())
    elif self.final_act == 'tanh': self.add_module('fa', nn.Tanh())
    else: raise NotImplementedError

  def dummy_forward_pass(self):
    '''
    Dummy forward pass to be able to backpropagate to activate gradient hooks
    '''
    return torch.mean(self.forward(self.dummy_inp))

  def forward(self, x, weights=None, prefix=''):
    '''
    Runs the net forward; if weights are None it uses 'self' layers,
    otherwise keeps the structure and uses 'weights' instead.
    '''
    if weights is None:
      x = self.features(x)
      x = self.fa(x)
    else:
      for i in range(self.num_layers):
        x = linear(x, weights[prefix+'fc'+str(i)+'.weight'],
                weights[prefix+'fc'+str(i)+'.bias'])
        if i < self.num_layers-1: x = relu(x)
      x = self.fa(x)
    return x

  def net_forward(self, x, weights=None):
    return self.forward(x, weights)

  #pytorch-maml's init_weights not implemented; no need right now.

  def copy_weights(self, net):
    '''Set this module's weights to be the same as those of 'net' '''
    for m_from, m_to in zip(net.modules(), self.modules()):
      if (isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d)
          or isinstance(m_to, nn.BatchNorm2d)):
        m_to.weight.data = m_from.weight.data.clone()
        if m_to.bias is not None:
            m_to.bias.data = m_from.bias.data.clone()

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, out_h=None, **kwargs):
        super().__init__()
        self.out_h=out_h
        self.inp = in_channels
        self.out = out_channels
        self.module = MetaSequential(OrderedDict([
            ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
            ('norm', MetaBatchNorm2d(out_channels, momentum=1., affine=True,
                track_running_stats=False)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool2d(2))
        ]))
        self.hooks = []
    def forward(self, x):
        return self.module(x)
    
    def freeze(self):
        #TODO: make freezing more comprehensible
        def hook(module, grad_input, grad_output):
            # replace gradients with zeros
            return (torch.zeros(grad_input[0].size()),torch.zeros(grad_input[1].size()),torch.zeros(grad_input[2].size()),)
        # for n, p in self.named_parameters():
        #     self.hooks.append(p.register_hook(lambda grad: torch.zeros(grad.shape)))
        class FixedParameter(nn.Parameter):
            @property
            def grad(self):
                return super().grad            
            #we dont want to set the gradient
            @grad.setter
            def grad(self, *args, **kwargs):
                self._grad = None
        #monkey patch the parameters
        for p in self.parameters():
            p.__class__ = FixedParameter
            #self.hooks.append(p.register_hook(hook)) 

    def unfreeze(self):
        for p in self.parameters():
            p.__class__ = nn.Parameter
    
def get_conv_modules_list(depth:int, in_size:int, n_modules:int, channels_in:int, channels:int, conv_kernel:int, padding:int, maxpool_kernel:int, shared_modules:bool=True):
    components = nn.ModuleList()
    out_h = in_size
    if not shared_modules:
        for l in range(depth):
            components_l = nn.ModuleList()
            for _ in range(n_modules):
                conv = conv_block(channels_in, channels, kernel_size=conv_kernel, padding=padding, bias=True)
                components_l.append(conv)
            out_h = out_h + 2 * padding - conv_kernel + 1
            out_h = (out_h - maxpool_kernel) // maxpool_kernel + 1
            components.append(components_l)
            channels_in = channels
    else:
        #create a pool of mofules shared across the layers
        for _ in range(n_modules):
            conv = conv_block(channels, channels, kernel_size=conv_kernel, stride=1, padding=padding, bias=True)
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

class SoftOrderNetConstructor(SoftOrderingNet):
    def __init__(self, 
                args,
                depth,
                i_size, 
                n_modules,
                hidden_size=64,
                num_classes_per_task=5,
                module_type = 'conv',
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
                        
        self.structure = nn.ParameterList([nn.Parameter(torch.ones_like(self.n_modules).unsqueeze(0)) for t in range(self.num_tasks)]).to(device)
        self.init_ordering()   
        self.softmax = nn.Softmax(dim=0)
     
    def init_ordering(self):
        if self.mask_activation=='softmax':
            self.activation=nn.Softmax()
            for s in self.structure:
                s.data = torch.ones(s.size(0), s.size(1)).to(s.device)
        elif self.mask_activation=='sigmoid':
            self.activation=nn.Sigmoid()
            for s in self.structure:
                    s.data = torch.zeros(s.size(0), s.size(1)).to(s.device)

     
    def get_mask(self, temp=None, train_inputs=None, task_id=0, params=None):
        temp = temp if temp is not None else self.temp
        return self.activation(self.structure[task_id]/temp)
    
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

    def _forward_indep_modules(self, X, task_id=0, params=None, ssl=False, temp=None):
        n = X.shape[0]
        c = X.shape[1]
        s = self.get_mask(temp, task_id)
        for layer in range(self.depth):
            X_tmp = 0.
            #start = time.time()
            for module in range(int(self.n_modules[layer])):
                comp = self.components[layer][module]
                X_tmp += s[module, layer] * self.dropout(comp(X, params=self.get_subdict(params, f'components.{layer}.{module}')))
            #end = time.time()
            #print(end-start)
            X = X_tmp                        
        X = X.view(n, -1)
        return self.forward_template(mask=s, hidden=X, ssl_pred=self.ssl_pred(X), logit=self.classifier(X))
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
