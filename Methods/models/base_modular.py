
import copy
from ssl import Options
import torch
import numpy as np
import torch.nn as nn 
from collections import deque
from typing import Union, Dict, Iterable, Optional
from runstats import Statistics
import torch.nn.functional as F
from dataclasses import dataclass       
from collections import OrderedDict, namedtuple
from simple_parsing import ArgumentParser, choice
from Utils.utils import RunningStats, DequeStats, cosine_rampdown, standardize
from abc import ABCMeta, abstractmethod
from Methods.models.resnet import BasicBlock, conv1x1

device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def bn_eval(module:nn.Module, freeze=True):
    for layer in module.children():      
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
          if freeze:
            layer.eval()
          else:
            layer.train()
        elif isinstance(layer.children(), Iterable):
            bn_eval(layer, freeze)

class FixedParameter(nn.Parameter):
            @property
            def grad(self):
                return super().grad            
            #we dont want to set the gradient
            @grad.setter
            def grad(self, *args, **kwargs):
                self._grad = None
                
class Conv_Module(nn.Module):                     
  def __init__(self,in_channels, out_channels, kernel_size, track_running_stats, pooling_kernel, pooling_stride=None, pooling_padding=0, affine_bn =True, momentum=1, decode=False, use_bn=True, **kwargs):
    super().__init__()
    self.module = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)),  
                ('norm', nn.BatchNorm2d(out_channels, momentum=momentum, affine=affine_bn,
                    track_running_stats=track_running_stats)) if use_bn else ('norm', nn.Identity()),
                ('relu', nn.ReLU()),
                ('pool', nn.MaxPool2d(pooling_kernel,pooling_stride, pooling_padding))
            ]))
    # self.decoder=None
    # if decode:
    #   # H_out ​= (H_in​−1)*stride[0] − 2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1
    #   self.decoder=nn.Sequential(OrderedDict([
    #             ('conv_t1', nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size-1, padding=0, stride=1)),  
    #             ('norm', nn.BatchNorm2d(out_channels, momentum=1., affine=affine_bn,
    #                 track_running_stats=track_running_stats)) if use_bn else ('norm', nn.Identity()),
    #             ('relu_2', nn.ReLU()),
    #             ('conv_t2', nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size-1, padding=1, stride=2)),  
    #             ('relu_2', nn.Sigmoid()),

    #         ]))
  def forward(self, x):
    return self.module(x)
    # d = None
    # if self.decoder is not None:
    #   d = self.decoder(e)
    # return

  # def decode(self, x):
  #   return self.decoder(x)

########################
#copied from https://github.com/FerranAlet/modular-metalearning

class exponential(nn.Module):
    def __init__(self):
        super(exponential, self).__init__()
    def forward(self, x):
        return torch.exp(x)

class torch_NN(nn.Module):
  '''
  Mimic the pytorch-maml/src/ominglot_net.py structure
  '''
  def __init__(self, inp=1, out=1, hidden=[], final_act='affine', bias=True, loss_fn=None):
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
        return nn.Linear(inp, out, bias=bias)
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
########################
class conv_block_base(nn.Module):           
    @dataclass
    class Options:
        kernel_size: int = 3 #conv kernel size
        padding: int = 1 #padding
        maxpool_kernel: int = 2 #maxpool kernel size
        maxpool_stride:Optional[int]=None #maxpool stride
        maxpool_padding:int=0 #maxpool padding

        dropout:float=0. #dropout probability
        track_running_stats_bn: bool = 0 #wether to track running stats of the batch norm
        affine_bn: bool =1 #affine parameter of the batchnorm
        use_bn: bool = 1 # -
        momentum_bn = 1. # -

        keep_bn_in_eval_after_freeze: bool = 0 #-

        #for the filterwise module
        tied_weights: bool = 1 # - 
        decode_channelwise: bool = 1 # -             

        #for HAT        
        n_tasks: Optional[int] = None #number of tasks in the sequence
    def __init__(self, in_channels, out_channels, i_size, name=None, module_type='conv', n_layers=1, stride=1, bias=True, deeper=False,
                            options:Options=Options(), n_classes=None,
                            **kwargs):
        super().__init__()
        self.args:conv_block_base.Options=copy.copy(options)
        self.name=name          
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.module_kwargs = kwargs
        self.dropout = nn.Dropout(self.args.dropout)
        self.i_size=i_size
        self.deeper=deeper
        self.module_type=module_type
        self.depth=1
        
        out_h = self.i_size
        if module_type == 'conv':             
          if not deeper:
            self.module = Conv_Module(self.in_channels, self.out_channels, self.args.kernel_size, self.args.track_running_stats_bn, self.args.maxpool_kernel, pooling_stride=self.args.maxpool_stride, 
                                      pooling_padding=self.args.maxpool_padding, padding=self.args.padding, affine_bn=self.args.affine_bn, momentum=self.args.momentum_bn, use_bn=self.args.use_bn, stride=stride, bias=bias, **self.module_kwargs).to(device)
            #TODO: thos doesnt account for stride (assumes stride = 1), stride is other then 1 for Resnet, but resnet doesnt use variable out_h
            out_h = out_h + 2 * self.args.padding - self.args.kernel_size + 1  
            out_h = (out_h - self.args.maxpool_kernel) // self.args.maxpool_kernel + 1
          else:
            module1 = Conv_Module(self.in_channels, self.out_channels, self.args.kernel_size, self.args.track_running_stats_bn, self.args.maxpool_kernel, padding=self.args.padding, affine_bn=self.args.affine_bn, momentum=self.args.momentum_bn, use_bn=self.args.use_bn, **self.module_kwargs).to(device)
            out_h = out_h + 2 * self.args.padding - self.args.kernel_size + 1  
            out_h = (out_h - self.args.maxpool_kernel) // self.args.maxpool_kernel + 1
            module2 = Conv_Module(self.out_channels, self.out_channels, self.args.kernel_size, self.args.track_running_stats_bn, self.args.maxpool_kernel, padding=self.args.padding, affine_bn=self.args.affine_bn, momentum=self.args.momentum_bn, use_bn=self.args.use_bn, **self.module_kwargs).to(device)
            self.module=nn.Sequential(module1,module2)
            out_h = out_h + 2 * self.args.padding - self.args.kernel_size + 1  
            out_h = (out_h - self.args.maxpool_kernel) // self.args.maxpool_kernel + 1

        elif module_type == 'linear':
            self.module = nn.Sequential(OrderedDict([
                              ('flatten', nn.Flatten().to(device)), 

                              ('lin', torch_NN(inp=in_channels, out=out_channels, hidden=[], final_act='relu').to(device)),
                              ('norm', nn.BatchNorm1d(out_channels, momentum=self.args.momentum_bn, affine=self.args.affine_bn,
                                  track_running_stats=self.args.track_running_stats_bn)) if self.args.use_bn else ('norm', nn.Identity()),
                                
                          ]))
            out_h = 1
        elif module_type == 'resnet_block':
            def _make_layer(in_channels, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
                block=BasicBlock
                norm_layer = nn.BatchNorm2d
                downsample = None
                previous_dilation = 1
                if dilate:
                    self.dilation *= stride
                    stride = 1
                if stride != 1 or in_channels != planes * block.expansion:
                    downsample = nn.Sequential(
                        conv1x1(in_channels, planes * block.expansion, stride),
                        norm_layer(planes * block.expansion),
                    )

                layers = []
                layers.append(block(in_channels, planes, stride, downsample, 1,
                                    64, previous_dilation, norm_layer))
                in_channels = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(in_channels, planes, groups=1,
                                        base_width=64, dilation=1,
                                        norm_layer=norm_layer))

                return nn.Sequential(*layers)
            self.module = _make_layer(in_channels, planes=out_channels, blocks=int(n_layers/2), stride=stride)        

        elif module_type == 'expert': 
            self.module = Expert(depth=4, i_size=self.i_size, channels=self.in_channels, hidden_size=self.out_channels, num_classes=n_classes, module_options=self.args)
            out_h = self.module.out_h
            self.depth=4
        # elif module_type == 'invertible':
        #     self.module = MintNet_BasicBlock()

        self.out_h=out_h
        self.register_buffer('module_learned_buffer', torch.tensor(0.))
        #wether batch nroms should always be in eval mode flag  
        self.register_buffer('bn_eval_buffer', torch.tensor(0.))
        self.out_channels = out_channels

    def reset(self):
        self.module_learned_buffer = torch.tensor(0.)            
    
    @property
    def module_learned(self):
        return int(self.module_learned_buffer.item())

    @module_learned.setter
    def module_learned(self, v):  
        self.module_learned_buffer.data = torch.tensor(float(v))
    
    def forward(self, x):
        return self.dropout(self.module(x))
      
    def freeze_module(self):        
          self.module_learned_buffer = torch.tensor(1.)  
          if self.args.keep_bn_in_eval_after_freeze:  
                bn_eval(self)
                # self.bn_eval_buffer=torch.tensor(1.)    
          for p in self.parameters():
              #setting requires_grad to False would prevent updating in the inner loop as well
              p.requires_grad = False
              #this would prevent updating in the outer loop:
              p.__class__ = FixedParameter      
          print(f'freezing module {self.name}')
          return True
    
    def train(self, *args, **kwargs):                                   
      if self.args.keep_bn_in_eval_after_freeze and self.module_learned_buffer:
          r = super().train(*args, **kwargs)
          bn_eval(self)
          return r
      else:
          return super().train(*args, **kwargs)

class Expert(nn.Module):
        def __init__(self, depth: int, net_arch:int, i_size: tuple, channels, hidden_size, num_classes, module_type, module_options:conv_block_base.Options=conv_block_base.Options(), Module=conv_block_base):
            super().__init__()
            self.depth = depth 
            self.i_size = i_size
            self.num_classes = num_classes
            self.hidden_size = hidden_size


            components = []    
            channels_in = channels
            out_h=self.i_size      
            #TODO: make this implementation here more comprehensible
            #only important for resnet
            stride=1
            bias=True
            n_layers=1
            ##################
            dropout_before=module_options.dropout
            for l in range(self.depth):
                if net_arch=='alexnet':
                    ksizes=[4,3,2,0,0]          
                    #can resemble the Alexnet architecture
                    # if using module_type conv
                    if l>0:           
                        hidden_size*=2
                    if ksizes[l]>0:              
                        module_options.kernel_size=ksizes[l]
                    else:
                        module_type='linear'
                        hidden_size=2048
                        channels_in = out_h*out_h*channels_in
                        out_h=1
                        module_options.dropout=0.5

                elif net_arch=='resnet':
                  layers=[1,4,4,4,2,2]
                  strides=[2,1,2,2,2,1]
                  module_type='resnet_block'
                  n_layers=layers[l]
                  stride=strides[l]
                  bias=True
                  # module_options.kernel_size=3
                  if n_layers==1:
                    module_type='conv'
                    # module_options.padding=1
                    module_options.use_bn=1
                    bias=False
                    module_options.maxpool_kernel=3
                    module_options.maxpool_stride=2
                    module_options.maxpool_padding=1
                    #out_h is  nto important for resnet here

                conv = Module(channels_in, hidden_size, out_h, name=f'components.{l}', module_type=module_type, n_layers=n_layers, stride=stride, bias=bias, options=module_options)
                module_options.dropout=dropout_before
                components.append(conv)
                channels_in = hidden_size
                out_h=conv.out_h
            self.components = nn.Sequential(*components)
            self.out_h=out_h
            self.pooling=nn.Identity()
            if net_arch=='resnet':
              pool_kernel=1
              self.pooling=nn.AdaptiveAvgPool2d((pool_kernel, pool_kernel))
              out_h=pool_kernel
            self.decoder = nn.Linear(channels_in*out_h*out_h, self.num_classes)

        def forward(self, x, *kwargs):
            n = x.size(0)     
            e = self.components(x)
            e=self.pooling(e)
            e_flat = e.view(n, -1)
            return e, self.decoder(e_flat)
        
        def freeze_functional(self, requieres_grad=True):
            for p in self.parameters():
                #setting requires_grad to False would prevent updating in the inner loop as well
                p.requires_grad = requieres_grad
                #this would prevent updating in the outer loop:
                p.__class__ = FixedParameter
                #self.hooks.append(p.register_hook(hook))  

class ModularBaseNet(nn.Module):

    @dataclass 
    class Options():      
      n_modules:int=1 #numbe rof modules to begin with 

      depth: int = 4      #network depth
      lr: float = 0.01 #meta-learning rate (outer loop)   

      module_init: str = choice('mean', 'identical', 'existing', 'none', 'previously_active', 'most_likely', default='previously_active')#how to init the modules

      #regime: meta pr normal
      regime:str=choice('meta', 'normal', default='meta')
      #mask activation
      mask_activation: str = 'relu'

      #tempreture for masking
      temp:float = 1.

      #minimal tempreture
      min_temp: float = 1.
      
      #net achitecture
      net_arch:str = choice('alexnet', 'resnet', 'none', default='none')

      #max pooling kerne size
      maxpool_kernel:int=2

      #module tpye
      module_type: str =choice('conv', 'linear', 'resnet_block', 'invertible', 'expert', default='conv') #-

      #deeper firt layer
      depper_first_layer: bool = 0 #if 'True', modules in the first layer will be composed of two modules

      wdecay: float = 0. #weight decay for Adam optimizer

      #multihead mode, is none use single-head
      multihead:bool=choice('none', 'modulewise', 'usual', 'gated_linear', 'gated_conv', default='none')

    def __init__(self, options:Options=Options(), i_size:int = 28, channels:int=1, hidden_size:int=64, num_classes:int = 5):
        super().__init__()
        self.args: ModularBaseNet.Options=copy.copy(options)
        ##########################################
        #fields returned by forward pass
        fields = ['mask','mask_bf_act','hidden', 'ssl_pred', 'logit', 'regularizer', 'info']
        self.forward_template = namedtuple('forward', fields, defaults=(None,) * len(fields))
        ##########################################
        ##############################
        #buffers
        #n_modules - might change over the runtime
        self.register_buffer('_n_modules', torch.tensor([float(self.args.n_modules)]))        
        # minimal tempreture used: in case we are annealing the tempreture we should stre the minimal one
        self.register_buffer('min_temp_buffer', torch.tensor(float(self.args.temp)))
        ##############################
        self.hidden_size=hidden_size
        self.i_size=i_size
        self.num_classes=num_classes
        self.channels=channels
        self.relu = nn.ReLU() 
        self.depth = self.args.depth
        self.maxpool = nn.MaxPool2d(self.args.maxpool_kernel)
        self.components: nn.ModuleList = nn.ModuleList()
        self.per_layer_module_initialization = [None]*self.args.depth
        self._learnable_parameters = None
        self._learnable_named_parameters = None

        self.register_buffer('_log_at_this_iter_flag_buffer', torch.tensor(0.))
      
    @property
    def log_at_this_iter_flag(self):
      return int(self._log_at_this_iter_flag_buffer)
      
    @log_at_this_iter_flag.setter
    def log_at_this_iter_flag(self, v):
      if v:
        self._log_at_this_iter_flag_buffer=torch.tensor(1.)
      else:
        self._log_at_this_iter_flag_buffer=torch.tensor(0.)

    @property
    def min_temp(self):
        return self.min_temp_buffer
    @min_temp.setter
    def min_temp(self, v:torch.Tensor):
      self.min_temp_buffer = v

    @property
    def n_modules(self):
        return int(self._n_modules[0].item())
    
    @n_modules.setter
    def n_modules(self, v): 
        if isinstance(v, torch.Tensor):
            self._n_modules = v
        else:
            self._n_modules[0] = torch.tensor(float(v))

    def _reset_references_to_learnable_params(self):
        self._learnable_parameters = list(filter(lambda x: x.requires_grad, self.parameters()))
        self._learnable_named_parameters = list(filter(lambda x: x[1].requires_grad, self.named_parameters()))

    def fix_oh(self, *args, **kwargs):
      pass  
    
    @abstractmethod
    def init_modules(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_modules(self, at_layer, *args, **kwargs):
        pass
    
    @property
    def learnable_parameters(self):
        if self._learnable_parameters is None:
            self._reset_references_to_learnable_params()
        return self._learnable_parameters
    
    @property
    def learnable_named_parameters(self):
        if self._learnable_named_parameters is None:
            self._reset_references_to_learnable_params()
        return self._learnable_named_parameters
            
    def load_state_dict(self, state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], strict: bool):
        #TODO: test this
        k=None
        if '_n_modules' in state_dict:
            k='_n_modules'
        elif 'n_modules' in state_dict:
            k='n_modules'
        if k is not None:                    
            for l, n_modules_in_layer in enumerate(state_dict[k]):
                # print(int(self._n_modules[l].item()))
                for _ in range(int(n_modules_in_layer.item()) - int(self._n_modules[l].item())):
                    self.add_modules(at_layer=l)

        components = filter( lambda x: '_backup' in x, state_dict.keys())

        for m in list(components):
            layer, comp = [int(i) for i in m if i.isdigit()][:2]
            if 'functional' in m:
                self.components[layer][comp]._maybe_backup_functional(verbose=False)
            if 'inv' in m:
                self.components[layer][comp]._maybe_backup_structural(verbose=False)               
                     
        return super().load_state_dict(state_dict,strict)


    
