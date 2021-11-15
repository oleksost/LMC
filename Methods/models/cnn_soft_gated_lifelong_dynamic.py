from networkx.algorithms import components
import torch
import copy           
from simple_parsing import choice
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .base_net_classes import SoftGatedNet
from .base_modular import conv_block_base, ModularBaseNet, bn_eval

# addopted from https://github.com/Lifelong-ML/Mendez2020Compositional
device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CNNSoftGatedLLDynamic(SoftGatedNet):
    @dataclass
    class Options(ModularBaseNet.Options):  
        depth: int = 4 # -  
        num_tasks: Optional[int] = None # -
        # conv_kernel: int = 3 # -
        # maxpool_kernel: int =2 #-
        # padding: int = 0 #-
        num_init_tasks: Optional[int] = None #-
        max_components: int =-1 #-  
        init_ordering_mode: str = 'random' #-
        use_single_controller: int = 0 #-
        keep_bn_in_eval_after_freeze: int = 0 #-
        seperate_pool_per_layer: int = 0 # -
        #-
        searchspace: str = choice('topdown', 'bottomup', default='topdown')
        #-
        dropout: float = 0. 
        #-
        padd_input: bool = 0 
        #-
        single_oh_controller: int = 1

    def __init__(self, options: Options = Options(),        
                       module_options: conv_block_base.Options = conv_block_base.Options(), 
                       i_size: int = 28, channels:int = 1, hidden_size = 64, num_classes: int = 5
                ):
        assert options.num_tasks is not None
        super().__init__(i_size,
            options.depth,
            num_classes,
            options.num_tasks,                    
            num_init_tasks=options.num_init_tasks,    
            init_ordering_mode=options.init_ordering_mode,
            device=device)
        self.device=device
        print(self.device)
        self.args:CNNSoftGatedLLDynamic.Options=copy.copy(options)
        self.hidden_size  =hidden_size
        self.channels = channels
        # self.conv_kernel = conv_kernel
        # self.padding = padding
        self.max_components = self.args.max_components if self.args.max_components != -1 else np.inf
        self.keep_bn_in_eval_after_freeze = self.args.keep_bn_in_eval_after_freeze
        self.num_components = self.depth
        self.freeze_encoder = True
        self.module_options=module_options

        self.components = nn.ModuleList()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(module_options.maxpool_kernel)
        self.dropout = nn.Dropout(self.args.dropout)
        self.last_learned_task=0
        
        out_h = self.i_size[0] 
        if self.args.seperate_pool_per_layer:
            if self.args.padd_input:
                for i in range(self.depth):
                    layer_pool = nn.ModuleList()
                    conv = conv_block_base(self.hidden_size, self.hidden_size, out_h, name=f'components.{i}.{0}', options=module_options) #(channels, channels, conv_kernel, padding=padding)
                    layer_pool.append(conv)
                    out_h = conv.out_h # (out_h - maxpool_kernel) // maxpool_kernel + 1
                    self.components.append(layer_pool)
            else:
                for i in range(self.depth):
                    layer_pool = nn.ModuleList()
                    conv = conv_block_base(channels, self.hidden_size, out_h, name=f'components.{i}.{0}', options=module_options) #(channels, channels, conv_kernel, padding=padding)
                    layer_pool.append(conv)
                    out_h = conv.out_h # (out_h - maxpool_kernel) // maxpool_kernel + 1
                    self.components.append(layer_pool)
                    channels=self.hidden_size
        else:
            for i in range(self.depth):
                conv = conv_block_base(self.hidden_size, self.hidden_size, out_h, name=f'components.{i}', options=module_options) #(channels, channels, conv_kernel, padding=padding)
                self.components.append(conv)
                out_h = conv.out_h # (out_h - maxpool_kernel) // maxpool_kernel + 1

        self.decoder = nn.ModuleList()
        self.binary = False
        for t in range(self.num_tasks):
            if self.num_classes[t] == 2: self.binary = True      
            decoder_t = nn.Linear(out_h * out_h * self.hidden_size, self.num_classes[t] if self.num_classes[t] != 2 else 1)
            self.decoder.append(decoder_t)

        self.structure = nn.ModuleList()
        self.structure_head = []
        if not self.args.use_single_controller:
            n_controllers=self.num_tasks
        else:
            n_controllers=1
        convs=[]
        fcs=[]
        if self.args.seperate_pool_per_layer:
            for t in range(self.num_tasks):
                # channels=self.channels
                structure_head_t = []
                structure_t = nn.ModuleList()
                structure_conv = []
                if self.args.padd_input:
                    if t<n_controllers:
                        for i in range(self.depth):         
                            conv = nn.Conv2d(self.hidden_size, self.hidden_size, module_options.kernel_size, padding=module_options.padding) #conv_kernel, padding=padding)
                            convs.append(conv)
                            fc = nn.Linear(out_h * out_h * self.hidden_size, 1)
                            fcs.append(fc)
                            structure_conv.append(nn.Sequential(conv, self.maxpool, self.relu))
                            structure_head_t.append(fc)
                            structure_t.append(nn.Sequential(*structure_conv, nn.Flatten(), fc))
                        self.structure.append(structure_t[::-1])
                    else:
                        for i in range(self.depth):                       
                            # fc = nn.Linear(out_h * out_h * self.hidden_size, 1)
                            structure_conv.append(nn.Sequential(convs[i], self.maxpool, self.relu))
                            structure_head_t.append(fcs[i])
                            structure_t.append(nn.Sequential(*structure_conv, nn.Flatten(), fcs[i]))
                        self.structure.append(structure_t[::-1])
                else:
                    if t<n_controllers:
                        for i in range(self.depth):
                            if i==self.depth-1:
                                channels=self.channels
                            else:
                                channels=self.hidden_size
                            conv = nn.Conv2d(channels, self.hidden_size, module_options.kernel_size, padding=module_options.padding) #conv_kernel, padding=padding)
                            convs.append(conv)
                            fc = nn.Linear(out_h * out_h * self.hidden_size, 1)
                            structure_conv.insert(0, nn.Sequential(conv, self.maxpool, self.relu))
                            structure_head_t.append(fc)
                            structure_t.append(nn.Sequential(*structure_conv, nn.Flatten(), fc))
                        self.structure.append(structure_t[::-1])
                    else:
                        for i in range(self.depth):                       
                            fc = nn.Linear(out_h * out_h * self.hidden_size, 1)
                            structure_conv.insert(0, nn.Sequential(convs[i], self.maxpool, self.relu))
                            structure_head_t.append(fc)
                            structure_t.append(nn.Sequential(*structure_conv, nn.Flatten(), fc))
                        self.structure.append(structure_t[::-1])
                
                self.structure_head.append(structure_head_t[::-1])
            if self.args.use_single_controller:
                assert self.structure[0][0][0][0]==self.structure[1][0][0][0]
                assert self.structure[1][0][0][0]==self.structure[3][0][0][0]
        else:
            for t in range(self.num_tasks):
                structure_head_t = []
                structure_t = nn.ModuleList()
                structure_conv = []
                if t<n_controllers:
                    for i in range(self.depth):
                        conv = nn.Conv2d(self.hidden_size, self.hidden_size, module_options.kernel_size, padding=module_options.padding) #conv_kernel, padding=padding)
                        convs.append(conv)
                        fc = nn.Linear(out_h * out_h * self.hidden_size, self.depth)
                        structure_conv.append(nn.Sequential(conv, self.maxpool, self.relu))
                        structure_head_t.append(fc)
                        fcs.append(fc)
                        structure_t.append(nn.Sequential(*structure_conv, nn.Flatten(), fc))
                    self.structure.append(structure_t[::-1])
                else:
                    for i in range(self.depth):
                        fc = nn.Linear(out_h * out_h * self.hidden_size, self.depth)
                        structure_conv.append(nn.Sequential(convs[i], self.maxpool, self.relu))
                        structure_head_t.append(fcs[i])
                        structure_t.append(nn.Sequential(*structure_conv, nn.Flatten(), fcs[i]))
                    self.structure.append(structure_t[::-1])
                self.structure_head.append(structure_head_t[::-1])
            if self.args.use_single_controller:
                assert self.structure[0][0][0][0]==self.structure[1][0][0][0]
                assert self.structure[1][0][0][0]==self.structure[3][0][0][0]
        
        
                
        self.init_ordering()
        self.softmax = nn.Softmax(dim=1)
        self.to(device)
    @property
    def n_modules(self):
        if not self.args.seperate_pool_per_layer:
            return len(self.components)
        return sum([len(l) for l in self.components])
    
    def n_modules_at_layer(self, l):
        if not self.args.seperate_pool_per_layer:
            return self.n_modules
        return len(self.components[l])
    
    def add_tmp_module(self, task_id): 
        if self.num_components < self.max_components:
            for t in range(task_id, self.num_tasks):
                for k in range(self.depth):
                    size = self.structure_head[t][k].in_features
                    new_node = nn.Linear(size, 1).to(device)
                    if t < task_id:
                        new_node.weight.data[:] = -np.inf
                        new_node.bias.data[:] = -np.inf
                    else:
                        assert self.structure_head[t][k].weight.grad is None
                        assert self.structure_head[t][k].bias.grad is None
                    self.structure_head[t][k].weight.data = torch.cat((self.structure_head[t][k].weight.data, new_node.weight.data), dim=0)
                    self.structure_head[t][k].bias.data = torch.cat((self.structure_head[t][k].bias.data, new_node.bias.data), dim=0)
            conv = conv_block_base(self.hidden_size, self.hidden_size, self.i_size[0], name=f'components.{len(self.components)+1}', options=self.module_options).to(device) #nn.Conv2d(self.channels, self.channels, self.conv_kernel, padding=self.padding).to(device)
            self.components.append(conv)
            self.num_components += 1

    def freeze_modules(self, freeze=True):
        for param in self.components.parameters():
            param.requires_grad = not freeze
            if freeze:
                param.grad = None
            if self.keep_bn_in_eval_after_freeze:
                bn_eval(self.components, freeze)

    def hide_tmp_module(self):
        self.num_components -= 1

    def recover_hidden_module(self):
        self.num_components += 1

    def remove_tmp_module(self):  
        self.components = self.components[:-1] 
        self.num_components = len(self.components)
        for t in range(self.num_tasks):
            for k in range(self.depth):
                self.structure_head[t][k].weight.data = self.structure_head[t][k].weight.data[:self.num_components, :]
                self.structure_head[t][k].bias.data = self.structure_head[t][k].bias.data[:self.num_components]        
    
    def add_modules(self,task_id, at_layer, state_dict=None):
        if self.num_components < self.max_components:
            if self.args.use_single_controller:
                t=task_id
                for k in range(self.depth):
                    if k==at_layer:         
                        size = self.structure_head[t][k].in_features
                        new_node = nn.Linear(size, 1).to(device)
                        # if t < task_id:
                        #     new_node.weight.data[:] = -np.inf
                        #     new_node.bias.data[:] = -np.inf
                        # else:
                        if self.args.single_oh_controller:
                            self.structure_head[t][k].weight.data=self.structure_head[t-1][k].weight.data
                            self.structure_head[t][k].bias.data=self.structure_head[t-1][k].bias.data
                        assert self.structure_head[t][k].weight.grad is None
                        assert self.structure_head[t][k].bias.grad is None
                        self.structure_head[t][k].weight.data = torch.cat((self.structure_head[t][k].weight.data, new_node.weight.data), dim=0)
                        self.structure_head[t][k].bias.data = torch.cat((self.structure_head[t][k].bias.data, new_node.bias.data), dim=0)
                        assert self.structure[0][0][0][0]==self.structure[1][0][0][0]
                        assert self.structure[1][0][0][0]==self.structure[3][0][0][0]
            else:
                for t in range(task_id, self.num_tasks):
                    for k in range(self.depth):
                        if k==at_layer:     
                            size = self.structure_head[t][k].in_features
                            new_node = nn.Linear(size, 1).to(device)
                            if t < task_id:
                                new_node.weight.data[:] = -np.inf
                                new_node.bias.data[:] = -np.inf
                            else:
                                assert self.structure_head[t][k].weight.grad is None
                                assert self.structure_head[t][k].bias.grad is None
                            self.structure_head[t][k].weight.data = torch.cat((self.structure_head[t][k].weight.data, new_node.weight.data), dim=0)
                            self.structure_head[t][k].bias.data = torch.cat((self.structure_head[t][k].bias.data, new_node.bias.data), dim=0)

        channels = self.components[at_layer][-1].in_channels
        conv = conv_block_base(channels, self.hidden_size, self.i_size[0], name=f'components.{at_layer}.{len(self.components[at_layer])}', options=self.module_options).to(device) #nn.Conv2d(self.channels, self.channels, self.conv_kernel, padding=self.padding).to(device)
        if state_dict is not None:
            conv.load_state_dict(state_dict)
        self.components[at_layer].append(conv)
        self.num_components += 1
    
    def init_structure(self, task_id, structure:List=None):
        #strcture should be given
        #1. create a random model     
        # args = copy.deepcopy(self.args)
        # args.num_tasks=1
        temp_model = copy.deepcopy(self) #CNNSoftGatedLLDynamic(args, module_options=self.module_options, i_size=self.i_size, 
                     #          channels=self.channels, hidden_size=self.hidden_size, num_classes=self.num_classes).to(device)
        #2. Load selected modules into the model
        for l, module in enumerate(structure):
            if task_id>0:   
                for m in temp_model.components[l]:
                    m.freeze_module()
            if module==1:     
                temp_model.add_modules(task_id, at_layer=l)
        temp_model.to(device)
        # temp_model.structure = structure
        #return a model with selected modules
        return temp_model


    def create_search_space(self, task_id):
        if self.args.searchspace=='topdown':
            #1. most likeliy structure sofar           
            best_structure = [0 for _ in range(self.depth)]        
            best_model = self.init_structure(task_id, best_structure).to(device)
            yield best_model, best_structure #[0 for _ in range(self.depth)]
            new_structure=copy.copy(best_structure) #[0 for _ in range(self.depth)]
            if task_id>0:
                for i in range(self.depth):
                    l = self.depth-1-i
                    if self.components[l][-1].module_learned:
                        new_structure[l]+=1  
                        model = self.init_structure(task_id, structure=new_structure).to(device)
                        yield model, new_structure
        elif self.args.searchspace=='bottomup':
            #1. most likeliy structure sofar                                   
            best_structure = [0 for _ in range(self.depth)]
            best_model = self.init_structure(task_id, best_structure).to(device)
            yield best_model, best_structure #[0 for _ in range(self.depth)]
            new_structure=copy.copy(best_structure) #[0 for _ in range(self.depth)]
            for i in range(self.depth):
                l = i
                if self.components[l][-1].module_learned:
                    new_structure[l]=len(self.components[l])     
                    model = self.init_structure(task_id, structure=new_structure).to(device)
                    yield model, new_structure
        else:
            raise NotImplemented

    def forward(self, X, task_id, return_mask=False, *args, **kwargs):
        n = X.shape[0]
        c = X.shape[1]           
        if task_id>self.last_learned_task and self.training:
            self.last_learned_task=task_id
        if not self.args.seperate_pool_per_layer or self.args.padd_input:    
            X = F.pad(X, (0,0, 0,0, 0,self.hidden_size-c, 0,0))
        mask=[]
        for k in range(self.depth):
            X_tmp = 0.
            if self.args.single_oh_controller and not self.training:
                #at test time select last controller
                s = self.structure[self.last_learned_task][k](X)
            else:
                s = self.structure[task_id][k](X) #if not self.args.use_single_controller else self.structure[-1][k](X)            
            if self.args.seperate_pool_per_layer:
                s = self.softmax(s[:, :self.n_modules_at_layer(k)])   # include in the softmax only num_components (i.e., if new component is hidden, ignore it)
                for j in range(min(len(self.components[k]), s.shape[1])):
                    conv = self.components[k][j]
                    X_tmp += s[:, j].view(-1, 1, 1, 1) * self.dropout(conv(X))
            else:
                s = self.softmax(s[:, :self.num_components])   # include in the softmax only num_components (i.e., if new component is hidden, ignore it)
                for j in range(min(self.num_components, s.shape[1])):
                    conv = self.components[j]
                    X_tmp += s[:, j].view(-1, 1, 1, 1) * self.dropout(conv(X))
            X = X_tmp
            mask.append(s.mean(0))
        X = X.view(-1, X.shape[1] * X.shape[2] * X.shape[3])
        if return_mask:
            return self.decoder[task_id](X), torch.stack(mask).T.cpu()
        return self.decoder[task_id](X)     