import copy
import torch
import scipy 
import torch.nn as nn 
from typing import List, Union, Dict
from dataclasses import dataclass
from .base_modular import ModularBaseNet, conv_block_base

from simple_parsing import choice
device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ComponentList(nn.ModuleList):
    def _init__(self, *args, **kwargs):
        super(ComponentList).__init__(*args,**kwargs)
    
    @property
    def all_modules_learned(self):
        return not any(not x.module_learned for x in self)

    @property
    def optimizer_update_needed(self):
        ret = 0
        for m in self:
            if hasattr(m, 'optimizers_update_needed_buffer'):
                if m.optimizers_update_needed_buffer:
                    m.optimizers_update_needed_buffer=torch.tensor(0.)
                    ret = 1
        return ret
 
class MNTDP_net(ModularBaseNet):
    @dataclass
    class Options(ModularBaseNet.Options):
        #use deviation from mean for structural selection
        mask_with: bool = choice('str_act','z_score', 'deviation', default='str_act')
        #if true use multihead regime (task id should be passed to the forward using test-time and train-time oracle parameters)
        multihead:bool=choice('none', 'modulewise', 'usual', default='none')
        #learnign rate used in case of offline ('normal') training
        lr:float = 0.001
        #searchspae of MNTDP
        searchspace: str = choice('topdown', 'bottomup', default='topdown')
        # if 'True' infers task uysing entropy of the output head
        entropy_task_inf:int = 0

    def __init__(self, options:Options = Options(), module_options:conv_block_base.Options=conv_block_base.Options(), i_size: int = 28, channels:int = 1, hidden_size=64, num_classes:int=5):
        super(MNTDP_net, self).__init__(options, i_size, channels, hidden_size, num_classes)
        
        self.args: MNTDP_net.Options = copy.copy(options)
        # self.bn = not module_options.batchnorm_inside_of_module
        self.module_options=module_options      
        ##############################
        #n_modules - might change over the runtime
        self.register_buffer('_n_modules', torch.tensor([float(self.n_modules)]*self.depth))    
        self.register_buffer('_steps_since_last_addition', torch.tensor(0.))   
        ##############################
        self.init_modules()    
        self.modules_to_unfreeze_func=[]
        self.structure: List = [0 for _ in range(self.depth)]
        self.structure_pool: List[List] = []
    
    def add_structure_to_pool(self, new_structure:List):     
        # new_structure=[int(n+nm-1) for n, nm in zip(new_structure, self.n_modules.cpu().numpy())]
        self.structure_pool.append(new_structure)
    
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

    def init_modules(self, structure:List=None):

        channels_in = self.channels
        if self.args.module_type=='linear':
            channels_in = self.i_size * self.i_size * self.channels  

        out_h=self.i_size  
        deeper=0
        hidden_size=self.hidden_size

        if len(self.components)==0:
            for i in range(self.depth):
                components_l = ComponentList()
                # if i>0:
                #     self.str_priors.append(StructuralPrior(self.n_modules_at_layer(i-1), self.n_modules_at_layer(i)))
                dropout_before=self.module_options.dropout
                for m_i in range(self.n_modules_at_layer(i)):
                    if self.args.depper_first_layer and i==0:
                        deeper=1
                    else:
                        deeper=0  
                    if self.args.module_type=='resnet_block':
                        if i==0:
                            module_type='conv'
                        else:
                            module_type='resnet_block'
                            # self.hidden_size*=2
                    else:
                        module_type=self.args.module_type   

                    if self.args.net_arch=='alexnet':
                        ksizes=[4,3,2,0,0]     
                        #can resemble the Alexnet architecture
                        # if using module_type conv
                        if i>0:          
                            hidden_size*=2
                        if ksizes[i]>0:
                            self.module_options.kernel_size=ksizes[i]
                        else:
                            module_type='linear'
                            hidden_size=2048
                            channels_in = out_h*out_h*channels_in
                            out_h=1
                    self.block_constructor=conv_block_base         
                    conv = self.block_constructor(channels_in, hidden_size, out_h, name=f'components.{i}.{m_i}', module_type=module_type, deeper=deeper, options=self.module_options)
                    self.module_options.dropout=dropout_before                
                    ##################################################
                    ###Initialize all modules in layers identically###   
                    if self.per_layer_module_initialization[i] is not None:
                        conv.load_state_dict(self.per_layer_module_initialization[i])
                    if self.args.module_init=='identical':
                        self.per_layer_module_initialization[i] = conv.state_dict()
                    ##################################################
                    components_l.append(conv)
                out_h = conv.out_h
                # if module_type!='linear':
                channels_in = hidden_size               

                self.components.append(components_l)
                # if self.bn:
                #     self.bnorms.append(nn.BatchNorm2d(self.hidden_size, momentum=1., affine=True,
                #         track_running_stats=False))
                # else:
                #     self.bnorms.append(nn.Identity())
                    
            if self.args.module_type=='resnet_block':
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                out_h=1
            else:
                self.avgpool=nn.Identity()
            if self.args.module_type=='linear':
                out_h=1                             

        else:
            #strcture should be given
            #1. create a random model     
            temp_model = MNTDP_net(self.args, module_options=self.module_options, i_size=self.i_size, 
                                    channels=self.channels, hidden_size=self.hidden_size, num_classes=self.num_classes).to(device)
            #2. Load selected modules into the model
            for l, module in enumerate(structure):
                if len(self.components[l])-1>=module:   
                    temp_model.components[l][-1].load_state_dict(self.components[l][module].state_dict())
                    if self.components[l][module].module_learned_buffer:
                        temp_model.components[l][-1].freeze_module()
            # temp_model.structure = structure
            #return a model with selected modules
            return temp_model

        self.representation_dim = channels_in*out_h*out_h                                 
        self.decoder = nn.Linear(self.representation_dim, self.num_classes) if self.args.multihead=='none' else nn.ModuleList([nn.Linear(self.representation_dim, self.num_classes)])

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

    def add_output_head(self, num_classes=None, state_dict=None): 
        num_classes = self.num_classes if num_classes is None else num_classes
        if self.args.multihead=='usual':        
            l=nn.Linear(self.representation_dim, num_classes)
            if state_dict is None:
                l.load_state_dict(copy.deepcopy(self.decoder[-1].state_dict()))
            else:
                l.load_state_dict(state_dict)
            self.decoder.append(l.to(device))
        try:          
            if self.args.regime=='normal':  
                self.optimizer, self.optimizer_structure = self.get_optimizers()
        except ValueError:
            #will throw Value error if all modules are frozen
            pass

    def freeze_permanently_functional(self, free_layer=None, inner_loop_free=True):
        for l, layer in enumerate(self.components):
            if l!=free_layer:
                for c, component in enumerate(layer):
                    for p in self.module.parameters():
                        #setting requires_grad to False would prevent updating in the inner loop as well
                        p.requires_grad = False
        if self.args.regime=='normal':
            self.optimizer, self.optimizer_structure = self.get_optimizers()
        self._reset_references_to_learnable_params()
    
    def add_modules(self, block_constructor=None, at_layer=None, strict=False, verbose=True, state_dict=None):
        #if at_layer == None add to all layers
        if block_constructor is None:            
            block_constructor = self.block_constructor      
        for l, layer in enumerate(self.components):  
            ################################################
            if l == at_layer or at_layer==None:
                if verbose:
                    print(f'adding modules at layer {l}')
                self._n_modules[l]+=1
                ################################################
                #new module's params
                channels_in = layer[0].in_channels
                channels = layer[0].out_channels
                i_size = layer[0].i_size
                deeper = layer[0].deeper
                module_type = layer[0].module_type   
                self.module_options.kernel_size=layer[0].args.kernel_size
                self.module_options.dropout=layer[0].args.dropout
                # out_h = layer[0].out_h       
                ################################################
                new_module = block_constructor(channels_in, channels, i_size,   
                                                name=f'components.{l}.{len(layer)}', 
                                                module_type=module_type, deeper=deeper, options=self.module_options)
                if state_dict is not None:
                    new_module.load_state_dict(state_dict)
                new_module.reset()
                new_module.to(device)
                # new_module.reset()
                layer.append(new_module)
        self._reset_references_to_learnable_params()        
        if self.args.regime=='normal':  
            self.optimizer, self.optimizer_structure = self.get_optimizers()
             
    def get_optimizers(self):                               
        #set seperate optimizer for structure and model
        structure_names = ['inv_block', 'structure']          
        model_params = [param for n, param in self.named_parameters() if not any(map(n.__contains__, structure_names)) and param.requires_grad]
        # for n, p in self.named_parameters():
        #     print(n, p.is_leaf)
        optimizer = torch.optim.Adam(model_params, lr=self.args.lr, weight_decay=self.args.wdecay)
        structure_param_groups = []  
        #create module specific parameter groups
        if self.args.module_type!='filterwise':
            for layer in self.components:
                for component in layer:   
                    if hasattr(component, 'inv_block'):               
                        structure_param_groups.append({'params': filter(lambda p: p.requires_grad, component.inv_block.parameters()), 'lr': component.block_lr_buffer, 'name': component.name})
        else:
            for layer in self.components:
                for component in layer:                
                    pass
                    #structure_param_groups.append({'params': filter(lambda p: p.requires_grad, component.inv_block.parameters()), 'lr': component.block_lr_buffer, 'name': component.name})
        if len(structure_param_groups)>0:
            optimizer_structure = torch.optim.Adam(structure_param_groups,weight_decay=self.args.wdecay)
        else:
            optimizer_structure=None
        return optimizer, optimizer_structure
    
    def forward(self, X, task_id=None, strucure: List=None, force_structure=False, *args, **kwargs):
        #start = time.time() 
        # assert task_id is not None  
        if task_id is not None:
            if force_structure and strucure is not None:
                strucure=strucure
            else:
                if len(self.structure_pool)==0:
                    strucure = self.structure if strucure is None else strucure
                else:
                    strucure = self.structure_pool[task_id]
            assert strucure is None or len(strucure)==self.depth

            if self.args.module_type=='linear':
                    X=X.view(X.size(0), -1)
            n = X.shape[0]
            c = X.shape[1]          
            for layer, comp_idx in zip(self.components,strucure):
                    X = layer[comp_idx](X)
            X_flattened = X.reshape(n, -1)   
            if self.args.multihead=='none':
                raise NotImplementedError
                logit=self.decoder(X_flattened) 
            elif self.args.multihead=='usual':
                if task_id is not None and task_id<=(len(self.decoder)-1):         
                    logit= self.decoder[task_id](X_flattened)
                else:
                    logit=self.decoder[-1](X_flattened)   
            return self.forward_template(logit=logit, info={'embeddings':X})
        else:
            if self.args.entropy_task_inf:
                smalles_entropy, logits_best, oh_selected = None,None,None
                X_in=X
                for task_id in range(len(self.structure_pool)):
                    X=X_in
                    strucure = self.structure_pool[task_id]  
                    assert strucure is None or len(strucure)==self.depth

                    if self.args.module_type=='linear':
                            X=X.view(X.size(0), -1)
                    n = X.shape[0]
                    c = X.shape[1]          
                    for layer, comp_idx in zip(self.components,strucure):
                            X = layer[comp_idx](X)
                    X_flattened = X.reshape(n, -1)   
                    if self.args.multihead=='none':
                        raise NotImplementedError
                        logit=self.decoder(X_flattened) 
                    elif self.args.multihead=='usual':
                        if task_id is not None and task_id<=(len(self.decoder)-1):         
                            logit= self.decoder[task_id](X_flattened)
                        else:
                            logit=self.decoder[-1](X_flattened)  
                    entropy = scipy.stats.entropy(torch.softmax(logit, dim=1).cpu().detach().numpy(), base=2, axis=1).mean()
                    if smalles_entropy is None or entropy<smalles_entropy:
                        smalles_entropy=entropy
                        logits_best=logit
                        oh_selected=task_id
                return self.forward_template(logit=logits_best, info={'embeddings':X, 'selected_decoder':oh_selected})
            else:
                X_in=X
                logit_pool=[]
                for task_id in range(len(self.structure_pool)):
                    X=X_in
                    strucure = self.structure_pool[task_id]   
                    assert strucure is None or len(strucure)==self.depth

                    if self.args.module_type=='linear':
                            X=X.view(X.size(0), -1)
                    n = X.shape[0]
                    c = X.shape[1]          
                    for layer, comp_idx in zip(self.components,strucure):
                            X = layer[comp_idx](X)
                    X_flattened = X.reshape(n, -1)   
                    if self.args.multihead=='none':
                        raise NotImplementedError
                        logit=self.decoder(X_flattened) 
                    elif self.args.multihead=='usual':
                        if task_id is not None and task_id<=(len(self.decoder)-1):         
                            logit= self.decoder[task_id](X_flattened)
                        else:
                            logit=self.decoder[-1](X_flattened)  
                    logit_pool.append(logit)
                logits_best=torch.stack(logit_pool).max(0)[0]
                return self.forward_template(logit=logits_best, info={'embeddings':X})#, 'selected_decoder':oh_selected})


    def load_state_dict(self, state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], strict: bool=True):
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
                    self.add_modules(at_layer=l, verbose=False)
        components = filter( lambda x: '_backup' in x, state_dict.keys())
        for m in list(components):
            layer, comp = [int(i) for i in m if i.isdigit()][:2]
            if 'functional' in m:
                self.components[layer][comp]._maybe_backup_functional(verbose=False)
            if 'inv' in m:
                self.components[layer][comp]._maybe_backup_structural(verbose=False)  
        if self.args.multihead=='modulewise':      
            if any(map(lambda x: 'decoder' in x and not 'components' in x, state_dict.keys())): #if state dict contains decoder
                classifier_dict = {k: v for k, v in state_dict.items() if 'decoder' in k and not 'components' in k}
                for l in self.components:
                    for c in l:
                        if hasattr(c, 'num_classes'): 
                            print(f'Loading classifier into {c.name}')
                            if c.num_classes>0:
                                c.load_state_dict(classifier_dict, strict=False)
        return super().load_state_dict(state_dict,strict)

    def create_search_space(self, best_structure:List):
        if self.args.searchspace=='topdown':
            #1. most likeliy structure sofar                               
            best_structure = [0 for _ in range(self.depth)] if best_structure is None else best_structure
            best_model = self.init_modules(best_structure)
            yield best_model, best_structure #[0 for _ in range(self.depth)]
            new_structure=copy.copy(best_structure) #[0 for _ in range(self.depth)]
            for i in range(self.depth):
                l = self.depth-1-i
                if self.components[l][-1].module_learned:
                    new_structure[l]=len(self.components[l])  
                    model = self.init_modules(structure=new_structure)
                    yield model, new_structure
        elif self.args.searchspace=='bottomup':
            #1. most likeliy structure sofar                                   
            best_structure = [0 for _ in range(self.depth)] if best_structure is None else best_structure
            best_model = self.init_modules(best_structure)
            yield best_model, best_structure #[0 for _ in range(self.depth)]
            new_structure=copy.copy(best_structure) #[0 for _ in range(self.depth)]
            for i in range(self.depth):
                l = i
                if self.components[l][-1].module_learned:
                    new_structure[l]=len(self.components[l])  
                    model = self.init_modules(structure=new_structure)
                    yield model, new_structure
        else:
            raise NotImplemented