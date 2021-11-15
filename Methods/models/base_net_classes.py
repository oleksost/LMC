import torch
import torch.nn as nn
import numpy as np

# addopted from https://github.com/Lifelong-ML/Mendez2020Compositional
device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CompositionalNet(nn.Module):
    def __init__(self,
                i_size,
                depth, 
                num_classes,
                num_tasks,
                num_init_tasks=None,
                init_ordering_mode='one_module_per_task',
                device='cuda'):
        super().__init__()
        self.device = device
        self.depth = depth
        self.num_tasks = num_tasks
        if num_init_tasks is None:
            num_init_tasks = 1 #depth
        self.num_init_tasks = num_init_tasks
        self.init_ordering_mode = init_ordering_mode
        self.i_size = i_size
        if isinstance(self.i_size, int):
            self.i_size = [self.i_size] * num_tasks
        self.num_classes = num_classes
        if isinstance(self.num_classes, int):
            self.num_classes = [self.num_classes] * num_tasks

    def init_ordering(self):
        raise NotImplementedError('Init ordering must be architecture specific')

    '''
    In both freeze functions we need to take care of making 
    grad=None for any parameter that does not require_grad.
    The zero_grad() function does not take care of this, and
    otherwise Adam will treat non-updates as updates (because
    grad is not None)
    '''
    def freeze_modules(self, freeze=True):
        for param in self.components.parameters():
            param.requires_grad = not freeze
            if freeze:
                param.grad = None


    def freeze_structure(self, freeze=True):
        raise NotImplementedError('Freeze structure must be architecture specific')

class SoftOrderingNet(CompositionalNet):
    def __init__(self,
                i_size,
                depth,
                num_classes,
                num_tasks,
                num_init_tasks=None,
                init_ordering_mode='one_module_per_task',
                device='cuda'):
        super().__init__(i_size,
                depth,
                num_classes,
                num_tasks,
                num_init_tasks=num_init_tasks,
                init_ordering_mode=init_ordering_mode,
                device=device)

        self.structure = nn.ParameterList([nn.Parameter(torch.ones(self.depth, self.depth)) for t in range(self.num_tasks)])
        self.init_ordering()

        self.softmax = nn.Softmax(dim=0)

    def init_ordering(self):
        if self.init_ordering_mode == 'one_module_per_task':
            assert self.num_init_tasks == self.depth, \
             'Initializing one module per task requires the number of initialization tasks to be the same as the depth'
            for t in range(self.num_init_tasks):     # first "k" tasks, use sinle layer repeated
                self.structure[t].data = -np.inf * torch.ones(self.depth, self.depth)
                self.structure[t].data[t, :] = 1
        elif self.init_ordering_mode == 'random_onehot':
            while True:
                initialized_modules = set()
                for t in range(self.num_init_tasks):
                    modules = np.random.randint(self.depth, size=self.depth)
                    self.structure[t].data = -np.inf * torch.ones(self.depth, self.depth)
                    self.structure[t].data[modules, np.arange(self.depth)] = 1
                    for m in modules:
                        initialized_modules.add(m)
                if len(initialized_modules) == self.depth:
                    break
        elif self.init_ordering_mode == 'random':
            raise NotImplementedError
        elif self.init_ordering_mode == 'uniform':
            pass
        else:
            raise ValueError('{} is not a valid ordering initialization mode'.format(self.init_ordering_mode))

    def freeze_structure(self, freeze=True, task_id=None):
        '''
        Since we are using Adam optimizer, it is important to
        set requires_grad = False for every parameter that is 
        not currently being optimized. Otherwise, even if they
        are untouched by computations, their gradient is all-
        zeros and not None, and Adam counts it as an update.
        '''
        if task_id is None:
            for param in self.structure:
                param.requires_grad = not freeze
                if freeze:
                    param.grad = None
            for param in self.decoder.parameters():
                param.requires_grad = not freeze
                if freeze:
                    param.grad = None
            if not self.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = not freeze
                    if freeze:
                        param.grad = None
        else:
            self.structure[task_id].requires_grad = not freeze
            if freeze:
                self.structure[task_id].grad = None
            for param in self.decoder[task_id].parameters():
                param.requires_grad = not freeze
                if freeze:
                    param.grad = None
            if not self.freeze_encoder:
                for param in self.encoder[task_id].parameters():
                    param.requires_grad = not freeze
                    if freeze:
                        param.grad = None

class SoftGatedNet(CompositionalNet):
    def __init__(self,
                i_size,
                depth,
                num_classes,
                num_tasks,
                num_init_tasks=None,
                init_ordering_mode='one_module_per_task',
                device='cuda'):
        super().__init__(i_size,
                depth,
                num_classes,
                num_tasks,
                num_init_tasks=num_init_tasks,
                init_ordering_mode=init_ordering_mode,
                device=device)

    def init_ordering(self):   
        if self.init_ordering_mode == 'one_module_per_task':
            raise NotImplementedError
        elif self.init_ordering_mode == 'random_onehot':
            raise NotImplementedError
        elif self.init_ordering_mode == 'random':
            pass
        else:
            raise ValueError('{} is not a valid ordering initialization mode'.format(self.init_ordering_mode))

    def freeze_structure(self, freeze=True, task_id=None):
        '''
        Since we are using Adam optimizer, it is important to
        set requires_grad = False for every parameter that is 
        not currently being optimized. Otherwise, even if they
        are untouched by computations, their gradient is all-
        zeros and not None, and Adam counts it as an update.
        '''
        
        if task_id is None:
            for param in self.structure.parameters():
                param.requires_grad = not freeze
                if freeze:
                    param.grad = None
            for param in self.decoder.parameters():
                param.requires_grad = not freeze
                if freeze:
                    param.grad = None
            if not self.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = not freeze
                    if freeze:
                        param.grad = None
        else:
            # if task_id>=len(self.structure):
            #     str_idx=-1
            # else:
            #     str_idx=task_id
            for param in self.structure[task_id].parameters():
                param.requires_grad = not freeze
                if freeze:
                    param.grad = None
            for param in self.decoder[task_id].parameters():
                param.requires_grad = not freeze
                if freeze:
                    param.grad = None
            if not self.freeze_encoder:
                for param in self.encoder[task_id].parameters():
                    param.requires_grad = not freeze
                    if freeze:
                        param.grad = None   