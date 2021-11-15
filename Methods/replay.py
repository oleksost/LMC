import torch
import numpy as np
import torch.nn as nn
from torch import Tensor     
from collections import OrderedDict, Iterable
from typing import Tuple, Dict, Type, Optional

device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# addopted from https://github.com/lebrice/Sequoia/blob/master/sequoia/methods/experience_replay.py
class Buffer(nn.Module):
    def __init__(self,
                 capacity: int,
                 input_shape: Tuple[int, ...],
                 extra_buffers: Dict[str, Type[torch.Tensor]] = None,
                 rng: np.random.RandomState = None,
                 ):
        super().__init__()
        self.rng = rng or np.random.RandomState()

        bx = torch.zeros([capacity, *input_shape], dtype=torch.float)
        by = torch.zeros([capacity], dtype=torch.long)

        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.buffers = ['bx', 'by']

        extra_buffers = extra_buffers or {}
        for name, dtype in extra_buffers.items():
            tmp = dtype(capacity).fill_(0)
            self.register_buffer(f'b{name}', tmp)
            self.buffers += [f'b{name}']

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0
        # (@lebrice) args isn't defined here:
        # self.to_one_hot  = lambda x : x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def unique_tasks_in_buffer(self):
        if self.balance_key is not None:
            return torch.unique(getattr(self, f'b{self.balance_key}'))
        else:
            #ignore task information
            return torch.tensor([0])

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        raise NotImplementedError("Can't make y one-hot, dont have n_classes.")
        return self.to_one_hot(self.by[:self.current_index])

    def add_reservoir(self, batch: Dict[str, Tensor]) -> None:
        n_elem = batch['x'].size(0)

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)

        if place_left:
            offset = min(place_left, n_elem)

            for name, data in batch.items():
                buffer = getattr(self, f'b{name}')
                if isinstance(data, Iterable):
                    buffer[self.current_index: self.current_index + offset].data.copy_(data[:offset])
                else:
                    buffer[self.current_index: self.current_index + offset].fill_(data)

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == batch['x'].size(0):
                return

        x = batch['x']
        self.place_left = False
        

        indices = torch.FloatTensor(x.size(0)-place_left).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices: Tensor = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero(as_tuple=False).squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        # perform overwrite op      
        for name, data in batch.items(): 
            buffer = getattr(self, f'b{name}')
            if isinstance(data, Iterable):
                data = data[place_left:]
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data

    def sample(self, n_samples: int, only_task: int = None, exclude_task: int = None) -> Dict[str, Tensor]:
        buffers = OrderedDict()
        if exclude_task is not None:
            assert hasattr(self, 'bt')
            valid_indices = torch.nonzero(self.bt != exclude_task, as_tuple=True)[0].squeeze()
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]
        elif only_task is not None:
            assert hasattr(self, 'bt')
            valid_indices = torch.nonzero(self.bt == only_task, as_tuple=True)[0].squeeze()
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[valid_indices]
        else:
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[:self.current_index]

        bx = buffers['bx']
        if bx.size(0) < n_samples:
            return OrderedDict({k[1:]: v for (k,v) in buffers.items()})
        else:
            indices_np = self.rng.choice(bx.size(0), n_samples, replace=False)
            indices = torch.from_numpy(indices_np).to(self.bx.device)
            return OrderedDict({k[1:]: v[indices] for (k,v) in buffers.items()})

class BalancedBuffer(nn.Module):
    def __init__(self,
                 capacity: int,
                 input_shape: Tuple[int, ...], 
                 extra_buffers: Dict[str, Type[torch.Tensor]] = None,
                 rng: np.random.RandomState = None,
                 balance_key: str = 't',
                 ):
        super().__init__()
        self.decreasing_prob_adding = True 
        self.rng = rng or np.random.RandomState()
        self.balance_key = balance_key

        bx = torch.zeros([capacity, *input_shape], dtype=torch.float)
        by = torch.zeros([capacity], dtype=torch.long)

        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.buffers = ['bx', 'by']
        extra_buffers = extra_buffers or {}
        for name, dtype in extra_buffers.items():
            tmp = dtype(capacity).fill_(0)
            self.register_buffer(f'b{name}', tmp)
            self.buffers += [f'b{name}']

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def unique_tasks_in_buffer(self):
        if self.balance_key is not None:
            return torch.unique(getattr(self, f'b{self.balance_key}'))
        else:
            #ignore task information
            return torch.tensor([0])
    
    @property
    def buffer_size(self):
        return self.bx.size(0)
    
    def calculate_per_task_distribution(self, left_to_place, new:int=1):
                    def f(x):
                        return len(np.array_split(np.arange(min(left_to_place, len(x))), len(self.unique_tasks_in_buffer))[-1])
                    return list(map(f, np.array_split(np.arange(self.buffer_size), len(self.unique_tasks_in_buffer)+new))) 

    def add_reservoir(self, batch: Dict[str, Tensor]) -> None:

        n_elem = batch['x'].size(0)   
            
        # add whatever still fits in the buffer
        place_left = max(0, self.buffer_size - self.current_index)
        offset = min(place_left, n_elem)
        if place_left:

            for name, data in batch.items():
                buffer = getattr(self, f'b{name}')
                if isinstance(data, Iterable):
                    buffer[self.current_index: self.current_index + offset].data.copy_(data[:offset])
                else:
                    buffer[self.current_index: self.current_index + offset].fill_(data)

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == batch['x'].size(0):
                return
           
        x = batch['x']   
        left_to_place = n_elem-offset
        if self.balance_key is not None:
            #try to balance the distributions of data samples according to the key (task_id)
            unique_tasks = self.unique_tasks_in_buffer
            if batch[self.balance_key] not in unique_tasks:
                #new key
                #space_per_key = list(map(len, np.array_split(np.arange(self.buffer_size), len(unique_tasks)+1))) #self.buffer_size // (len(unique_tasks)+1)
                space_needed_from_existing_tasks = self.calculate_per_task_distribution(left_to_place=left_to_place, new=1)
                # min(left_to_place, space_per_key) // len(unique_tasks)
                indicies = []
                for i,k in  enumerate(unique_tasks):
                    indicies+=list(self.rng.choice(torch.where(getattr(self,f'b{self.balance_key}')==k)[0].cpu(), size=space_needed_from_existing_tasks[i], replace=False))
                
                idx_buffer = torch.LongTensor(indicies).to(x.device)
                
                idx_new_data = torch.from_numpy(self.rng.choice(np.arange(left_to_place),idx_buffer.numel(), replace=False)).to(x.device)
                self.n_seen_so_far += idx_buffer.numel()

            else:
                #key is already in the buffer  
                space_per_key = len(np.array_split(np.arange(self.buffer_size), len(unique_tasks))[-1])
                idxs_in_buffer = torch.where(getattr(self,f'b{self.balance_key}')==batch[self.balance_key])[0]
                if len(idxs_in_buffer)<space_per_key-1:
                    #add some more
                    #space_per_key = self.buffer_size // len(unique_tasks)
                    space_needed_from_existing_tasks = self.calculate_per_task_distribution(left_to_place=left_to_place, new=0)
                    #(min(space_per_key-len(idxs_in_buffer), left_to_place)) // len(unique_tasks)
                    indicies = []
                    for i,k in enumerate(unique_tasks):
                        if k!=batch[self.balance_key]:
                            indicies+=list(self.rng.choice(torch.where(getattr(self,f'b{self.balance_key}')==k)[0].cpu(), size=space_needed_from_existing_tasks[i], replace=False))
                    idx_buffer = torch.LongTensor(indicies).to(x.device)
                    idx_new_data = torch.from_numpy(self.rng.choice(np.arange(left_to_place),idx_buffer.numel(), replace=False)).to(x.device)
                    self.n_seen_so_far += idx_buffer.numel() # min(space_per_key-len(idxs_in_buffer), left_to_place)

                else:
                    #overwrite existing              
                    indices = torch.FloatTensor(min(left_to_place,len(idxs_in_buffer))).uniform_(0, self.n_seen_so_far).long()
                    valid_indices: Tensor = torch.tensor(np.isin(indices, idxs_in_buffer.cpu())).long().to(x.device)

                    idx_new_data = valid_indices.nonzero(as_tuple=False).squeeze(-1)
                    idx_buffer   = indices[idx_new_data].to(x.device)

                    self.n_seen_so_far += left_to_place

                    if idx_buffer.numel() == 0:
                        return
        else:
            #ignore task information
            if self.decreasing_prob_adding:
                #prob of adding a new sample is max(buffer_size/n_seen_so_far,1)     
                indices = torch.FloatTensor(x.size(0)-place_left).to(x.device).uniform_(0, self.n_seen_so_far).long()
                valid_indices: Tensor = (indices < self.buffer_size).long()
                idx_new_data = valid_indices.nonzero(as_tuple=False).squeeze(-1)
                idx_buffer   = indices[idx_new_data]
            else:
                #prob of adding a new sample is 1
                idx_buffer = torch.LongTensor(self.rng.choice(self.buffer_size, x.size(0)-place_left)).to(x.device)#.uniform_(0, self.n_seen_so_far).long()
                idx_new_data = torch.LongTensor(np.arange(x.size(0)-place_left)).to(x.device)

            self.n_seen_so_far += left_to_place

            if idx_buffer.numel() == 0:
                return

        # perform overwrite op      
        for name, data in batch.items():  
            buffer = getattr(self, f'b{name}')
            if isinstance(data, Iterable):
                data = data[place_left:]
                buffer[idx_buffer] = data[idx_new_data]
            else:
                buffer[idx_buffer] = data
        # print('done')
    
    # @staticmethod  
    # def prepare_samples(buffer):
    #         return OrderedDict({'train':[buffer['bx_train'], buffer['by_train']], 'test':[buffer['bx_test'], buffer['by_test']], 't':buffer['bt']})

           
    def sample(self, n_samples: int, only_task: int = None,  exclude_task: int = None) -> Dict[str, Tensor]:
        buffers = OrderedDict()
        if self.balance_key is None:
            #simpli sample uniformly
            exclude_task = None #cant exclude

            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[:self.current_index]
            
            bx = buffers['bx']
            if bx.size(0) < n_samples:
                return OrderedDict({k[1:]: v for (k,v) in buffers.items()})
            else:
                indices_np = self.rng.choice(bx.size(0), n_samples, replace=False)
                indices = torch.from_numpy(indices_np).to(self.bx.device)
            return self.prepare_samples(OrderedDict({k: v[indices] for (k,v) in buffers.items()}))
        
        else:   
            #sinclude samples sampled uniformaly from each task
            # if exclude_task is not None:
            #     valid_indices = (self.bt != exclude_task).nonzero(as_tuple=False).squeeze()
            # else:
            #     valid_indices = (self.bt>=0).nonzero(as_tuple=False).squeeze()

            
            
            n_samples_per_task = map(len, np.array_split(np.arange(min(n_samples, self.buffer_size)), max(1,len(self.unique_tasks_in_buffer) - (int(exclude_task!=None)+int(only_task!=None) ))))
            indicies = []
            for t, n in enumerate(n_samples_per_task):
                if t!=exclude_task:
                    if only_task is not None:
                        if t==only_task:
                            ixs_task = (self.bt.cpu()==t).nonzero(as_tuple=False).squeeze()
                            indicies += list(self.rng.choice(ixs_task, min(n,len(ixs_task)), replace=False ))
                    
                    else:
                        ixs_task = (self.bt.cpu()==t).nonzero(as_tuple=False).squeeze()
                        indicies += list(self.rng.choice(ixs_task, min(n,len(ixs_task)), replace=False ))

            indicies = torch.LongTensor(indicies).to(device)
            
            for buffer_name in self.buffers:
                buffers[buffer_name] = getattr(self, buffer_name)[indicies]
                
            bx = buffers['bx']   
            if bx.size(0) < n_samples:
                return OrderedDict({k[1:]: v for (k,v) in buffers.items()})
            else:
                indices_np = self.rng.choice(bx.size(0), n_samples, replace=False)
                indices = torch.from_numpy(indices_np).to(self.bx.device)
            return OrderedDict({k[1:]: v[indices] for (k,v) in buffers.items()})