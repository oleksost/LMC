import copy
import math
import time
import random
# import bocd
from collections import OrderedDict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from runstats import Statistics 
from sortedcontainers import SortedList

device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
transforms_rainbow={
    'bckgrnd_[255, 0, 0]': 'B-Red \n D-Black',
    'bckgrnd_[0, 255, 0]': 'B-Green \n D-Black',
    'bckgrnd_[0, 0, 255]': 'B-Blue \n D-Black',
    'digit_[255, 0, 0]': 'B-Black \n D-Red',
    'digit_[0, 255, 0]': 'B-Black \n D-Green',
    'digit_[0, 0, 255]': 'B-Black \n D-Blue'
}
def construct_name_ctrl(descr:str,name='')-> str:
        min_pos=np.inf
        end_pos=np.inf
        value=''
        for k in transforms_rainbow.keys():
            i = descr.find(k)
            if i>0 and i<min_pos:
                min_pos=i
                value=transforms_rainbow.get(k)
                end_pos=i+len(k)
        if min_pos<np.inf:
            name+=value
            descr=descr[end_pos:]
            return construct_name_ctrl(descr, name)
        else:
            return name

def ordered_dict_mean(d, weights=None):
    res = OrderedDict()
    def sum_val_for_key(d, key, val, weights=None):
        val_ = [val.clone().to(device)] #val.clone()
        n = 1
        for k,v in d.items():
            kk = k[2:]
            keyk = key[2:]
            if k != key and kk==keyk:
                n+=1
                #val+=v
                val_.append(v.to(device))

        if weights is not None:                
            for i in range(len(val_)):
                val_[i]*=weights[i]
                
            val_ = torch.stack(val_).to(device)
            val = val_.sum(0)
        else:    
            val = torch.stack(val_).sum(0).to(device)
        return val, n

    for k, v in d.items():
        kk = k[2:]
        if kk not in res.keys():
            vv, n =sum_val_for_key(d,k,v, weights)
            if weights is None:
                res[kk] = vv/n
            else:
                res[kk] = vv
    return res
def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def one_hot(indices, width):
    indices = indices.squeeze().unsqueeze(1)
    oh = torch.zeros(indices.size()[0], width).to(indices.device)
    oh.scatter_(1, indices, 1)
    return oh

def match_dims(mask, dim=0):   
    try:
        #zero pad along the dim with max dim size among the elements of mask
        #makes all elements same size
        if isinstance(mask, list):    
            max_dim = max(list(map(lambda x: x.size(), mask)))  
            if max_dim[dim]>1:
                pad = [0,0]*len(max_dim)
                pad[-(2*(dim)+1)]=1 
                #print(pad)
                mask = list(map(lambda x: F.pad(input=x, pad=[p*(max_dim[dim]-x.size(dim)) for p in pad], mode='constant', value=0) if x.size(dim)<max_dim[dim] else x, mask)) 
    except:
            return mask
    return mask

def set_seed(args=None, manualSeed=None):
    assert manualSeed is not None
    #####seed#####
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    if args is not None:
        if args.Global.device != "cpu":
            torch.cuda.manual_seed(manualSeed)
            torch.cuda.manual_seed_all(manualSeed)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        if device != "cpu":
            torch.cuda.manual_seed(manualSeed)
            torch.cuda.manual_seed_all(manualSeed)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    ######################################################

def cosine_rampdown(current, rampdown_length): 
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    if not (0 <= current <= rampdown_length):
        return torch.tensor(0.)
    return .5 * (torch.cos(torch.tensor(np.pi) * current / rampdown_length) + 1)

def standardize(tensor, dim=1):
    mean = torch.mean(tensor, dim=dim)
    std = torch.std(tensor, dim=dim)
    return (tensor - mean)/std

class DequeStats():
    def __init__(self): 
        self.queue = deque(maxlen=100)
    def mean(self):
        return np.mean(self.queue)    
    def stddev(self):
        return np.std(self.queue)   
    @property 
    def _count(self):
        return len(self.queue)
    def push(self,v):
        self.queue.append(v)


# class BayessianCPDetetor(nn.Module):
#     def __init__(self):
#         self.bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(300), bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1))

#     def push(self, x):
#         self.bc.update(x)


#https://lingpipe-blog.com/2009/07/07/welford-s-algorithm-delete-online-mean-variance-deviation/

class RunninStatsManager(nn.Module):            
    def __init__(self, max_len=2000, keep_median=False):
        super().__init__()
        self._inner_stats = RunningStats2(max_len, keep_median)
        self._outer_stats = RunningStats2(max_len, keep_median)
    
    @property
    def n(self):
        return self._outer_stats.n
    @property
    def _count(self):
        return self._outer_stats.n
    def push(self, x):
        self._inner_stats.push(x)
    def mean(self):
        return self._outer_stats.mean()
    def variance(self):
        return self._outer_stats.variance()
    def stddev(self):
        return self._outer_stats.stddev()    
    def consolidate_stats_from_inner_to_outer(self): 
        self._outer_stats.load_state_dict(self._inner_stats.state_dict())

    def consolidate_stats_from_outer_to_inner(self): 
        self._inner_stats.load_state_dict(self._outer_stats.state_dict())


class RunninStatsManagerOMF(nn.Module):            
    def __init__(self, max_len=2000, keep_median=False):
        super().__init__()
        self._inner_stats = RunningStats(max_len, keep_median)
        self._outer_stats = RunningStats(max_len, keep_median)
    
    @property
    def n(self):
        return self._outer_stats.n
    @property
    def _count(self):
        return self._outer_stats.n
    def push(self, x):
        self._inner_stats.push(x)
    def mean(self):
        return self._outer_stats.mean()
    def variance(self):
        return self._outer_stats.variance()
    def stddev(self):
        return self._outer_stats.stddev()    
    def consolidate_stats_from_inner_to_outer(self): 
        self._outer_stats.load_state_dict(self._inner_stats.state_dict())

    def consolidate_stats_from_outer_to_inner(self): 
        self._inner_stats.load_state_dict(self._outer_stats.state_dict())

class RunningStats(nn.Module):
    def __init__(self, max_len=2000, keep_median=False):
        super().__init__()
        self.register_buffer('n', torch.tensor(0.))
        self.register_buffer('old_m', torch.tensor(0.))
        self.register_buffer('new_m', torch.tensor(0.))
        self.register_buffer('old_s', torch.tensor(0.))
        self.register_buffer('new_s', torch.tensor(0.))
        self.max_len = max_len
        #self.queue =  deque(maxlen=self.max_len)
        self.register_buffer('queue', torch.zeros(self.max_len))

        self.register_buffer('current_idx_in_buffer', torch.LongTensor([0]))
        self.keep_median = keep_median
        if keep_median:
            self.register_buffer('_mad', torch.tensor(-1.))
            self.sorted_list = SortedList([])
            # self.register_buffer('sorted_list_median_devition', torch.zeros(self.max_len))
            self.register_buffer('median', torch.tensor(0.))

    @property
    def _count(self):
        return self.n
    
    def clear(self):
        self.n.data = torch.tensor(0.)
    def _unHandle(self, x):
        if (self.n == 0).item():
            raise NotImplementedError
        elif (self.n == 1).item():
            self.n.data = torch.tensor(0.)
            self.new_m.data = torch.tensor(0.) 
            self.new_s.data = torch.tensor(0.)

        else:
            mMOld  = ((self.n * self.new_m) - x)/(self.n-1)
            self.new_s -= (x -  self.new_m) * (x - mMOld)
            self.new_m = mMOld
            self.n = self.n - 1.

        self.old_m = self.new_m
        self.old_s = self.new_s

        # if self.new_s<0:
        #     print('_unHandle', x)
        #     print(self.new_s)
        #     print(self.new_m)
        #     print(self.queue)
        #     print(x)

            

    def push(self, x):
        self.n = self.n + 1

        if (self.n == 1).item():     
            self.old_m = self.new_m = x 
            self.old_s.data = torch.tensor(0.)
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n             
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s
        device = self.queue.device
        if self.n.item()>self.max_len:
            self._unHandle(self.queue[0])
            if self.keep_median:
                try:
                    self.sorted_list.remove(self.queue[0].item())#.cpu())     
                except ValueError:
                    self.sorted_list=SortedList(map( lambda x: x.item(), self.queue[:self.current_idx_in_buffer.item()]))
                    self.sorted_list.remove(self.queue[0].item())
                # self.sorted_list_median_devition[:self.current_idx_in_buffer] = torch.abs(torch.tensor(self.sorted_list)-self._median)
                #self.queue = torch.cat((self.queue[1:], torch.Tensor([x]).to(self.queue.device)))

            # self.queue.append(x)                       
            #if self.current_idx_in_buffer.item()>=self.max_len:
            self.queue = torch.cat((self.queue[1:], torch.Tensor([x]).to(device)))
        else:
            self.queue[self.current_idx_in_buffer.item()]=x      
            self.current_idx_in_buffer=self.current_idx_in_buffer+1

        if self.new_s<0:
            #TODO: sometimes std becomes < 0 , investigate why, for now, whenever it happens we simply recompute mean and std from the queue
            # print(self.queue)
            self.recompute_stats_from_queue()
            
        if self.keep_median:
            self.sorted_list.add(x.item())
            if len(self.sorted_list)!=self.current_idx_in_buffer.item():
                self.sorted_list=SortedList(map( lambda x: x.item(), self.queue[:self.current_idx_in_buffer.item()]))
            new_median = self.compute_median(self.sorted_list)
            self.median = torch.tensor(new_median, device=device)
            #median has changed, mad should be recomputed
            self._mad = torch.tensor(-1.)
            # self.sorted_list_median_devition[:self.current_idx_in_buffer] = torch.sort(torch.abs(torch.tensor(self.sorted_list)-new_median))[0]
    
    def mad(self):
        if self._mad <0:   
            if len(self.sorted_list)!=self.current_idx_in_buffer.item():
                self.sorted_list=SortedList(map( lambda x: x.item(), self.queue[:self.current_idx_in_buffer.item()]))
            #self._mad=self.compute_median(torch.sort(torch.abs(torch.tensor(self.sorted_list, device=device)-self.median))[0]) * 1.4826
            self._mad=torch.median(torch.abs(torch.tensor(self.sorted_list, device=device)-self.median)) * 1.4826
        return self._mad

    def recompute_stats_from_queue(self):
        self.new_m = torch.mean(self.queue[:self.current_idx_in_buffer.item()])
        self.new_s = torch.var(self.queue[:self.current_idx_in_buffer.item()])*(self.n-1)

    def compute_median(self, sorted_list):
        if not len(sorted_list)==0:
            if self.n.item()%2==0:
                _median = sorted_list[int((self.n.item())/2)-1]
                return _median
            else:
                _median = sorted_list[int((self.n.item()+1)/2)-1]
                return _median
        else: 
            return torch.tensor(0.)

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def stddev(self):
        return math.sqrt(self.variance())

def create_mask(mask, label):   
            max_dim = max(list(map(lambda x: x.size(0), mask)))
            mask = list(map(lambda x: x[:,:].mean(1), mask))
            return list(map(lambda x: torch.cat((x.cpu(),torch.zeros((max_dim-x.size(0))))) if x.size(0)<max_dim else x.cpu(), mask)) 


class RunningStats2(nn.Module):
    '''
    http://www.taylortree.com/2010/06/running-simple-moving-average-sma.html
    '''
    def __init__(self, max_len=2000, keep_median=False):
        super().__init__()
        self.register_buffer('n', torch.tensor(0.))
        self.register_buffer('_mean', torch.tensor(0.))
        self.register_buffer('new_powersumavg', torch.tensor(0.))
        self.register_buffer('new_var', torch.tensor(0.))
        self.max_len = max_len
        #self.queue =  deque(maxlen=self.max_len)
        self.register_buffer('queue', torch.zeros(self.max_len))

    @property
    def _count(self):
        return self.n
    
    def clear(self):
        self.n.data = torch.tensor(0.)           

    
    def cumulative_sma(self):
        """
        Returns the cumulative or unweighted simple moving average.
        Avoids sum of series per call.

        Keyword arguments:
        bar     --  current index or location of the value in the series
        series  --  list or tuple of data to average
        prevma  --  previous average (n - 1) of the series.
        """

        if self.n <= 0:
            return self.queue[0]

        return self._mean + ((self.queue[int(self.n.item())] - self._mean) / (self.n + 1.0))

        # return prevma + ((series[bar] - prevma) / (bar + 1.0))

    def running_sma(self):
        """
        Returns the running simple moving average - avoids sum of series per call.

        Keyword arguments:
        bar     --  current index or location of the value in the series
        series  --  list or tuple of data to average
        period  --  number of values to include in average
        prevma  --  previous simple moving average (n - 1) of the series
        """

        if self.n <= 0:
            return self.queue[0]

        elif self.n <= self.max_len-1:
            return self.cumulative_sma()

        #queue is full
        return self._mean + ((self.queue[-1] - self.queue[0]) / float(self.max_len))
    
    def powersumavg(self):
        """
        Returns the power sum average based on the blog post from
        Subliminal Messages.  Use the power sum average to help derive the running
        variance.
        sources: http://subluminal.wordpress.com/2008/07/31/running-standard-deviations/
    
        Keyword arguments:
        bar     --  current index or location of the value in the series
        series  --  list or tuple of data to average
        period  -- number of values to include in average
        pval    --  previous powersumavg (n - 1) of the series.
        """

        if self.n < 0:
            self.n = 0
        
        newamt = self.queue[int(self.n.item())]
    
        if self.n <= self.max_len-1:
            result = self.new_powersumavg + (newamt * newamt - self.new_powersumavg) / (self.n + 1.0)
    
        else:
            oldamt = self.queue[0]
            result = self.new_powersumavg + (((newamt * newamt) - (oldamt * oldamt)) / self.max_len)
    
        return result
    
    def running_var(self):
        """
        Returns the running variance based on a given time period.
        sources: http://subluminal.wordpress.com/2008/07/31/running-standard-deviations/

        Keyword arguments:
        bar     --  current index or location of the value in the series
        series  --  list or tuple of data to average
        asma    --  current average of the given period
        apowsumavg -- current powersumavg of the given period
        """
             
        if self.n <= 0:
            return torch.tensor(0.0)

        windowsize = self.n + 1.0
        if windowsize >= self.max_len:
            windowsize = self.max_len

        return (self.new_powersumavg * windowsize - windowsize * self._mean * self._mean) / windowsize

    def push(self, x):       
        device = self.queue.device
        if self.n.item()<=self.max_len-1:
            self.queue[int(self.n.item())]=x 
            self._mean=self.running_sma()
            self.new_powersumavg = self.powersumavg()
            self.new_var = self.running_var()           
        
        else:            
            self.queue = torch.cat((self.queue[:], torch.FloatTensor([x]).to(device)))
            self._mean=self.running_sma()
            self.new_powersumavg = self.powersumavg()
            self.n-=1
            self.queue = self.queue[1:]
            self.new_var = self.running_var()
            
        self.n=min(self.n+1, self.max_len) 
        # print(self._mean)
        # print(self.n)

        if self.new_var<0: #just in case
            self.recompute_stats_from_queue()

    def mean(self):
        return self._mean if self.n else 0.0
    def variance(self):
        return self.new_var               
    def stddev(self):
        return math.sqrt(self.variance())

    def recompute_stats_from_queue(self):
        self._mean = torch.mean(self.queue[:int(self.n.item())])
        self.new_var = torch.var(self.queue[:int(self.n.item())])
