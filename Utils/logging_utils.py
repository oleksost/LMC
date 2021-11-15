import os
import torch
import logging 
from pathlib import Path
from typing import Any, Dict, Iterable, List, TypeVar, Union, Callable, Tuple
import tqdm
import wandb
from collections import OrderedDict

device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)
# logging.getLogger('simple_parsing').addHandler(logging.NullHandler())
root_logger = logging.getLogger()
T = TypeVar("T")
      
#log masks from validation  
def log_masks(metalearner, results, results_test, state, epoch, ds_train='omniglot', ds_eval='omniglot'):
        # try:
        #     if 'test/masks' in results:
        #         #examples of masks 
        #         mask = results['test/masks']   
        #         mask = torch.cat((mask, torch.zeros(mask.size(0), mask.size(1), 1)), dim=2)     
        #         x_labels = [f'mask_{m}_layer_{l}' for m in range(mask.size(0)) for l in range(mask.size(2))]
        #         y_labels = [f'm_{m}_e{epoch}_t_{ds_train}_e_{ds_eval}' for m in range(mask.shape[1])]
        #         log_wandb({f'{ds_train}/gating_mask_epoch_{epoch}_t_{ds_train}_e_{ds_eval}': wandb.plots.HeatMap(x_labels, y_labels, mask.permute(1,0,2).reshape(metalearner.model.n_modules, -1), show_text=True)})
        # except:
        #         print('Unable to log masks') 
        try:
            #Module usage averaged over the tasks in validation set
            if 'mean_module_usage' in results['test/mask_stats']:     
                m_usage = results['test/mask_stats']['mean_module_usage']   
                x_labels = [f'layer_{l}' for l in range(m_usage.size(1))]
                y_labels = [f'm_{m}_e{epoch}_t_{ds_train}_e_{ds_eval}' for m in range(m_usage.shape[0])] 
                log_wandb({f'{ds_train}/mean_module_usage_mask_{epoch}_t_{ds_train}_e_{ds_eval}': wandb.plots.HeatMap(x_labels, y_labels, m_usage, show_text=True)}, step=('epoch', state.global_epoch))
        except:
                print('Unable to log masks') 
        try:
            if 'test/raw_masks' in results:
                ###################################
                #log specialization values per module per layer
                per_module_specialization = {f'l_{l}_m_{m}':v for m, ms in enumerate(results['test/raw_masks']) for l, v in enumerate(ms)}
                log_wandb(per_module_specialization, prefix=f'{ds_eval}/module_specialization/', step=('epoch', state.global_epoch))
        except:
                print('Unable to log masks') 
        
        try:
            if 'test/outlier_signals' in results:
                ###################################
                #log specialization values per module per layer
                outlier_signals = {f'l_{l}_m_{m}':v for m, ms in enumerate(results['test/outlier_signals']) for l, v in enumerate(ms)}
                log_wandb(outlier_signals, prefix=f'{ds_eval}/outlier_signals/', step=('epoch', state.global_epoch))
        except:
            print('Unable to log masks') 
        try:
            if 'test/outlier_signals_after' in results:
                ###################################
                #log specialization values per module per layer
                outlier_signals = {f'l_{l}_m_{m}':v for m, ms in enumerate(results['test/outlier_signals_after']) for l, v in enumerate(ms)}
                log_wandb(outlier_signals, prefix=f'{ds_eval}/outlier_signals_after/', step=('epoch', state.global_epoch))
        except:
            print('Unable to log masks') 

        try:
            #log specialization difference in/out
            if 'test/raw_masks' in results and 'test/raw_masks' in results_test:
                diff = results_test['test/raw_masks']- results['test/raw_masks']      
                diff = {f'l_{l}_m_{m}':v for m, ms in enumerate(diff) for l, v in enumerate(ms)}
                log_wandb({'module_specialization_difference': diff}, prefix=f'{ds_eval}/', step=('epoch', state.global_epoch))
        except:
                print('Unable to log raw_masks module_specialization_difference') 
        try:
            #log specialization difference in/out
            if 'test/raw_masks_after' in results and 'test/raw_masks_after' in results_test:
                diff = results_test['test/raw_masks_after']- results['test/raw_masks_after']     
                diff = {f'l_{l}_m_{m}':v for m, ms in enumerate(diff) for l, v in enumerate(ms)}
                log_wandb({'module_specialization_difference_after': diff}, prefix=f'{ds_eval}/', step=('epoch', state.global_epoch))
        except:
                print('Unable to log masks raw_masks_after module_specialization_difference_after') 
        try:                                
            if 'test/deviation_masks' in results:
                ###################################
                #log deviation values per module per layer
                deviation_masks = {f'l_{l}_m_{m}':v for m, ms in enumerate(results['test/deviation_masks']) for l, v in enumerate(ms)}
                log_wandb(deviation_masks, prefix=f'{ds_eval}/deviation_masks/', step=('epoch', state.global_epoch))

        except:
                print('Unable to log masks') 
        try:                                
            if 'test/deviation_masks_after' in results:
                ###################################
                #log deviation values per module per layer
                deviation_masks = {f'l_{l}_m_{m}':v for m, ms in enumerate(results['test/deviation_masks_after']) for l, v in enumerate(ms)}
                log_wandb(deviation_masks, prefix=f'{ds_eval}/deviation_masks_after/', step=('epoch', state.global_epoch))

        except:
                print('Unable to log masks') 
        try:                                
            if 'test/modules_ema_alpha' in results:
            ###################################
                #log deviation values per module per layer             
                modules_ema_alpha = {f'l_{l}_m_{m}':v for m, ms in enumerate(results['test/modules_ema_alpha']) for l, v in enumerate(ms)}
                log_wandb(modules_ema_alpha, prefix=f'{ds_eval}/modules_ema_alpha/', step=('epoch', state.global_epoch))

        except:
                print('Unable to log masks') 

def backup_model(args, metalearner, state: dict, epoch: int, dataset: str):
    #if not args.debug:
    if args.Global.output_folder is not None:                    
        save_to = args.Global.output_folder + f'/{args.Global.name}'
        Path(save_to).mkdir(parents=True, exist_ok=True)      
        if args.Global.eai:
            save_to += f'/{args.Global.name}.pt'
            torch.save({'state_dict':metalearner.model.state_dict(),
                    'optimizers': metalearner.all_optimizers_state_dict,
                    'args':vars(args),
                    'experimen_state': state,
                    'replay_dict': metalearner.replay_buffer.state_dict() if metalearner.replay_buffer is not None else None,
                    'crp': metalearner.CRP.state_dict() if hasattr(metalearner, 'CRP') else None,
            }, save_to)            
        # else:
        save_to = args.Global.output_folder + f'/{args.Global.name}/{args.Global.name}_model_pretrain_e{epoch}_{dataset}.pt'
        torch.save({'state_dict':metalearner.model.state_dict(),
                    'optimizers': metalearner.all_optimizers_state_dict,
                    'args':vars(args),
                    'experimen_state': state,
                    'replay_dict': metalearner.replay_buffer.state_dict() if metalearner.replay_buffer is not None else None,
                    'crp': metalearner.CRP.state_dict() if hasattr(metalearner, 'CRP') else None,
            }, save_to)
        # #set ENV variables to handle preamtions
        # try:  
        #     import os
        #     runname=os.environ['SLURM_JOB_ID']
        #     file = open(os.environ['SLURM_TMPDIR']+f'/{runname}.sh', "w") 
        #     file.write(f"export LAST_BACKUP_PATH={save_to}") 
        #     file.close() 
        # except:
        #     pass

def check_for_backup(args):
    if args.output_folder is not None:          
        if os.path.exists(args.output_folder+f'/{args.name}/{args.name}.pt'):
            return load_model(args.output_folder+f'/{args.name}/{args.name}.pt')
    return None

def load_model(path_to_model):
    checkpoint = torch.load(path_to_model, map_location=torch.device(device))     
    model_state_dict = checkpoint['state_dict']
    optimizers_state_dict=None
    try:
        optimizers_state_dict=checkpoint['optimizers']
    except:
        print('tried loading optimizer state dict but failed')
    args=None
    if 'args' in checkpoint: 
        args = checkpoint['args']
    state = checkpoint['experimen_state']
    if 'replay_dict' in checkpoint:
        replay_dict = checkpoint['replay_dict']
    else:
        replay_dict = None
    if 'crp' in checkpoint:
        crp_state_dict = checkpoint['crp']
    else:
        crp_state_dict = None
    return args, model_state_dict, state, replay_dict, optimizers_state_dict, crp_state_dict



def log_wandb(message, step:Tuple[str,Any]=None, prefix=None, print_message=False, clean=True):
    # for k, v in message.items():
    #         if hasattr(v, 'to_log_dict'):
    #             message[k] = v.to_log_dict()
    if clean:
        try: 
            message = cleanup(message, sep="/")
        except:
            pass
    if prefix:
        message = add_prefix(message, prefix)
    if step is not None:
        message[step[0]] = step[1]
    try:
        wandb.log(message)#, step=step)
    except: # ValueError:
        pass #wandb is not innitialized
    if print_message:
        print(message)

def add_prefix(some_dict: Dict[str, T], prefix: str="") -> Dict[str, T]:
    """Adds the given prefix to all the keys in the dictionary that don't already start with it. 
    
    Parameters
    ----------
    - some_dict : Dict[str, T]
    
        Some dictionary.
    - prefix : str, optional, by default ""
    
        A string prefix to append.
    
    Returns
    -------
    Dict[str, T]
        A new dictionary where all keys start with the prefix.
    """
    if not prefix:
        return OrderedDict(some_dict.items())
    result: Dict[str, T] = OrderedDict()
    for key, value in some_dict.items():
        new_key = key if key.startswith(prefix) else (prefix + key)
        result[new_key] = value
    return result

def pbar(dataloader: Iterable[T], description: str="", *args, **kwargs) -> Iterable[T]:
    kwargs.setdefault("dynamic_ncols", True)
    pbar = tqdm.tqdm(dataloader, *args, **kwargs)
    if description:
        pbar.set_description(description)
    return pbar


def get_logger(name: str) -> logging.Logger:
        """ Gets a logger for the given file. Sets a nice default format. 
        TODO: figure out if we should add handlers, etc. 
        """
        try:
            p = Path(name)
            if p.exists():
                name = str(p.absolute().relative_to(Path.cwd()).as_posix())
        except:
            pass
        from sys import argv
            
        logger = root_logger.getChild(name)
        if "-d" in argv or "--debug" in argv:
            logger.setLevel(logging.DEBUG)
        # logger = logging.getLogger(name)
        # logger.addHandler(TqdmLoggingHandler())
        return logger


def get_new_file(file: Path) -> Path:
    """Creates a new file, adding _{i} suffixes until the file doesn't exist.
    
    Args:
        file (Path): A path.
    
    Returns:
        Path: a path that is new. Might have a new _{i} suffix.
    """
    if not file.exists():
        return file
    else:
        i = 0
        file_i = file.with_name(file.stem + f"_{i}" + file.suffix)
        while file_i.exists():
            i += 1
            file_i = file.with_name(file.stem + f"_{i}" + file.suffix)
        file = file_i
    return file

D = TypeVar("D", bound=Dict)
def flatten_dict(d: D, separator: str="/") -> D:
    """Flattens the given nested dict, adding `separator` between keys at different nesting levels.

    Args:
        d (Dict): A nested dictionary
        separator (str, optional): Separator to use. Defaults to "/".

    Returns:
        Dict: A flattened dictionary.
    """
    result = type(d)()
    for k, v in d.items():
        if isinstance(v, dict):
            for ki, vi in flatten_dict(v, separator=separator).items():
                key = f"{k}{separator}{ki}"
                result[key] = vi
        else:
            result[k] = v
    return result

             
def cleanup(message: Dict[str, Union[Dict, str, float, Any]], sep: str="/") -> Dict[str, Union[str, float, Any]]:
    """Cleanup a message dict before it is logged to wandb.

    Args:
        message (Dict[str, Union[Dict, str, float, Any]]): [description]
        sep (str, optional): [description]. Defaults to "/".

    Returns:
        Dict[str, Union[str, float, Any]]: [description]
    """
    # Flatten the log dictionary
    
    message = flatten_dict(message, separator=sep)

    # TODO: Remove redondant/useless keys
    for k in list(message.keys()):
        if k.endswith((f"{sep}n_samples", f"{sep}name")):
            message.pop(k)
            continue

        v = message.pop(k)
        # Example input:
        # "Task_losses/Task1/losses/Test/losses/rotate/losses/270/metrics/270/accuracy"
        # Simplify the key, by getting rid of all the '/losses/' and '/metrics/' etc.
        things_to_remove: List[str] = [f"{sep}losses{sep}", f"{sep}metrics{sep}"]
        for thing in things_to_remove:
            while thing in k:
                k = k.replace(thing, sep)
        # --> "Task_losses/Task1/Test/rotate/270/270/accuracy"
        if 'Task_losses' in k and 'accuracy' in k and not 'AUC' in k:
            k = k.replace('Task_losses', 'Task_accuracies')

        if 'Cumulative' in k and 'accuracy' in k and not 'AUC' in k:
            k = 'Task_accuracies/'+k
        
        if 'coefficient' in k:
            k = 'coefficients/'+k
        
        # Get rid of repetitive modifiers (ex: "/270/270" above)
        parts = k.split(sep)
        k = sep.join(unique_consecutive(parts))
        # Will become:
        # "Task_losses/Task1/Test/rotate/270/accuracy"
        
        if isinstance(v, Iterable):
            for i, el in enumerate(v):
                k_new = k + f'/{i}'
                message[k_new] = el
        else:
            message[k] = v

    return message


def unique_consecutive(iterable: Iterable[T], key: Callable[[T], Any]=None) -> Iterable[T]:
    """List unique elements, preserving order. Remember only the element just seen.
    
    >>> list(unique_consecutive('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D', 'A', 'B']
    >>> list(unique_consecutive('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'A', 'D']
    
    Recipe taken from itertools docs: https://docs.python.org/3/library/itertools.html
    """

    import operator
    from itertools import groupby
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)  
