import copy
import os

import ctrl
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from simple_parsing import ArgumentParser

import wandb
from Utils.logging_utils import log_wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import seaborn as sns

from main_transfer import (ArgsGenerator, create_dataloader_ctrl,
                           get_accs_for_tasks, get_oh_init_idx, init_model,
                           set_seed, test, train_on_task)


def main(args):
    if args.task_sequence =='s_pnp':
        args.task_sequence_train ='s_reuse_1' 
    elif args.task_sequence =='s_pnp_tr':
        args.task_sequence_train ='s_reuse_1_tr' 
    elif args.task_sequence =='s_pnp_comp':
        args.task_sequence_train ='s_pnp_comp_1'
    else:
        args.task_sequence_train='s_reuse_1' 
    task_gen_1 = ctrl.get_stream(args.task_sequence_train, seed=args.stream_seed)  
    
    t = task_gen_1.add_task()    
    if args.task_sequence_train =='s_pnp_comp_1':
        task_gen_1_test = ctrl.get_stream('s_pnp_comp_1_test', seed=args.stream_seed)  
        t_test = task_gen_1_test.add_task() 
    else:
        task_gen_1_test = task_gen_1
        t_test = t

    model=init_model(args, args.gating, n_classes=t.n_classes.item(),  i_size=t.x_dim[-1]) 
    er_buffer = None
    n_tasks=args.n_tasks
    test_loaders_1=[]
    valid_loaders_1=[] 
    test_accuracies_past_1 = []
    valid_accuracies_past_1 = [] 
    fim_prev=[]

    for i in range(n_tasks):
        epochs=args.epochs
        print('==='*10)
        print(f'Task train {i}, Classes: {t.concepts}')   
        try:    
            print(t.transformation.trans_descr)   
        except:
            pass
        print('==='*10)                                                                                                                                                               
        train_loader_current, valid_dataloader, test_loader_current = create_dataloader_ctrl(task_gen_1, t, args,0, batch_size=args.batch_size, regime='labeled', task_n=i), create_dataloader_ctrl (task_gen_1, t, args,1,args.batch_size, regime='labeled', shuffle_test=('ood' in args.task_sequence), task_n=i), create_dataloader_ctrl (task_gen_1_test, t_test, args,2,args.batch_size, regime='labeled', shuffle_test=('ood' in args.task_sequence), task_n=i)         
        model.args.projection_phase_length = args.projection_phase_length*len(train_loader_current)
        model=train_on_task(model, args, train_loader_current, valid_dataloader, test_loader_current, epochs=epochs, anneal=args.anneal, str_anneal=args.str_anneal, task_id=i, reg_factor=args.reg_factor, fims=fim_prev, er_buffer=er_buffer)
        test_loaders_1.append(test_loader_current)
        valid_loaders_1.append(valid_dataloader)
        test_acc = test(model, None, test_loader_current, model.min_temp, model.min_str_prior_temp, task_id=i if not args.task_agnostic_test else None)[0].cpu().item()
        test_accuracies_past_1.append(test_acc)
        ####################
        #Current accuracy     
        log_wandb({f'test/test_acc_{i}':test_acc})
        try:
            t = task_gen_1.add_task()
            if args.task_sequence_train=='s_pnp_comp_1':
                t_test = task_gen_1_test.add_task() 
            else:
                t_test = t
        except:
            print(i)
            break      
        if args.multihead!='none':
            model.fix_oh(i)   
            init_idx=get_oh_init_idx(model, create_dataloader_ctrl (task_gen_1, t, args,0,batch_size=args.batch_size, regime='labeled', task_n=i), args)
            print('init_idx', init_idx)        
            model.add_output_head(t.n_classes.item(), init_idx=init_idx)
        else:
            #single head mode: create new, larger head
            model.add_output_head(model.decoder.out_features+t.n_classes.item(), state_dict=model.decoder.state_dict())
        if args.gating not in ['experts', 'hat']:
            if args.use_structural:      
                if args.use_backup_system:
                    model.freeze_permanently_structure()
                else:
                    for l,layer in enumerate(model.components):   
                        for m in layer:                          
                            m.freeze_functional(inner_loop_free=False)
                            m.freeze_structural()     
                            m.module_learned=torch.tensor(1.)
                            # model.add_modules(at_layer=l)
            model.optimizer, model.optimizer_structure = model.get_optimizers()
    #########################
    accs_test, Fs, masks_test, task_selection_accs = get_accs_for_tasks(model, args, test_loaders_1, test_accuracies_past_1, task_agnostic_test=args.task_agnostic_test)
    for ti, (acc, Frg, task_selection_acc) in enumerate(zip(accs_test, Fs, task_selection_accs)):
        log_wandb({f'test_acc_{ti}':acc}, prefix='test_model1/')
        #Forgetting (test)
        log_wandb({f'F_test_{ti}':Frg}, prefix='test_model1/')           
        #Task selection accuracy (only relevant in not ask id is geven at test time) (test)
        log_wandb({f'Task_selection_acc{ti}':task_selection_acc}, prefix='test_model1/')    
    ####################
    #Average accuracy (test) at the end of the sequence 
    print(accs_test)
    print(task_selection_accs)
    log_wandb({"mean_test_acc":np.mean(accs_test)})#, prefix='test/')
    #Average forgetting (test)
    log_wandb({"mean_test_F":np.mean(Fs)})#, prefix='test/')
    ####################
    #Model
    n_modules = torch.tensor(model.n_modules).cpu().numpy()     
    log_wandb({'total_modules': np.sum(np.array(n_modules))}, prefix='model/')

    ############
    #Model 2
    ###########
    if args.task_sequence=='s_pnp':   
        args.task_sequence_train2='s_reuse_2' 
    elif args.task_sequence=='s_pnp_tr':
        args.task_sequence_train2='s_reuse_2_tr' 
    elif args.task_sequence=='s_pnp_comp':
        args.task_sequence_train2='s_pnp_comp_2' 
    else:
        args.task_sequence_train2='s_reuse_2'
    task_gen_2 = ctrl.get_stream(args.task_sequence_train2, seed=args.stream_seed)  

    t = task_gen_2.add_task()    
    if args.task_sequence_train2=='s_pnp_comp_2':
        task_gen_2_test = ctrl.get_stream('s_pnp_comp_2_test', seed=args.stream_seed)  
        t_test = task_gen_2_test.add_task() 
    else:
        task_gen_2_test = task_gen_2
        t_test = t

    model2=init_model(args, args.gating, n_classes=t.n_classes.item(), i_size=t.x_dim[-1]) 
    er_buffer = None
    n_tasks=args.n_tasks
    test_loaders_2=[]
    valid_loaders_2=[]
    test_accuracies_past_2 = []
    for i in range(n_tasks):
        epochs=args.epochs
        print('==='*10)
        print(f'Task train {i}, Classes: {t.concepts}')   
        try:    
            print(t.transformation.trans_descr)   
        except:
            pass
        print('==='*10)                                                                                                          
        train_loader_current, valid_dataloader, test_loader_current = create_dataloader_ctrl (task_gen_2, t, args,0, batch_size=args.batch_size, regime='labeled', task_n=i), create_dataloader_ctrl (task_gen_2, t, args,1,args.batch_size, regime='labeled', shuffle_test=('ood' in args.task_sequence), task_n=i), create_dataloader_ctrl (task_gen_2_test, t_test, args,2,args.batch_size, regime='labeled', shuffle_test=('ood' in args.task_sequence), task_n=i) 
        model2.args.projection_phase_length = args.projection_phase_length*len(train_loader_current)
        model2=train_on_task(model2, args, train_loader_current, valid_dataloader, test_loader_current, epochs=epochs, anneal=args.anneal, str_anneal=args.str_anneal, task_id=i, reg_factor=args.reg_factor, fims=fim_prev, er_buffer=er_buffer)
        test_loaders_2.append(test_loader_current)
        valid_loaders_2.append(valid_dataloader)
        test_acc = test(model2, None, test_loader_current, model2.min_temp, model2.min_str_prior_temp, task_id=i if not args.task_agnostic_test else None)[0].cpu().item()
        test_accuracies_past_2.append(test_acc)
        ####################
        #Current accuracy     
        log_wandb({f'test_model2/test_acc_{i}':test_acc})
        try:
            t = task_gen_2.add_task()
            if args.task_sequence_train2=='s_pnp_comp_2':
                t_test = task_gen_2_test.add_task() 
            else:
                t_test = t
        except:
            print(i)
            break       
        if args.multihead!='none':
            model2.fix_oh(i)   
            init_idx=get_oh_init_idx(model2, create_dataloader_ctrl (task_gen_2, t, args,0,batch_size=args.batch_size, regime='labeled', task_n=i), args)
            print('init_idx', init_idx)        
            model2.add_output_head(t.n_classes.item(), init_idx=init_idx)
        else:
            #single head mode: create new, larger head
            model2.add_output_head(model2.decoder.out_features+t.n_classes.item(), state_dict=model2.decoder.state_dict())
        if args.gating not in ['experts', 'hat']:
            if args.use_structural:      
                if args.use_backup_system:
                    model2.freeze_permanently_structure()
                else:
                    for l,layer in enumerate(model2.components):   
                        for m in layer:                          
                            m.freeze_functional(inner_loop_free=False)
                            m.freeze_structural()     
                            m.module_learned=torch.tensor(1.)
                            # model2.add_modules(at_layer=l)
            model2.optimizer, model2.optimizer_structure = model2.get_optimizers()
    #########################
    accs_test, Fs, masks_test, task_selection_accs = get_accs_for_tasks(model2, args, test_loaders_2, test_accuracies_past_2, task_agnostic_test=args.task_agnostic_test)
    for ti, (acc, Frg, task_selection_acc) in enumerate(zip(accs_test, Fs, task_selection_accs)):
        log_wandb({f'test_acc_{ti}':acc}, prefix='test_model2/')
        #Forgetting (test)
        log_wandb({f'F_test_{ti}':Frg}, prefix='test_model2/')           
        #Task selection accuracy (only relevant in not ask id is geven at test time) (test)
        log_wandb({f'Task_selection_acc{ti}':task_selection_acc}, prefix='test_model2/')    
    ####################
    #Average accuracy (test) at the end of the sequence 
    print(accs_test)
    print(task_selection_accs)
    log_wandb({"mean_test_acc2":np.mean(accs_test)})#, prefix='test/')
    #Average forgetting (test)
    log_wandb({"mean_test_F2":np.mean(Fs)})#, prefix='test/')
    ####################
    #Model
    n_modules2 = torch.tensor(model2.n_modules).cpu().numpy()     
    log_wandb({'total_modules2': np.sum(np.array(n_modules2))}, prefix='model/')

    ############
    #Model 3
    ###########
    #consolidate   
    model3=copy.deepcopy(model)
    for l, layer in enumerate(model3.components):
        for c in range(len(model2.components[l])):
            model3.add_modules(at_layer=l)
            model3.components[l][-1].load_state_dict(model2.components[l][c].state_dict())
    for oh in model2.decoder:
        try:
            model3.add_output_head(oh.num_classes)
        except:
            model3.add_output_head(oh.weight.size(0))
        model3.decoder[-1].load_state_dict(oh.state_dict())
    #Model
    n_modules3 = torch.tensor(model3.n_modules).cpu().numpy()     
    log_wandb({'total_modules3': np.sum(np.array(n_modules3))}, prefix='model/')



    accs_test, Fs, masks_test, task_selection_accs = get_accs_for_tasks(model3, args, test_loaders_1+test_loaders_2, test_accuracies_past_1+test_accuracies_past_2, task_agnostic_test=args.task_agnostic_test)
    for ti, (acc, Frg ,task_selection_acc) in enumerate(zip(accs_test, Fs, task_selection_accs)):
            log_wandb({f'test_acc_{ti}':acc}, prefix='test_model3/')
            #Forgetting (test)
            log_wandb({f'F_test_{ti}':Frg}, prefix='test_model3/')           
            #Task selection accuracy (only relevant in not ask id is geven at test time) (test)
            log_wandb({f'Task_selection_acc{ti}':task_selection_acc}, prefix='test_model3/')    
    ####################
    #Average accuracy (test) at the end of the sequence 
    print(accs_test)
    print(task_selection_accs)
    log_wandb({"mean_test_acc3":np.mean(accs_test)})#, prefix='test/')
    #Average forgetting (test)
    log_wandb({"mean_test_F3":np.mean(Fs)})#, prefix='test/')
    ####################
    #Masks / Module usage.     
    if len(masks_test)>0 and args.gating=='locspec':         
        pyplot.clf()
        # fig, axs = pyplot.subplots(1,len(test_loaders),figsize=(15,15))     
        fig, axs = pyplot.subplots(1,len(test_loaders_1+test_loaders_2),figsize=(15,4))
        for i, ax in enumerate(axs): 
            im=sns.heatmap(F.normalize(masks_test[i].cpu().T, p=1, dim=0), vmin=0, vmax=1, cmap='Blues', cbar=False, ax=ax, xticklabels=[0,1,2,3])
            ax.set_title(f'Task {i}')
            for _, spine in im.spines.items():
                spine.set_visible(True)

        for i in range(len(masks_test)):
            print(masks_test[i].cpu().T)
        for i in range(len(masks_test)):
            print(F.normalize(masks_test[i].cpu().T, p=1, dim=0))
        pyplot.setp(axs[:], xlabel=f'layer')
        pyplot.setp(axs[0], ylabel='module')
    log_wandb({f"module usage": wandb.Image(fig)})
    if args.save_figures:
            pyplot.savefig(f'module_selection_plug_and_play.pdf', format='pdf', dpi=300)

    accs_valid, Fs_valid, _, task_selection_accs = get_accs_for_tasks(model3, args, valid_loaders_1+valid_loaders_2, None, task_agnostic_test=args.task_agnostic_test)        
    for ti, (acc, Frg, task_selection_acc) in enumerate(zip(accs_valid, Fs_valid, task_selection_accs)):
        log_wandb({f'valid_acc_{ti}':acc}, prefix='valid3/')
        #Forgetting (valid)
        log_wandb({f'F_valid_{ti}':Frg}, prefix='valid3/')
        #Task selection accuracy (only relevant in not ask id is geven at test time)(valid)
        log_wandb({f'Task_selection_acc{ti}':task_selection_acc}, prefix='valid3/')        
    ####################
    #Average accuracy (valid) at the end of the sequence 
    log_wandb({"mean_valid_acc3":np.mean(accs_valid)})#, prefix='valid/')
    #Average forgetting (valid)
    log_wandb({"mean_valid_F3":np.mean(Fs_valid)})#, prefix='test/')


if __name__== "__main__": 
    parser = ArgumentParser()     
    parser.add_arguments(ArgsGenerator, dest="Global")
    args = parser.parse_args()
    args_generator = args.Global
    dfs=[]
    pr_name=f'lmc' if args_generator.pr_name is None else args_generator.pr_name

    if args_generator.debug:
        pr_name='test'
    # if not args_generator.debug:
    run = wandb.init(project=pr_name, notes=args_generator.wand_notes, settings=wandb.Settings(start_method="fork"), reinit=(args_generator.n_runs>1))
    if args_generator.regenerate_seed:
        args_generator.generate_seed()
    if not args_generator.debug:      
        wandb.config.update(args_generator, allow_val_change=False)  
    set_seed(manualSeed=args_generator.seed)
    main(args_generator)
    