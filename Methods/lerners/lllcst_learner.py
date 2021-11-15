import torch
import torch.nn as nn
import os
import copy
from itertools import zip_longest
from sklearn.metrics import roc_auc_score
import numpy as np

class Learner():  
    def __init__(self, net, lr=0.001, wdecay=0, results_dir='./tmp/results/'):
        self.net = net
        if hasattr(self.net, 'regression') and self.net.regression:
            self.loss = nn.MSELoss()
            self.regression = True
        else:
            self.loss = nn.BCEWithLogitsLoss() if net.binary else nn.CrossEntropyLoss()
            self.regression = False
        self.lr=lr
        self.wdecay=wdecay
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=wdecay)
        self.T = 0
        self.observed_tasks = set()
        self.results_dir = results_dir
        self.init_trainloaders = None

    def train(self, *args, **kwargs):
        raise NotImplementedError('Training loop is algorithm specific')

    def init_train(self, trainloader, task_id, num_epochs, save_freq=1, testloaders=None):
        if self.init_trainloaders is None:
            self.init_trainloaders = {}
        self.init_trainloaders[task_id] = trainloader
        eval_bool = testloaders is not None
        if len(self.init_trainloaders) == self.net.num_init_tasks:
            iter_cnt = 0
            for i in range(num_epochs): 
                for XY_all in zip_longest(*self.init_trainloaders.values()):
                    for task, XY in zip(self.init_trainloaders.keys(), XY_all):
                        if XY is not None:
                            X, Y = XY
                            X = X.to(self.net.device, non_blocking=True)
                            Y = Y.to(self.net.device, non_blocking=True)
                            self.gradient_step(X, Y, task)
                            iter_cnt += 1
                if i % save_freq == 0 or i == num_epochs - 1:
                    if eval_bool:
                        self.evaluate(testloaders)
                    self.save_data(i + 1, task_id, save_eval=eval_bool)
            for task, loader in self.init_trainloaders.items():
                self.update_multitask_cost(loader, task)

    def evaluate(self, testloaders):
        was_training = self.net.training 
        prev_reduction = self.loss.reduction 
        self.loss.reduction = 'sum'     # make sure the loss is summed over instances
        self.net.eval()
        with torch.no_grad():
            self.test_loss = {}
            self.test_acc = {}
            self.test_auc = {}
            for task, loader in testloaders.items():
                l = 0.
                a = 0.
                auc = 0. 
                n = len(loader.dataset)
                compute_auc =  n <= loader.batch_size and self.net.binary
                for X, Y in loader:
                    X = X.to(self.net.device, non_blocking=True)
                    Y = Y.to(self.net.device, non_blocking=True)
                    Y_hat = self.net(X, task)
                    l += self.loss(Y_hat, Y).item()
                    if not self.regression:
                        a += ((Y_hat > 0) == (Y == 1) if self.net.binary else Y_hat.argmax(dim=1) == Y).sum().item()
                        if compute_auc:
                            auc += roc_auc_score(Y.squeeze().cpu(), Y_hat.squeeze().cpu())  # only works if batch learning (not mini-batch)
                if self.regression:
                    self.test_loss[task] = np.sqrt(l / n)
                else:
                    self.test_loss[task] = l / n
                    self.test_acc[task] = a / n
                    if compute_auc:
                        self.test_auc[task] = auc

        self.loss.reduction = prev_reduction
        if was_training:
            self.net.train()

    def gradient_step(self, X, Y, task_id, model=None, optimizer=None):
        if model is None and optimizer is None:
            Y_hat = self.net(X, task_id=task_id)
            l = self.loss(Y_hat, Y)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
        else:
            assert model is not None
            assert optimizer is not None
            Y_hat = model(X, task_id=task_id)
            l = self.loss(Y_hat, Y)
            model.zero_grad()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

    def save_data(self, epoch, task_id, save_eval=False):
        task_results_dir = os.path.join(self.results_dir, 'task_{}'.format(task_id))
        os.makedirs(task_results_dir, exist_ok=True)
        if epoch == 0 or epoch % 10 == 0:
            path = os.path.join(task_results_dir, 'checkpoint.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'observed_tasks': self.observed_tasks,
                }, path)
        if save_eval:
            log_open_mode = 'a' if epoch > 0 else 'w'   # make sure to overwrite previous file  if it exists
            with open(os.path.join(task_results_dir, 'log.txt'), log_open_mode) as f:
                line = 'epochs: {}, training task: {}\n'.format(epoch, task_id)
                f.write(line)
                print(line, end='')
                for task in self.test_loss:
                    if self.regression:
                        line = '\ttask: {}\tloss: {}\n'.format(task, self.test_loss[task])
                    elif len(self.test_auc) > 0:
                        line = '\ttask: {}\tloss: {}\tacc: {}\tauc: {}\n'.format(task, self.test_loss[task], self.test_acc[task], self.test_auc[task])
                    else:
                        line = '\ttask: {}\tloss: {}\tacc: {}\n'.format(task, self.test_loss[task], self.test_acc[task])                        
                    f.write(line)
                    print(line, end='')
    
    def update_multitask_cost(self, loader, task_id):
        raise NotImplementedError('Update update_multitask is algorithm specific')

class CompositionalLearner(Learner):
    def train(self, trainloader, task_id, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None):
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        eval_bool = testloaders is not None
        if eval_bool:
            self.evaluate(testloaders)
        self.save_data(0, task_id, save_eval=eval_bool)
        if self.T <= self.net.num_init_tasks:
            self.net.freeze_structure()
            self.init_train(trainloader, task_id, num_epochs, save_freq, testloaders)
        else:
            self.net.freeze_modules()
            self.net.freeze_structure()     # freeze structure for all tasks
            self.net.freeze_structure(freeze=False, task_id=task_id)    # except current one
            iter_cnt = 0
            for i in range(num_epochs):
                if (i + 1) % component_update_freq == 0:
                    self.update_modules(trainloader, task_id)   # replace one structure epoch with one module epoch
                else:
                    for X, Y in trainloader:
                        X = X.to(self.net.device, non_blocking=True)
                        Y = Y.to(self.net.device, non_blocking=True)
                        self.update_structure(X, Y, task_id)
                        iter_cnt += 1
                if i % save_freq == 0 or i == num_epochs - 1:
                    if eval_bool:
                        self.evaluate(testloaders)
                    self.save_data(i + 1, task_id, save_eval=eval_bool)
            self.update_multitask_cost(trainloader, task_id)


    def update_structure(self, X, Y, task_id, **kwargs):
        self.gradient_step(X, Y, task_id, **kwargs)    # assume shared parameters are frozen and just take a gradient step on the structure

    def update_modules(self, *args, **kwargs):
        raise NotImplementedError('Update modules is algorithm specific')

class JointLearner(Learner):
    pass

class NoComponentsLearner(Learner):
    pass

class CompositionalDynamicLearner(CompositionalLearner): 
    def create_optimizer(self, net=None):
        if net is None:      
            return self.optimizer
        else:
            return torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.lr, weight_decay=self.wdecay)

    def train(self, trainloader, task_id, valloader, component_update_freq=100, num_epochs=100, save_freq=1, testloaders=None):
        if task_id not in self.observed_tasks:
            self.observed_tasks.add(task_id)
            self.T += 1
        eval_bool = testloaders is not None
        if eval_bool:
            self.evaluate(testloaders)       
        self.save_data(0, task_id, save_eval=eval_bool)         
        if self.T <= self.net.num_init_tasks and not self.net.args.seperate_pool_per_layer:
            if not self.net.args.seperate_pool_per_layer:
                self.net.freeze_structure()
            self.init_train(trainloader, task_id, num_epochs, save_freq, testloaders)
            if self.net.args.seperate_pool_per_layer:
                for l in self.net.components:
                    for m in l:
                        m.freeze_module()
        else:                 
            if not self.net.args.seperate_pool_per_layer:
                self.net.freeze_modules()
                self.net.freeze_structure()     # freeze structure for all tasks
                self.net.add_tmp_module(task_id)   # freeze original modules and structure
                self.optimizer.add_param_group({'params': self.net.components[-1].parameters()})
                if hasattr(self, 'preconditioner'):
                    self.preconditioner.add_param_group(self.net.components[-1])

                self.net.freeze_structure(freeze=False, task_id=task_id)    # unfreeze (new) structure for current task
                iter_cnt = 0

                for i in range(num_epochs):
                    if (i + 1) % component_update_freq == 0:
                        self.update_modules(trainloader, task_id)
                    else:
                        for X, Y in trainloader:
                            X_cpu, Y_cpu = X, Y
                            X = X.to(self.net.device, non_blocking=True)
                            Y = Y.to(self.net.device, non_blocking=True)
                            self.update_structure(X, Y, task_id)
                            self.net.hide_tmp_module()
                            self.update_structure(X, Y, task_id)
                            self.net.recover_hidden_module()
                            iter_cnt += 1
                    if i % save_freq == 0 or i == num_epochs - 1:
                        if eval_bool:
                            self.evaluate(testloaders)           
                        self.save_data(i + 1, task_id, save_eval=eval_bool)
                self.conditionally_add_module(valloader, task_id)
                self.save_data(num_epochs + 1, task_id, save_eval=False, final_save=True)
                self.update_multitask_cost(trainloader, task_id)
            else:   
                searchspace = self.net.create_search_space(task_id)
                #2. Search for the bast model on the given task
                best_valid_acc, best_model, best_structure, best_idx = None, None, None, None
                for _m, (model, structure) in enumerate(searchspace):
                    model.train()
                    model.freeze_structure()  
                    for param in model.decoder[task_id].parameters():
                        param.requires_grad = True
                    if task_id>0:  
                        model.freeze_structure(freeze=False, task_id=task_id)    # unfreeze (new) structure for current task
                    optimizer = self.create_optimizer(net=model)
                    print('Trying structure ', structure)
                    # if hasattr(self, 'preconditioner'):
                    #     self.preconditioner.add_param_group(self.net.components[-1])
                    iter_cnt = 0
                    best_val=0.        
                    best_model_dict_early_stop=None
                    print(model.device)
                    for i in range(num_epochs):
                        model.train()
                        for X, Y in trainloader:
                            X_cpu, Y_cpu = X, Y                  
                            X = X.to(model.device, non_blocking=True)
                            Y = Y.to(model.device, non_blocking=True)
                            self.gradient_step(X, Y, task_id, model=model, optimizer=optimizer)
                            iter_cnt += 1
                            # early stopping:
                        if i%3==0:                    
                            valid_acc_e = self.evaluate_task(valloader, task_id, eval_no_update=False, model=model)[1]
                        if best_val < valid_acc_e:
                            best_val = valid_acc_e  
                            best_model_dict_early_stop = copy.deepcopy(model.state_dict())
                        if i % 10 == 0:
                            print("Vall acc ",valid_acc_e)
                        # if i % save_freq == 0 or i == num_epochs - 1:
                        #     if eval_bool:
                        #         self.evaluate(testloaders)
                        #     self.save_data(i + 1, task_id, save_eval=eval_bool)
                    if best_model_dict_early_stop is not None:   
                        model.load_state_dict(best_model_dict_early_stop, strict=True) 

                    valid_acc = self.evaluate_task(valloader, task_id, eval_no_update=False, model=model)[1]
                    if best_valid_acc is None or best_valid_acc<valid_acc:
                        best_valid_acc = copy.deepcopy(valid_acc)
                        best_model = copy.deepcopy(model)
                        best_structure = copy.deepcopy(structure)
                        best_idx=_m
                #3. Add new modules to the model_main if needed          
                print(f"Best structure selected task {task_id}", best_structure, 'best valid acc', best_valid_acc)
                for l, module_idx in enumerate(best_structure):
                    if module_idx>0:      
                        self.net.add_modules(at_layer=l, task_id=task_id, state_dict=best_model.components[l][-1].state_dict())
                    elif task_id==0:
                        self.net.components[l][-1].load_state_dict(best_model.components[l][-1].state_dict(), strict=True)
                    if not self.net.components[l][-1].module_learned:
                        self.net.components[l][-1].freeze_module()        
                self.net.structure[task_id].load_state_dict(best_model.structure[task_id].state_dict(), strict=True)    
                for l in range(len(self.net.structure_head[task_id])):
                    self.net.structure_head[task_id][l].load_state_dict(best_model.structure_head[task_id][l].state_dict(), strict=True)
                self.net.decoder[task_id].load_state_dict(best_model.decoder[task_id].state_dict())


    def conditionally_add_module(self, valloader, task_id):
        test_loss = self.test_loss
        test_acc = self.test_acc

        self.evaluate({task_id: valloader})
        update_loss, no_update_loss = self.test_loss[task_id]
        update_acc, no_update_acc = self.test_acc[task_id]
        print('W/update: {}, WO/update: {}'.format(update_acc, no_update_acc))
        if no_update_acc == 0 or (update_acc - no_update_acc) / no_update_acc > .05:
            print('Keeping new module. Total: {}'.format(self.net.num_components))
        else:
            self.net.remove_tmp_module()
            print('Not keeping new module. Total: {}'.format(self.net.num_components))

        self.test_loss = test_loss
        self.test_acc = test_acc

    def evaluate(self, testloaders, eval_no_update=True):
        was_training = self.net.training 
        prev_reduction = self.loss.reduction 
        self.loss.reduction = 'sum'     # make sure the loss is summed over instances
        self.net.eval()
        with torch.no_grad():
            self.test_loss = {}
            self.test_acc = {}
            self.test_auc = {}
            for task, loader in testloaders.items():
                test_loss, test_acc = self.evaluate_task(loader, task, eval_no_update)
                self.test_loss[task]=test_loss
                self.test_acc[task]=test_acc
        self.loss.reduction = prev_reduction
        if was_training:
            self.net.train()
    
    def evaluate_task(self,loader, task, eval_no_update=True, model=None):
        l = 0.
        a = 0.
        n = len(loader.dataset)
        model = self.net if model is None else model
        model.eval()
        for X, Y in loader:
            X = X.to(model.device, non_blocking=True)
            Y = Y.to(model.device, non_blocking=True)
            Y_hat = model(X, task)
            l += self.loss(Y_hat, Y).item()
            if not self.regression:
                a += ((Y_hat > 0) == (Y == 1) if model.binary else Y_hat.argmax(dim=1) == Y).sum().item()
        if eval_no_update and task == self.T - 1 and self.T > model.num_init_tasks:
            model.hide_tmp_module()
            l1 = 0.
            a1 = 0.
            for X, Y in loader:
                X = X.to(model.device, non_blocking=True)
                Y = Y.to(model.device, non_blocking=True)
                Y_hat = model(X, task)
                l1 += self.loss(Y_hat, Y).item()
                if not self.regression:
                    a1 += ((Y_hat > 0) == (Y == 1) if model.binary else Y_hat.argmax(dim=1) == Y).sum().item()
            test_loss = (l / n, l1 / n)
            test_acc = (a / n, a1 / n)
            model.recover_hidden_module()
        else: 
            test_loss = l / n
            test_acc = a / n
        return test_loss, test_acc

    def save_data(self, epoch, task_id, save_eval=False, final_save=False):
        pass
        # super().save_data(epoch, task_id, save_eval)
        # if final_save:
        #     task_results_dir = os.path.join(self.results_dir, 'task_{}'.format(task_id))
        #     with open(os.path.join(task_results_dir, 'num_components.txt'), 'w') as f:
        #         line = 'final components: {}'.format(self.net.num_components)
        #         f.write(line)

class CompositionalDynamicFM(CompositionalDynamicLearner):
    def __init__(self, net, lr=0.001, wdecay=0, results_dir='./tmp/results/'):
        super().__init__(net, lr, wdecay, results_dir)

    def update_modules(self, trainloader, task_id):
        # Just update structure instead
        for X, Y in trainloader:
            X = X.to(self.net.device, non_blocking=True)
            Y = Y.to(self.net.device, non_blocking=True)
            self.update_structure(X, Y, task_id)
            self.net.hide_tmp_module()
            self.update_structure(X, Y, task_id)
            self.net.recover_hidden_module()

    def update_multitask_cost(self, trainloader, task_id):
        pass