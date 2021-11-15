import sys,time
import numpy as np
import torch

from . import hat_utils as utils 
# addapted from https://github.com/joansj/hat
########################################################################################################################

class Hat_Network(torch.nn.Module): 
    def __init__(self,inputsize,taskcla, hidden_size, use_bn=True, per_task_bn=0):
        super(Hat_Network,self).__init__()
        self.per_task_bn=per_task_bn
        self.use_bn = use_bn
        ncha,size=inputsize
        self.taskcla=taskcla

        self.c1=torch.nn.Conv2d(3,hidden_size,kernel_size=3, stride=1, padding=2)
        if per_task_bn:
            self.bn1=torch.nn.ModuleList()
            for t,n in enumerate(self.taskcla): 
                self.bn1.append(torch.nn.BatchNorm2d(hidden_size))
        else:
            self.bn1 = torch.nn.BatchNorm2d(hidden_size)
        s=utils.compute_conv_output_size(size,3)
        s=s//2
        self.c2=torch.nn.Conv2d(hidden_size,hidden_size,kernel_size=3, stride=1, padding=2)
        if per_task_bn:
            self.bn2=torch.nn.ModuleList()
            for t,n in enumerate(self.taskcla): 
                self.bn2.append(torch.nn.BatchNorm2d(hidden_size))
        else:
            self.bn2 = torch.nn.BatchNorm2d(hidden_size)
        s=utils.compute_conv_output_size(s,3)
        s=s//2
        self.c3=torch.nn.Conv2d(hidden_size,hidden_size,kernel_size=3, stride=1, padding=2)
        if per_task_bn:
            self.bn3=torch.nn.ModuleList()
            for t,n in enumerate(self.taskcla): 
                self.bn3.append(torch.nn.BatchNorm2d(hidden_size))
        else:
            self.bn3 = torch.nn.BatchNorm2d(hidden_size)
        s=utils.compute_conv_output_size(s,3)
        s=s//2
        self.c4=torch.nn.Conv2d(hidden_size,hidden_size,kernel_size=3, stride=1, padding=2)
        if per_task_bn:
            self.bn4=torch.nn.ModuleList()
            for t,n in enumerate(self.taskcla): 
                self.bn4.append(torch.nn.BatchNorm2d(hidden_size))
        else:
            self.bn4 = torch.nn.BatchNorm2d(hidden_size)
        s=utils.compute_conv_output_size(s,3)
        s=s//2
        self.smid=s


        self.maxpool=torch.nn.MaxPool2d(2, stride=2)
        self.relu=torch.nn.ReLU()

        # self.drop1=torch.nn.Dropout(0.2)
        # self.drop2=torch.nn.Dropout(0.5)
        # self.fc1=torch.nn.Linear(256*self.smid*self.smid,2048)
        # self.fc2=torch.nn.Linear(2048,2048)

        if hidden_size==8:
            fc_size=72
        elif hidden_size==64:
            fc_size=576
        elif hidden_size==320:
            fc_size=2880
        elif hidden_size==384:
            fc_size=3456
        self.last=torch.nn.ModuleList()
        for t,n in enumerate(self.taskcla): 
            self.last.append(torch.nn.Linear(fc_size,n))

        self.gate=torch.nn.Sigmoid()
        # All embedding stuff should start with 'e'
        self.ec1=torch.nn.Embedding(len(self.taskcla),hidden_size)
        self.ec2=torch.nn.Embedding(len(self.taskcla),hidden_size)
        self.ec3=torch.nn.Embedding(len(self.taskcla),hidden_size)
        self.ec4=torch.nn.Embedding(len(self.taskcla),hidden_size)
        """ (e.g., used in the compression experiments)
        lo,hi=0,2
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        self.ec3.weight.data.uniform_(lo,hi)
        self.efc1.weight.data.uniform_(lo,hi)
        self.efc2.weight.data.uniform_(lo,hi)
        #"""

        return
    def forward(self,t,x,s=1):
        # Gates
        masks=self.mask(t,s=s)
        gc1,gc2,gc3,gc4=masks
        # Gated
        h=self.c1(x)
        if self.use_bn:
            if self.per_task_bn:
                h=self.maxpool(self.relu(self.bn1[t](h)))
            else:
                h=self.maxpool(self.relu(self.bn1(h)))
        else:
            h=self.maxpool(self.relu(h))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h=self.c2(h)
        if self.use_bn:
            if self.per_task_bn:
                h=self.maxpool(self.relu(self.bn2[t](h)))
            else:
                h=self.maxpool(self.relu(self.bn2(h)))
        else:
            h=self.maxpool(self.relu(h))
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h=self.c3(h)
        if self.use_bn:
            if self.per_task_bn:
                h=self.maxpool(self.relu(self.bn3[t](h)))
            else:
                h=self.maxpool(self.relu(self.bn3(h)))
        else:
            h=self.maxpool(self.relu(h))
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h=self.c4(h)
        if self.use_bn:
            if self.per_task_bn:
                h=self.maxpool(self.relu(self.bn4[t](h)))
            else:
                h=self.maxpool(self.relu(self.bn4(h)))
        else:
            h=self.maxpool(self.relu(h))
        h=h*gc4.view(1,-1,1,1).expand_as(h)
        h=h.view(x.size(0),-1)
        # y=[]
        # for i,_ in self.taskcla:
        y=self.last[t](h)
        return y,masks

    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gc4=self.gate(s*self.ec4(t))
        return [gc1,gc2,gc3,gc4]

    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gc4=masks
        # if n=='fc1.weight':
        #     post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
        #     pre=gc3.data.view(-1,1,1).expand((self.ec3.weight.size(1),self.smid,self.smid)).contiguous().view(1,-1).expand_as(self.fc1.weight)
        #     return torch.min(post,pre)
        # elif n=='fc1.bias':
        #     return gfc1.data.view(-1)
        # elif n=='fc2.weight':
        #     post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
        #     pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
        #     return torch.min(post,pre)
        # elif n=='fc2.bias':
        #     return gfc2.data.view(-1)
        if n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2.weight)
            return torch.min(post,pre)
        elif n=='c2.bias':
            return gc2.data.view(-1)
        elif n=='c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3.weight)
            return torch.min(post,pre)
        elif n=='c3.bias':
            return gc3.data.view(-1)
        elif n=='c4.weight':
            post=gc4.data.view(-1,1,1,1).expand_as(self.c3.weight)
            pre=gc3.data.view(1,-1,1,1).expand_as(self.c3.weight)
            return torch.min(post,pre)
        elif n=='c4.bias':
            return gc4.data.view(-1)
        return None

device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,lamb=0.75,smax=400, wdecay=0,args=None):
        self.model=model
        
        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.wdecay=wdecay
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        self.lamb=lamb          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        self.smax=smax          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
        
        # if len(args.parameter)>=1:
        #     params=args.parameter.split(',')
        #     print('Setting parameters to',params)
        #     self.lamb=float(params[0])
        #     self.smax=float(params[1])

        self.mask_pre=None
        self.mask_back=None

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr, weight_decay=self.wdecay)

    def train(self,t,xtrain,ytrain,xvalid,yvalid):
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                self.train_epoch(t,xtrain,ytrain)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain,ytrain)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer(lr)
                print()
        except KeyboardInterrupt:
            print()

        # Restore best validation model
        utils.set_model_(self.model,best_model)

        # Activations mask
        task=torch.autograd.Variable(torch.LongTensor([t]),volatile=False).to(device)
        mask=self.model.mask(task,s=self.smax)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        if t==0:
            self.mask_pre=mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

        # Weights mask
        self.mask_back={}
        for n,_ in self.model.named_parameters():
            vals=self.model.get_view_for(n,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals

        return

    def train_epoch(self,t,x,y,thres_cosh=50,thres_emb=6):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).to(device)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=torch.autograd.Variable(x[b],volatile=False)
            targets=torch.autograd.Variable(y[b],volatile=False)
            task=torch.autograd.Variable(torch.LongTensor([t]),volatile=False).to(device)
            s=(self.smax-1/self.smax)*i/len(r)+1/self.smax

            # Forward
            outputs,masks=self.model.forward(task,images,s=s)
            output=outputs
            loss,_=self.criterion(output,targets,masks)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den

            # Apply step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data=torch.clamp(p.data,-thres_emb,thres_emb)

            #print(masks[-1].data.view(1,-1))
            #if i>=5*self.sbatch: sys.exit()
            #if i==0: print(masks[-2].data.view(1,-1),masks[-2].data.max(),masks[-2].data.min())
        #print(masks[-2].data.view(1,-1))

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        total_reg=0

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).to(device)

        # Loop batches
        with torch.no_grad():
            for i in range(0,len(r),self.sbatch):
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images=torch.autograd.Variable(x[b])#,volatile=True)
                targets=torch.autograd.Variable(y[b])#,volatile=True)
                task=torch.autograd.Variable(torch.LongTensor([t])).to(device)#,volatile=True).to(device)

                # Forward
                outputs,masks=self.model.forward(task,images,s=self.smax)
                output=outputs#[t]
                loss,reg=self.criterion(output,targets,masks)
                _,pred=output.max(1)
                hits=(pred==targets).float()

                # Log
                total_loss+=loss.data.cpu().numpy().item()*len(b)
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=len(b)
                total_reg+=reg.data.cpu().numpy().item()*len(b)

        print('  {:.3f}  '.format(total_reg/total_num),end='')

        return total_loss/total_num,total_acc/total_num

    def criterion(self,outputs,targets,masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count
        return self.ce(outputs,targets)+self.lamb*reg,reg

########################################################################################################################

