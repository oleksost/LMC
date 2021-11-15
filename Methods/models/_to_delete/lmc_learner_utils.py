def print_masks(model, loader, temp=0.0001):
    model.eval()
    mask = []       
    model.args.automated_module_addition=0
    def create_mask(mask, label): 
        max_dim = max(list(map(lambda x: x.size(0), mask)))
        mask = list(map(lambda x: x[:,:].mean(1), mask))
        return list(map(lambda x: torch.cat((x.cpu(),torch.zeros((max_dim-x.size(0))))) if x.size(0)<max_dim else x.cpu(), mask))   
    for x,y in loader:
        forward_out = model(x.to(device),  temp = temp)
        mask.append(torch.stack(create_mask(forward_out.mask, 0)))
    mask=torch.stack(mask).mean(0)     
    fig, (ax1) = pyplot.subplots(1,1,figsize=(5,5))
    im = ax1.imshow(mask.cpu().T, cmap='Blues')
    fig.show()
    model.args.automated_module_addition=1

def prepare_masks(model, loaders, temp=0.0001):
    masks=[]        
    state_dict=copy.deepcopy(model.state_dict())
    for ti, loader in enumerate(loaders):
        model.load_state_dict(state_dict, strict=True)    
        model.train() 
        # no warm up for the last loader, if no batch norm is used, if gating=='experts'
        bn_warmup_steps=200*(1-int(ti==(len(loaders)-1)))*int(args.use_bn)*int(args.gating!='experts')
        if bn_warmup_steps>0:
            for i, (x,y) in enumerate(loader):
                model(x.to(device), record_stats=False, inner_loop=False)
                if i>=bn_warmup_steps:
                    break
        mask_ = []     
        model.eval()
        for x,y in loader:
            forward_out = model(x.to(device),  temp = temp)
            mask_.append(torch.stack(create_mask(forward_out.mask, 0)))
        mask=torch.stack(mask_).mean(0)     
        masks.append(mask)
    model.load_state_dict(state_dict, strict=True)   
    return masks