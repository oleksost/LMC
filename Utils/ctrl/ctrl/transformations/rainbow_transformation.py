# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
import numpy as np
import torch
from ctrl.transformations.transformation_tree import TransformationTree
from ctrl.transformations.utils import BatchedTransformation
from torchvision import transforms
from torchvision.transforms import RandomAffine

ROTATIONS = {
    'rotation_0': 0,
    # 'rotation_90': 90,
    # 'rotation_180': 180,
    # 'rotation_270': 270
}

COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
COLORS_RG = [[255, 0, 0], [0, 255, 0]]
OLD_BACKGOUND = [0]

SCALES = {
    'scale_full': 1,
    # '3/4': 0.75,
    # 'half': 0.5,
    # '1/4': 0.25
}


def get_rotations():
    transformations = {} 
    for name, angle in ROTATIONS.items():
        trans = transforms.Compose([
            transforms.ToPILImage(),
            RandomAffine(degrees=(angle, angle)),
            transforms.ToTensor()
        ])
        transformations[name] = BatchedTransformation(trans)
    return transformations


def get_scales():
    transformations = {}
    for name, scale in SCALES.items():
        trans = transforms.Compose([
            transforms.ToPILImage(),
            RandomAffine(degrees=0, scale=(scale, scale)),
            transforms.ToTensor()
        ])
        transformations[name] = BatchedTransformation(trans)
    return transformations


def change_background_color(images, old_background, new_background, new_background2=None, p=1):
    """
    :param images: BCHW
    :return:
    """
    if new_background2 is None:
        assert old_background == [0]     
        if not torch.is_tensor(new_background):
            new_background = torch.tensor(new_background, dtype=images.dtype)
            if images.max() <= 1 and new_background.max() > 1:
                new_background /= 255

        if images.size(1) == 1 and len(new_background) == 3:
            images = images.expand(-1, 3, -1, -1)
        else:
            assert images.size(1) == len(new_background)
            # raise NotImplementedError(images.size(), new_background)

        images = images.clone()

        new_background = new_background.view(-1, 1, 1)

        bg_ratio = images.max() - images
        bg = bg_ratio * new_background
        imgs = images + bg
        # print(images[:, 0, :, :].std().item(),images[:, 1, :, :].std().item(),images[:, 2, :, :].std().item())
        # print(imgs[:, 0, :, :].std().item(), imgs[:, 1, :, :].std().item(), imgs[:, 2, :, :].std().item())
        return imgs
    else:
        raise NotImplementedError

def change_background_color_balck_digit(images, old_background, new_background, new_background2=None, p=1):
    """
    :param images: BCHW
    :return:
    """
    if new_background2 is None:
        assert old_background == [0]     
        if not torch.is_tensor(new_background):
            new_background = torch.tensor(new_background, dtype=images.dtype)
            if images.max() <= 1 and new_background.max() > 1:
                new_background /= 255

        if images.size(1) == 1 and len(new_background) == 3:
            images = images.expand(-1, 3, -1, -1)
        else:
            assert images.size(1) == len(new_background)
            # raise NotImplementedError(images.size(), new_background)

        images = images.clone()

        new_background = new_background.view(-1, 1, 1)
        n=images.size(0)
        ch=images.size(1)
        if (images.view(n,ch,-1).sum(2)==0).sum(1).sum()>n:
            #when input is already colored (digit or background)
            non_zero_ch_idx=torch.nonzero(images[0].view(ch,-1).sum(1)).squeeze() #torch.nonzero(images[0].view(n,ch,-1).sum(2))
            non_zero_chnls = images[:,non_zero_ch_idx]
            if len(non_zero_chnls.shape)==3:
                non_zero_chnls=non_zero_chnls.unsqueeze(1)
            else:
                non_zero_chnls=non_zero_chnls[:,0].unsqueeze(1)
            if torch.sum(non_zero_chnls.view(n,-1)==0)>torch.sum(non_zero_chnls.view(n,-1)==1):
                #digit was previously colored
                bg_ratio = images.max() - non_zero_chnls
                bg = bg_ratio * new_background
                return images + bg
            else:
                #background is previously colored
                bg = (non_zero_chnls.expand(-1, 3, -1, -1)*new_background)
                images*=images.max()-new_background
                return images+bg
        else:
            #when input is greyscale
            bg_ratio = images.max() - images
            bg = bg_ratio * new_background
            # imgs = images + bg
            # print(images[:, 0, :, :].std().item(),images[:, 1, :, :].std().item(),images[:, 2, :, :].std().item())
            # print(imgs[:, 0, :, :].std().item(), imgs[:, 1, :, :].std().item(), imgs[:, 2, :, :].std().item())
            return bg #imgs
    else:
        assert old_background == [0]     
        if not torch.is_tensor(new_background):
            new_background = torch.tensor(new_background, dtype=images.dtype)
            if images.max() <= 1 and new_background.max() > 1:
                new_background /= 255
        if not torch.is_tensor(new_background2):
            new_background2 = torch.tensor(new_background2, dtype=images.dtype)
            if images.max() <= 1 and new_background2.max() > 1:
                new_background2 /= 255

        if images.size(1) == 1 and len(new_background) == 3:
            images = images.expand(-1, 3, -1, -1)
        else:
            assert images.size(1) == len(new_background)
            # raise NotImplementedError(images.size(), new_background)

        images = images.clone()

        new_background = new_background.view(-1, 1, 1)
        new_background2 = new_background2.view(-1, 1, 1)
        n=images.size(0)
        ch=images.size(1)

        if (images.view(n,ch,-1).sum(2)==0).sum(1).sum()>n:
            raise NotImplementedError
            #when input is already colored (digit or background)
            non_zero_ch_idx=torch.nonzero(images[0].view(ch,-1).sum(1)).squeeze() #torch.nonzero(images[0].view(n,ch,-1).sum(2))
            non_zero_chnls = images[:,non_zero_ch_idx]
            if len(non_zero_chnls.shape)==3:
                non_zero_chnls=non_zero_chnls.unsqueeze(1)
            else:
                non_zero_chnls=non_zero_chnls[:,0].unsqueeze(1)
            if torch.sum(non_zero_chnls.view(n,-1)==0)>torch.sum(non_zero_chnls.view(n,-1)==1):
                #digit was previously colored
                bg_ratio = images.max() - non_zero_chnls
                bg = bg_ratio * new_background
                return images + bg
            else:
                #background is previously colored
                bg = (non_zero_chnls.expand(-1, 3, -1, -1)*new_background)
                images*=images.max()-new_background
                return images+bg
        else:
            #when input is greyscale
            bg_ratio = images.max() - images
            idxs = torch.randperm(len(bg_ratio))
            n_imgs=int(p*len(bg_ratio))
            bg_ratio[idxs[:n_imgs]] *= new_background2
            bg_ratio[idxs[n_imgs:]] *= new_background
            # imgs = images + bg
            # print(images[:, 0, :, :].std().item(),images[:, 1, :, :].std().item(),images[:, 2, :, :].std().item())
            # print(imgs[:, 0, :, :].std().item(), imgs[:, 1, :, :].std().item(), imgs[:, 2, :, :].std().item())
            return bg_ratio #imgs



def change_digit_color(images, old_color, new_color):
    """
    :param images: BCHW
    :return:
    """
    assert old_color == [0]
    if not torch.is_tensor(new_color):
        new_color = torch.tensor(new_color, dtype=images.dtype)
        if images.max() <= 1 and new_color.max() > 1:
            new_color /= 255

    if images.size(1) == 1:
        images = images.expand(-1, 3, -1, -1)
    else:
        assert images.size(1) == len(new_color)
        # raise NotImplementedError(images.size(), new_background)

    images = images.clone()

    new_color = new_color.view(-1, 1, 1)
    n = images.shape[0]
    ch = images.shape[1]        
    if (images.view(n,ch,-1).sum(2)==0).sum(1).sum()>n:
        #when input is already colored
        non_zero_ch_idx=torch.nonzero(images[1].view(ch,-1).sum(1)).squeeze() #torch.nonzero(images[0].view(n,ch,-1).sum(2))
        if not len(non_zero_ch_idx.shape)==0:
            non_zero_ch_idx=non_zero_ch_idx[-1]
        non_zero_chnls = images[:,non_zero_ch_idx].unsqueeze(1)
        if torch.sum(non_zero_chnls.view(n,-1)==0)>torch.sum(non_zero_chnls.view(n,-1)==1):
            #digit was previously colored
            non_zero_chnls = non_zero_chnls.expand(-1, 3, -1, -1)*new_color
            imgs = images + non_zero_chnls
            # return images + bg
        else:
            #background is previously colored    
            non_zero_chnls = images.max()-non_zero_chnls.expand(-1, 3, -1, -1)
            non_zero_chnls *=new_color
            imgs = images + non_zero_chnls
    else:
        imgs = images * new_color
    # print(images[:, 0, :, :].std().item(),images[:, 1, :, :].std().item(),images[:, 2, :, :].std().item())
    # print(imgs[:, 0, :, :].std().item(), imgs[:, 1, :, :].std().item(), imgs[:, 2, :, :].std().item())
    return imgs



def get_colors(whiten_digit=True, stochastic=False, p=0.5, color_idx=None):    
    transformations = {}
    if color_idx is not None:
        i=color_idx
        color=COLORS[i]
        if not stochastic:
            if whiten_digit:
                trans = partial(change_background_color, old_background=OLD_BACKGOUND,
                                new_background=color)
            else:
                trans = partial(change_background_color_balck_digit, old_background=OLD_BACKGOUND,
                                new_background=color)
            transformations[f'bckgrnd_{str(color)}'] = trans
        else:
            if i==0:
                new_background2=COLORS[1]
            elif i==1:
                new_background2=COLORS[0]
            else:
                new_background2=COLORS[i-1]
            trans = partial(change_background_color_balck_digit, old_background=OLD_BACKGOUND,
                                new_background=color, new_background2=new_background2, p=p)
            transformations[f'bckgrnd_{str(color)}'] = trans
    else:
        for i, color in enumerate(COLORS):
            if not stochastic:
                if whiten_digit:
                    trans = partial(change_background_color, old_background=OLD_BACKGOUND,
                                    new_background=color)
                else:
                    trans = partial(change_background_color_balck_digit, old_background=OLD_BACKGOUND,
                                    new_background=color)
                transformations[f'bckgrnd_{str(color)}'] = trans
            else:
                if i==0:
                    new_background2=COLORS[1]
                elif i==1:
                    new_background2=COLORS[0]
                else:
                    new_background2=COLORS[i-1]
                trans = partial(change_background_color_balck_digit, old_background=OLD_BACKGOUND,
                                    new_background=color, new_background2=new_background2, p=p)
                transformations[f'bckgrnd_{str(color)}'] = trans
    return transformations

def get_colors_digits(colors=COLORS):
    transformations = {}
    for color in colors:     
        trans = partial(change_digit_color, old_color=OLD_BACKGOUND,
                        new_color=color)
        transformations[f'digit_{str(color)}'] = trans
    return transformations


class RainbowTransformationTree(TransformationTree):
    def __init__(self, *args, **kwargs):
        self.n_rotations = None
        self.n_colors = None
        self.n_scaless = None
        super(RainbowTransformationTree, self).__init__(*args, **kwargs)

    def build_tree(self):

        self.tree.add_node(self._node_index[self.name], name=self.name)

        rotations = get_rotations()
        colors = get_colors()
        scales = get_scales()
        levels = [rotations, scales, colors]

        prev_nodes = [self.name]

        for domain in levels:
            prev_nodes = self._add_transfos(prev_nodes, domain)

        self.leaf_nodes.update([self._node_index[node] for node in prev_nodes])

        self.depth = len(levels)

        return self._node_index[self.name]

    def _add_transfos(self, parent_nodes, transfos):
        nodes = []
        for parent in parent_nodes:
            for name, transfo in transfos.items():
                #remove transforms that color the entire image in one color
                # if ('digit_[' in name and 'bckgrnd' in parent) or ('digit_[' in parent and 'bckgrnd' in name):
                #     #prevent coloring digit and background with the same color
                #     cont=False
                #     for c in COLORS:
                #         if str(c) in name and str(c) in parent:
                #             cont=True
                #             break
                #     if cont:
                #         continue
                node_name = '{}_{}'.format(parent, name)    
                self.tree.add_node(self._node_index[node_name], name=node_name,
                                   last_transfo=name)

                self.tree.add_edge(self._node_index[parent],
                                   self._node_index[node_name],
                                   f=transfo, )
                nodes.append(node_name)
        return nodes

    def transformations_sim(self, t1, t2):
        """
        arccos((tr(R)−1)/2)
        :param t1:
        :param t2:
        :return:
        """
        t1_nodes = [t1.transfo_pool.tree.nodes()[id]['last_transfo'] for id in
                    t1.path[1:]]
        t2_nodes = [t2.transfo_pool.tree.nodes()[id]['last_transfo'] for id in
                    t2.path[1:]]
        n_eq = 0
        for op1, op2 in zip(t1_nodes, t2_nodes):
            if op1 == op2:
                n_eq += 1

        return n_eq / (len(t1_nodes))

class RainbowTransformationTreeBkgrndDigits(RainbowTransformationTree):
    def __init__(self, train=True, tree_depth=1, whiten_digits=True, *args, **kwargs):
        self.train=train
        self.tree_depth=tree_depth
        self.whiten_digits=whiten_digits
        super(RainbowTransformationTreeBkgrndDigits, self).__init__(*args, **kwargs)

    def build_tree(self):
        self.tree.add_node(self._node_index[self.name], name=self.name)

        rotations = get_rotations()        
        colors = get_colors(self.whiten_digits)
        scales = get_scales()
        if self.train:
            _colors=COLORS_RG
        else:
            _colors=COLORS

        digit_colors = get_colors_digits(colors=_colors)
        levels = [rotations, scales, colors, digit_colors]

        prev_nodes = [self.name]
        if self.train:
            if self.tree_depth!=1:   
                raise NotImplementedError
            for domain in levels[:2]:
                prev_nodes = self._add_transfos(prev_nodes, domain)

            prev_nodes_bgrnd = self._add_transfos(prev_nodes, levels[2])
            prev_nodes_digits = self._add_transfos(prev_nodes, levels[3])

            self.leaf_nodes.update([self._node_index[node] for node in prev_nodes_bgrnd+prev_nodes_digits])
        else:
            for domain in levels[:2]:    
                prev_nodes = self._add_transfos(prev_nodes, domain)

            prev_nodes_bgrnd = self._add_transfos(prev_nodes, levels[2])
            prev_nodes_digits = self._add_transfos(prev_nodes, levels[3])
            if self.tree_depth==2:
                prev_nodes_bgrnd = self._add_transfos(prev_nodes_bgrnd, dict(levels[2], **levels[3]))
                prev_nodes_digits = self._add_transfos(prev_nodes_digits, dict(levels[2], **levels[3]))
            if self.tree_depth>2:
                raise NotImplementedError

            self.leaf_nodes.update([self._node_index[node] for node in prev_nodes_bgrnd+prev_nodes_digits])
        self.depth = len(levels)
        return self._node_index[self.name]

class RainbowTransformationTreeBkgrndDigitsStochastic(RainbowTransformationTreeBkgrndDigits):
    def __init__(self, train=True, tree_depth=1, whiten_digits=True, trans_idx=None, *args, **kwargs):
        self.train=train  
        self.tree_depth=tree_depth
        self.whiten_digits=whiten_digits
        self.trans_idx=trans_idx
        super(RainbowTransformationTreeBkgrndDigitsStochastic, self).__init__(train=self.train, tree_depth=tree_depth, whiten_digits=whiten_digits, *args, **kwargs)
    def build_tree(self):
        self.tree.add_node(self._node_index[self.name], name=self.name)

        rotations = get_rotations()        
        
        if self.train:
            colors = get_colors(self.whiten_digits, stochastic=True, p=0.05, color_idx=self.trans_idx)
            scales = get_scales()
            if self.train:
                _colors=COLORS_RG
            else:
                _colors=COLORS

            digit_colors = get_colors_digits(colors=_colors)
            levels = [rotations, scales, colors, digit_colors]

            prev_nodes = [self.name]
            if self.tree_depth!=1:   
                raise NotImplementedError
            for domain in levels[:2]:
                prev_nodes = self._add_transfos(prev_nodes, domain)

            prev_nodes_bgrnd = self._add_transfos(prev_nodes, levels[2])
            prev_nodes_digits = self._add_transfos(prev_nodes, levels[3])

            self.leaf_nodes.update([self._node_index[node] for node in prev_nodes_bgrnd+prev_nodes_digits])
        else:
            colors = get_colors(self.whiten_digits, stochastic=True, p=0.5,color_idx=self.trans_idx)
            scales = get_scales()
            if self.train:
                _colors=COLORS_RG
            else:
                _colors=COLORS

            digit_colors = get_colors_digits(colors=_colors)
            levels = [rotations, scales, colors, digit_colors]

            prev_nodes = [self.name]
            for domain in levels[:2]:    
                prev_nodes = self._add_transfos(prev_nodes, domain)

            prev_nodes_bgrnd = self._add_transfos(prev_nodes, levels[2])
            prev_nodes_digits = self._add_transfos(prev_nodes, levels[3])
            if self.tree_depth==2:
                prev_nodes_bgrnd = self._add_transfos(prev_nodes_bgrnd, dict(levels[2], **levels[3]))
                prev_nodes_digits = self._add_transfos(prev_nodes_digits, dict(levels[2], **levels[3]))
            if self.tree_depth>2:
                raise NotImplementedError

            self.leaf_nodes.update([self._node_index[node] for node in prev_nodes_bgrnd+prev_nodes_digits])
        self.depth = len(levels)
        return self._node_index[self.name]

class RainbowTransformationDigits(RainbowTransformationTree): 
    def __init__(self, train=True, tree_depth=1, *args, **kwargs):
        self.train=train
        self.tree_depth=tree_depth
        super(RainbowTransformationDigits, self).__init__(*args, **kwargs)

    def build_tree(self):
        self.tree.add_node(self._node_index[self.name], name=self.name)
        
        rotations = get_rotations()
        scales = get_scales()
        digit_colors = get_colors_digits()
        levels = [rotations, scales, digit_colors]

        prev_nodes = [self.name]

        for domain in levels:
            prev_nodes = self._add_transfos(prev_nodes, domain)

        self.leaf_nodes.update([self._node_index[node] for node in prev_nodes])

        self.depth = len(levels)

        return self._node_index[self.name]

class RainbowTransformationBackground(RainbowTransformationTree):   
    def __init__(self, train=True, whiten_digits=True, *args, **kwargs):
        self.train=train
        self.whiten_digits=whiten_digits
        super(RainbowTransformationBackground, self).__init__(*args, **kwargs)

    def build_tree(self):
        self.tree.add_node(self._node_index[self.name], name=self.name)
        
        rotations = get_rotations()
        scales = get_scales()
        bkgrnd_colors = get_colors(self.whiten_digits)
        levels = [rotations, scales, bkgrnd_colors]

        prev_nodes = [self.name]

        for domain in levels:
            prev_nodes = self._add_transfos(prev_nodes, domain)

        self.leaf_nodes.update([self._node_index[node] for node in prev_nodes])

        self.depth = len(levels)

        return self._node_index[self.name]