import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.cuda.amp.autocast_mode import autocast
from fvcore.nn import sigmoid_focal_loss
#

def get_depth_loss( depth_labels, depth_preds, depth_channels= 48):
    
    depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
        -1,  depth_channels)
    fg_mask = torch.max(depth_labels, dim=1).values > 0.0

    # p
    with autocast(enabled=False):
        depth_loss = (F.binary_cross_entropy(
            depth_preds[fg_mask],
            depth_labels[fg_mask],
            reduction='none',
        ).sum() / max(1.0, fg_mask.sum()))

    return  depth_loss





class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False, top_k_ratio=1.0,  ):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        

    def forward(self, prediction, target, n_present=3): 
        
        b, s, c, h, w = prediction.shape  
        prediction = prediction.view(b * s, c, h, w)   
        target = target.view(b * s, h, w)  #
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device), ) 

        loss = loss.view(b, s, h, w) 

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # 
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)



class DepthLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255):
        super(DepthLoss, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        b, s, n, d, h, w = prediction.shape

        prediction = prediction.view(b*s*n, d, h, w)
        target = target.view(b*s*n, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights
        )
        return torch.mean(loss)




class SigmoidFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=-1.0,
        gamma=2.0,
        reduction='mean'
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label ):
        # print("pred--------shape", pred.dtype)
        pred = torch.argmax(pred, dim=2, keepdim=True)
        b, s, c, h, w = pred.shape  
        pred = pred.view(b * s, c, h, w)  # 
        label = label.float()  
        label = label.view(b * s,1, h, w)
        # print
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)




try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    
    if probas.numel() == 0:
        # only void pixel
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
   
    if probas.dim() == 3:
        #
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=255):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        b, s, c, h, w = probas.shape  #

        probas = probas.view(b * s, c, h, w)  
        probas = F.softmax(probas, dim=1)
        labels = labels.view(b * s, h, w)  # 
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)



if __name__ == '__main__':
    data1= torch.randn(2,3,2,200,200)*20
    data2= torch.zeros(2,3,1,200,200)
    ls = Lovasz_softmax()
    loss= ls(data1,data2)
    print("loss-----", loss)
