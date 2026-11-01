"""Diagnostic reconstruction normalization, all other current settings held fixed."""
import torch
import torch.nn.functional as F
from paper_protocol import CONFIG


def normalized_reconstruction_loss(h,reconstructed,sampled,features,adjacency,targets):
    cosine=F.normalize(h,p=2,dim=1) @ F.normalize(h,p=2,dim=1).T
    n,m=cosine.shape
    transition=sum((cosine-target).square().sum() for target in targets)/(len(targets)*n*m)
    feature=F.mse_loss(reconstructed,features)
    # Keep the current probability-BCE convention to isolate reconstruction;
    # this term has no parameter gradient for the fixed augmenter.
    augmentation=F.binary_cross_entropy(sampled,adjacency,reduction='sum') if sampled is not None else h.new_zeros(())
    total=CONFIG.beta1*transition+CONFIG.beta2*feature+CONFIG.alpha*augmentation
    return total,dict(transition=transition,feature=feature,augmentation=augmentation)
