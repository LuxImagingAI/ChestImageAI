import sys, os, json, glob
import numpy as np
import torchvision, torch

from bit_pytorch.models import ResNetV2, KNOWN_MODELS as BiTModels


def instantiate_model(architecture, num_classes, pretrained='imagenet', fresh_head_weights=False):
    
    pretrained_dict = {}
    if pretrained and (pretrained != 'imagenet'):
        pretrained_dict = torch.load(pretrained, map_location="cpu")['model']
        # we remove "module" if there since it was created by data parallel training
        pretrained_dict = {k.split('module.')[-1]: v for k, v in pretrained_dict.items()} 

    if architecture == 'densenet121':
        model = torchvision.models.densenet121(pretrained= (pretrained == 'imagenet'))
        model.classifier = torch.nn.Linear(1024, num_classes)
        head_name = 'classifier'
    
    if architecture.startswith('BiT'):
        model = BiTModels[architecture](head_size=num_classes, zero_head=True)
        head_name = 'head'
        if pretrained == 'imagenet':
            weights = np.load(f'{model_folder}/{architecture}.npz')
            model.load_from(weights)
            
    if pretrained_dict:
        if fresh_head_weights:
            pretrained_dict = {k:v for k,v in pretrained_dict.items() if (head_name not in k)}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        print(f'Loaded pretraining weights from {pretrained}')
    
    return model



def get_lr(step, supports=[3000, 7000, 9000, 10000], base_lr=0.003):
    """Returns learning-rate for `step` or None at the end."""
    # Linear warmup
    if step < supports[0]:
        return base_lr * step / supports[0]
    # End of training
    elif step >= supports[-1]:
        return None
    factor = np.sum(np.array(supports)<=step)-1
    factor = 10**factor
    return base_lr/factor


class maskedBCE():

    def __init__(self, train_cols=None, device=None):
    
        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cols = torch.tensor([train_cols]).to(device) if train_cols else None
    
    
    def __call__(self, x, y):
                
        mask = y>=0
        if self.train_cols is not None:
            mask *= self.train_cols
        if mask.sum()>0:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x[mask], y[mask], reduction='sum')
        else:
            loss = torch.tensor(0).to(y.device)

        return loss, mask.sum()