import sys, os, json, glob
import numpy as np
import torchvision, torch

from collections import OrderedDict
from torch import nn

from bit_pytorch.models import ResNetV2, KNOWN_MODELS as BiTModels


class ResNetV2_FeatExtractor(ResNetV2):
    
    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__(block_units, width_factor, head_size=21843, zero_head=False)
        self.pre_head = nn.Sequential(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048*width_factor)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool2d(output_size=1))
        ]))
        
    def forward(self, x):
        x = self.pre_head(self.body(self.root(x)))
        return x
    
    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))  # pylint: disable=line-too-long
            self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')
                    
    
class BiT_MetaMixin(torch.nn.Module):
    
    def __init__(self, num_meta_data, backbone):
        super().__init__()

        self.backbone = backbone
        self.meta_injection = num_meta_data
        img_feat = self.backbone.body.block4.unit03.conv3.out_channels 
        self.classifier = nn.Conv2d(img_feat + num_meta_data, 1, kernel_size=(1, 1), stride=(1, 1))
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, im, meta_data=None):
        
        feat = self.backbone(im)
        if meta_data is not None:
            feat = torch.cat([feat, meta_data.view([-1,1,1,1])], 1)      
        pred = self.classifier(feat)
        assert pred.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        
        return pred[...,0,0]
    
    
    def load(self, pretrained_dict, fresh_head=False):
        
        if fresh_head:
            pretrained_dict = {k:v for k,v in pretrained_dict.items() if ('head' not in k)}
            model_dict = self.backbone.state_dict()
            model_dict.update(pretrained_dict) 
            self.backbone.load_state_dict(model_dict)
            print('loaded pretrain')
        else:
            self.load_state_dict(pretrained_dict)



BiTFeatEx = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2_FeatExtractor([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2_FeatExtractor([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2_FeatExtractor([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2_FeatExtractor([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2_FeatExtractor([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2_FeatExtractor([3, 8, 36, 3], 4, *a, **kw)),
])


def load_pretrained(model, pretrained, head_name, fresh_head_weights):
    pretrained_dict = torch.load(pretrained, map_location="cpu")['model']
    # we remove "module" if there since it was created by data parallel training
    pretrained_dict = {k.split('module.')[-1]: v for k, v in pretrained_dict.items()} 
    
    if fresh_head_weights:
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if (head_name not in k)}
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    print(f'Loaded pretraining weights from {pretrained}')

    
def instantiate_model(architecture, num_classes, pretrained='imagenet', fresh_head_weights=False, num_meta=0):

    if architecture == 'densenet121':
        model = torchvision.models.densenet121(pretrained= (pretrained == 'imagenet'))
        model.classifier = torch.nn.Linear(1024, num_classes)
        if pretrained and (pretrained != 'imagenet'):
            head_name = 'classifier'
            load_pretrained(model, pretrained, head_name, fresh_head_weights)
                
    if architecture.startswith('BiT-'):
        print('BIT')
        assert num_meta == 0
        model = BiTModels[architecture](head_size=num_classes, zero_head=True)
        if pretrained == 'imagenet':
            weights = np.load(os.path.expanduser(f'~/models/BiT/{architecture}.npz'))
            model.load_from(weights)
        elif pretrained:
            head_name = 'head'
            load_pretrained(model, pretrained, head_name, fresh_head_weights)
    elif architecture.startswith('BiTX-'):
        print('BiTX')
        backbone = BiTFeatEx[architecture.replace('BiTX', 'BiT')](head_size=num_classes, zero_head=True)
        model = BiT_MetaMixin(num_meta, backbone)
        head_name = 'classifier'
        if pretrained == 'imagenet':
            weights = np.load(os.path.expanduser(f'~/models/BiT/{architecture}.npz'))
            model.backbone.load_from(weights)
        elif pretrained:
            print('pretrained')
            pretrained_dict = torch.load(pretrained, map_location="cpu")['model']
            pretrained_dict = {k.split('module.')[-1]: v for k, v in pretrained_dict.items()} 
            model.load(pretrained_dict, fresh_head_weights)
    else:
        assert False, 'Architecture unknown'

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

    def __init__(self, train_cols=None, device=None, pos_weight=None):
    
        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cols = torch.tensor([train_cols]).to(device) if train_cols is not None else None
        self.pos_weight = torch.tensor(pos_weight).to(device) if pos_weight is not None else None
    
    def __call__(self, x, y):
                
        mask = y>=0
        if self.train_cols is not None:
            mask *= self.train_cols
        if mask.sum()>0:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x[mask], y[mask], reduction='sum')
        else:
            loss = torch.tensor(0).to(y.device)

        return loss, mask.sum()


    
