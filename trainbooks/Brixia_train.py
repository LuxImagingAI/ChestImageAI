# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: base
#     language: python
#     name: base
# ---

# +
import os, glob, json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import collections
import cv2
import tqdm
from PIL import Image
import torch

import torchvision.models
import hashlib, pickle

# +
import sys
sys.path.append('/home/users/jsoelter/Code/ChestImageAI/utils/')
sys.path.append('/home/users/jsoelter/Code/big_transfer/')

import data_loader, evaluations, model_setup, sacred
# -

# ### Parameter managment

# + tags=["parameters"]
defaults = {} # default parameter, will be overwritten if set explicilty in this notebooke
overwrites = {} # parameter that /OVERWRITE all parameters set in this notebook. Usefull if Notebook is executed externally 
# -

p = sacred.ParameterStore(overwrites=overwrites)

# ### Data Setup

p.data_setup = dict(
    data = {
        'global_score': True, 
        'include_meta': [],
        'subset': {'View' : ['AP']}, # Define subsetting of data
        #'equalize': None
        'val_conf': {
                'salt': '40',
                'fraction': 0.05,
            }
    }, 
    prepro = [
        ('ToTensor', {}),
        ('Resize', {
            'size': 440 #smaller edege mapped to x
        })
    ],
    prepro_dynamic = [
        ('Normalize', {
            'mean': [0.485, 0.456, 0.406], 
            #'mean': (0.5, 0.5, 0.5),
            'std': [0.229, 0.224, 0.225]  
            #'std': (0.5, 0.5, 0.5)
        }),
    ],
    train_aug = [
        ('GaussianBlur', {
           'kernel_size': 5,
            'sigma': (0.1, 1)
        }),
        ('RandomRotation', {
            'degrees': 10
        }),
        ('CenterCrop', {
            'size': (400, 400)
        }),
        ('RandomCrop', {
            'size': (384, 384)
        }),

    ],
    test_aug = [
        ('CenterCrop', {
            'size': (384, 384)
        }),
    ],
)

# +
prepro = data_loader.transform_pipeline_from_dict(p.data_setup['prepro'])
train_aug = data_loader.transform_pipeline_from_listdict(p.data_setup, ['prepro_dynamic', 'train_aug'])
test_aug = data_loader.transform_pipeline_from_listdict(p.data_setup, ['prepro_dynamic', 'test_aug'])

train_data = data_loader.BrixiaData(transform=train_aug, deterministic_transform=prepro, **p.data_setup['data'], cache={})
val_data = data_loader.BrixiaData(transform=test_aug, deterministic_transform=prepro, **p.data_setup['data'], cache={}, validation=True)

#external_valid_data = data_loader.BrixiaData()
# -

# #### Visualization of Training Data

# +
im_num = np.random.randint(0, 2000, 4)

fig, axes = plt.subplots(1,4,figsize = (20,5))
fig.suptitle('raw images')
for ix, n in enumerate(im_num):
    row = train_data.meta_df.loc[n]
    dcm = data_loader.pydicom.dcmread(os.path.join(train_data.datapath, 'dicom_clean', row.Filename))
    axes[ix].imshow(dcm.pixel_array, cmap=plt.cm.Spectral_r)
    axes[ix].set_title( f'{int(row.BrixiaScore):06d}')
    
fig, axes = plt.subplots(1,4,figsize = (20,5))
fig.suptitle('train patches rand aug 1')
for ix, n in enumerate(im_num):
    im, tar, meta = train_data[n]
    axes[ix].imshow(im[0], cmap=plt.cm.Spectral_r)
    
fig, axes = plt.subplots(1,4,figsize = (20,5))
fig.suptitle('train patches rand aug 2')
for ix, n in enumerate(im_num):
    im, tar, meta = train_data[n]
    axes[ix].imshow(im[0], cmap=plt.cm.Spectral_r)
# -

fig, axes = plt.subplots(1,4,figsize = (20,5))
im_num = np.random.randint(0, 200, 4)
for ix, n in enumerate(im_num):
    im, tar, meta = val_data[n]
    axes[ix].imshow(im[0], cmap=plt.cm.Spectral_r)

# #### Caching for faster training

cache_folder = os.path.join('/work/projects/covid19_dv/models/brixia/cache')
train_data.preload(cache_folder)
val_data.preload(cache_folder)

# #### Data Loader

# +
p.batch = dict(
    real_batch_size = 16, #512, #64, #128
    batch_split = 1 #16 # #number of forward pathes before optimization is performed 
)
data_batch_size = int(p.batch['real_batch_size']/p.batch['batch_split'])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=data_batch_size, num_workers=8, shuffle=True, drop_last=False)
valid_int_loader = torch.utils.data.DataLoader(val_data, batch_size=data_batch_size, num_workers=8, shuffle=False)
#valid_ext_loader = torch.utils.data.DataLoader(external_valid_data, batch_size=16, num_workers=8)

len(train_loader)/p.batch['batch_split']
# -

import timm
#timm.list_models('*efficient*')

# ### Model Setup

import importlib
importlib.reload(model_setup)

p.model_dict =  dict(
    #architecture = 'BiT-M-R50x3',
    #architecture = 'resnet18',
    architecture = 'timm-resnet101d',
    num_classes = 1,
    #num_heads=6,
    pretrained = 'imagenet', 
    #pretrained = '/home/users/jsoelter/models/chexpert/fullmeta_503_consolidation_new/step00200.pt', #None, #'imagenet','imagenet', #
    fresh_head_weights = True,
    num_meta=0
)

# +
p.computation = {
    'model_out': '/work/projects/covid19_dv/models/brixia/jan/global/onlyAP',
    'device': "cuda:0"
}

if not os.path.exists(p.computation['model_out']):
    os.makedirs(p.computation['model_out'])

model = model_setup.instantiate_model(**p.model_dict)

saved_models = glob.glob(os.path.join(p.computation['model_out'], 'step*.pt'))
if not saved_models:
    checkpoint = None
    ledger = collections.defaultdict(list)
    step = 0
else:
    last_model = np.sort(saved_models)[-1]
    print(f"Resume training for saved model '{last_model}'")
    checkpoint = torch.load(last_model, map_location="cpu")
    re_keyed = {k.split('module.')[-1]: v for k, v in checkpoint['model'].items()}
    model.load_state_dict(re_keyed)
    
    ledger = json.load(open(os.path.join(p.computation['model_out'], 'train_ledger.json')))
    step = checkpoint["step"]

    
# Lets cuDNN benchmark conv implementations and choose the fastest.
# Only good if sizes stay the same within the main loop!
torch.backends.cudnn.benchmark = True

#model = torch.nn.DataParallel(model)
device = p.computation['device']
model = model.to(device)
# -

# ### Optimizer Setup

#opt = 'Adam' #'Adam'#
p.opt = {
    #'class': 'SGD',
    'class': 'Adam',
    'param': dict(
        lr = 1E-4,
        #momentum=0.9,
        #nesterov = True,
        weight_decay = 1E-3
    )
}

optim = getattr(torch.optim, p.opt['class'])(model.parameters(), **p.opt['param'])
if  checkpoint is not None:
    optim.load_state_dict(checkpoint["optim"])
else:
    optim.zero_grad()

# +
p.scheduler = {
    #'supports': [300, 20*len(train_loader), int(40*len(train_loader)), int(60*len(train_loader)), int(80*len(train_loader))]
    #'supports':  [100, 300, 500, 700, 800] 
    #'supports': [300, 1000, 2000, 4000, 6000, 8000]
    #'supports': [300, 600, 1500, 3000]
    'supports': [300, 1000, 2000, 3000]
}

print(p.scheduler['supports'])
# -

# ### Loss

# +
p.loss = {
    #'class': 'L1Loss'
    'class': 'MSELoss'
    #'class': 'CrossEntropyLoss'
}

base_loss = getattr(torch.nn, p.loss['class'])(**p.loss.get('param', {}), reduction='sum')
#crit = lambda x, y: (base_loss(torch.transpose(x, 1,2), y.long()), np.sum(y.shape))
crit = lambda x, y: (base_loss(x, y), np.sum(y.shape))
# -

# ### Initial errors

print(f'Crit: {evaluations.eval_crit(model, valid_int_loader, crit, device=device):.3f}')

# ### Training Loop

eval_intervall = 50
save_intervall = 1000

# +
accum_steps = 0
batch_loss, batch_samples = 0, 0
lr = p.opt['param']['lr']

train_setup = ledger.setdefault('train_setup', {})
train_setup[step] = {
    'setup': p.params
}

while lr:
    for x, y, m in train_loader:
        
        _ = model.train()

        # Update learning-rate, including stop training if over.
        lr = model_setup.get_lr(step, supports=p.scheduler['supports'], base_lr=p.opt['param']['lr'])
        if lr is None: break
        for param_group in optim.param_groups:
            param_group["lr"] = lr
        
        # Schedule sending to GPU(s)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if getattr(model, 'meta_injection', None):
            m = m.to(device, non_blocking=True)
            logits = model(x, m)
        else:
            logits = model(x)            
        loss, n_samples = crit(logits, y)
        if loss != 0:
            # Accumulate grads
            (loss / p.batch['batch_split'] / n_samples).backward()

        batch_loss += float(loss.data.cpu().numpy())  # Also ensures a sync point.
        batch_samples += n_samples

        accum_steps += 1

        # Update params
        if accum_steps == p.batch['batch_split']:
            optim.step()
            optim.zero_grad()
            train_loss = batch_loss/batch_samples
            ledger['train_loss'].append(train_loss)
            batch_loss, batch_samples = 0, 0
            ledger['lr'].append(lr)
            step += 1
            accum_steps = 0
            
            # Evaluate 
            if (step % eval_intervall) == 0:
                val = evaluations.eval_crit(model, valid_int_loader, crit, device=device)
                ledger['internal'].append((step-1, val))
                print(f'step {step} -> train: {train_loss:.3f},  val: {val:.3f}') 

            if (step % save_intervall) == 0:
                torch.save({
                        "step": step,
                        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        "optim": optim.state_dict(),
                    }, 
                    os.path.join(p.computation['model_out'], f'step{step:05d}.pt')
                )
                json.dump(ledger, open(os.path.join(p.computation['model_out'], 'train_ledger.json'), 'w'))
# -

torch.save({
        "step": step,
        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        "optim": optim.state_dict(),
    },
    os.path.join(p.computation['model_out'], f'step{step:05d}.pt')
)
json.dump(ledger, open(os.path.join(p.computation['model_out'], 'train_ledger.json'), 'w'))

p.computation


