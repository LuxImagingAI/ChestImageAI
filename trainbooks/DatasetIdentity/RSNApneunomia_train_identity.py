# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
import timm
from PIL import Image
import torch
import sklearn

import torchvision.models
# +
import sys
sys.path.append('../../utils/')
sys.path.append('../../../big_transfer/')

import data_loader, evaluations, model_setup, sacred
import train_utils
# -

# ### Parameter managment

# + tags=["parameters"]
defaults = {} # default parameter, will be overwritten if set explicilty in this notebooke
overwrites = {} # parameter that OVERWRITE all parameters set in this notebook. Usefull if Notebook is executed externally 
# -

p = sacred.ParameterStore(overwrites=overwrites)

# ### Model Setup

# +
#p.model_dict = {
#        'model_name': 'tf_efficientnet_b1',
#        'pretrained': 'imagenet', 
#        'num_classes': 1, 
#}

p.model_dict =  dict(
    architecture = 'BiT-M-R50x3',
    #architecture = 'densenet121',
    num_classes = 1,
    pretrained = 'imagenet', 
    #pretrained = '/home/users/jsoelter/models/chexpert/fullmeta_503_consolidation_new/step00200.pt', #None, #'imagenet','imagenet', #
    fresh_head_weights = True,
    num_meta=0
)

# +
p.computation = {
    'model_out': '/work/projects/covid19_dv/models/dataset_identity/jan/new_tiny_mucho',
    'device': "cuda:0"
}

if not os.path.exists(p.computation['model_out']):
    os.makedirs(p.computation['model_out'])

#model = timm.create_model(**p.model_dict)
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

# ### Data Setup

# +
p.datasets = {
    'data_enrich': {
        #'class_name': 'data_loader.BIMCVPneumoniaData',
        #'param_dict':  {
        #    'include_meta_features': ['source'],
        #    'meta_mapping': {
        #        'source': {'positive': 1, 'negative': 1},
        #    },
        #    'subset': {'AP/PA': ['vp-pa','vp-ap']},
        #    'val_conf': {
        #        'salt': '42',
        #        'fraction': 0.2,
        #}}
        'class_name': 'data_loader.ChexpertData',
        'param_dict' : {
            'meta_csv': '/work/projects/covid19_dv/raw_data/heavy_datasets/chexpert_stanford/CheXpert-v1.0/train.csv',
            'datapath': '/work/projects/covid19_dv/raw_data/heavy_datasets/chexpert_stanford/',
            'include_meta_features': ['source'],
            'labels': ['Pneumonia'],
            'subset': {
                'AP/PA':['AP', 'PA'],
                'Pneumonia': [1., np.nan]
            },
            'val_conf': {
                'salt': '42',
                'fraction': 0.06,
        }}
    },
    'data_base': {
        'class_name': 'data_loader.RSNAPneumoniaData',
        'param_dict': {
            'datapath': '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',
            'include_meta_features': ['source'],
            'val_conf': {
                'salt': '42',
                'fraction': 0.05,
        }}
    },
}

    
p.data_setup = {
    'transforms': [
        ('ToPILImage', {}),
        ('Resize', {
            'size': 256+32 #smaller edege mapped to x
        }),
        #('Resize', {
        #    'size': 544
        #}),
        ('RandomRotation', {
            'degrees': 5
        }),    
        ('RandomCrop', {
            'size': (256, 256)
        }),
        ('ToTensor', {}),
        ('Normalize', {
            'mean': [0.485, 0.456, 0.406], 
            #'mean': (0.5, 0.5, 0.5),
            'std': [0.229, 0.224, 0.225]  
            #'std': (0.5, 0.5, 0.5)
        })]
}

# +
p.subsample = {
    'data_base': {
        'num_positive': 500,
        'num_negative': 1500
    },
    'data_enrich': {
        'num_positive': 2000,
        'num_negative': 6000
    }
}

def subsample(df, num_positive=None, num_negative=None, column='Target', salt='42'):
    
    if (num_positive is not None) or (num_negative is not None):
        counts = df[column].value_counts()

        df['sampling_factor'] = 1
        df.loc[df[column]== 0, 'sampling_factor'] = num_negative / counts.loc[0]
        df.loc[df[column]== 1, 'sampling_factor'] = num_positive / counts.loc[1] 

        selection = df.apply(lambda x: data_loader.hash_sample(salt + x['Path'], x['sampling_factor']), axis=1)
        out = df[selection].reset_index().drop(columns=['index'])
    
    else:
        out = df 
    
    return out


# +
preprocess = data_loader.transform_pipeline_from_dict(p.data_setup['transforms'])

train_data1 = train_utils.instantiate_object(**p.datasets['data_base'], transform = preprocess, validation=False)
train_data1.meta_df['source'] = 1
train_data1.meta_df = subsample(train_data1.meta_df, **p.subsample['data_base'])

val_data1 = train_utils.instantiate_object(**p.datasets['data_base'], transform = preprocess, validation=True)
val_data1.meta_df['source'] = 1


train_data2 = train_utils.instantiate_object(**p.datasets['data_enrich'], transform = preprocess, validation=False)
train_data2.meta_df['source'] = 0

# Quick Hack: remove everything which is not Normal/Pneunomia
m = np.logical_or(train_data2.meta_df['Pneumonia'] > 0, train_data2.meta_df['No Finding'] > 0)
train_data2.meta_df = train_data2.meta_df[m].reset_index().drop(columns=['index'])
train_data2.meta_df = subsample(train_data2.meta_df, **p.subsample['data_enrich'], column='Pneumonia')

val_data2 = train_utils.instantiate_object(**p.datasets['data_enrich'], transform = preprocess, validation=True)
val_data2.meta_df['source'] = 0

m = np.logical_or(val_data2.meta_df['Pneumonia'] > 0, val_data2.meta_df['No Finding'] > 0)
val_data2.meta_df = val_data2.meta_df[m].reset_index().drop(columns=['index'])

train_data = torch.utils.data.ConcatDataset([train_data1, train_data2])
val_data = torch.utils.data.ConcatDataset([val_data1, val_data2])
# -
len(train_data1), len(train_data2)

# +
dataset = train_data

plt.figure(figsize=(20,8))
for i, ix in enumerate(np.random.randint(0, len(dataset), 8)):
    plt.subplot(2,4,i+1)
    im, tar, meta = dataset[ix]
    plt.title(f'has pneunomia: {tar == 1}, dataset_id: {meta}')
    plt.imshow(im[0], cmap='bone')

# +
real_batch_size = 64 #
batch_split = 4 #number of forward pathes before optimization is performed 

train_loader = torch.utils.data.DataLoader(train_data, batch_size=int(real_batch_size/batch_split), num_workers=8, shuffle=True, drop_last=False)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, num_workers=6)
val_loader1 = torch.utils.data.DataLoader(val_data1, batch_size=64, num_workers=6)
val_loader2 = torch.utils.data.DataLoader(val_data2, batch_size=64, num_workers=6)

steps_per_epoch = int(len(train_loader)/batch_split)
print(f'steps per epoch: {steps_per_epoch}')
# -

# ### Optimizer Setup

#opt = 'Adam' #'Adam'#
p.opt = {
    'class': 'SGD',
    'param': dict(
        lr = 2E-4,
        momentum=0.9,
        nesterov = True
    )
}

optim = getattr(torch.optim, p.opt['class'])(model.parameters(), **p.opt['param'])
if  checkpoint is not None:
    optim.load_state_dict(checkpoint["optim"])
else:
    optim.zero_grad()

# +
plateu_length = 200

p.scheduler = dict(
    supports = [int(0.05*plateu_length), 1*plateu_length, 2*plateu_length, 3*plateu_length, 4*plateu_length]
)
#supports = [2*steps_per_epoch, 3*steps_per_epoch, 4*steps_per_epoch, 6*steps_per_epoch, 8*steps_per_epoch]#[3000, 7000, 9000, 10000]
# -

# ### Loss

# +
p.loss_param = {
    #'pos_weight': 2.3
}

crit = model_setup.maskedBCE(**p.loss_param)


# -

# ### Initial errors

def batch_prediction(model, loader, max_batch=-1, tta_ensemble = 1, device=None):
    
    model.eval()
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble, targets, meta = [], [], []
    for i in range(tta_ensemble):
        preds, targs, metas = [], [], []
        with torch.no_grad():
            for i, (x, t, m) in enumerate(loader):
                x, m = x.to(device), m.numpy()
                logits = model(x)
                preds.append(logits.to('cpu').numpy())
                meta.append(m)
                targs.append(t)
                if i == max_batch:
                    break

        ensemble.append(np.vstack(preds))
        targets.append(np.vstack(targs))
        metas.append(np.vstack(meta))
   
    return np.array(ensemble).squeeze(), targets[0], metas[0]


# +
eval_meta = False # evaluate meta or target prediction

if eval_meta:
    preds, targets, meta = batch_prediction(model, val_loader, device=device)
    auc = sklearn.metrics.roc_auc_score(meta, preds.reshape(-1, 1))
    print(f'auc meta: {auc:.3f}')
else:
    preds, targets, meta = batch_prediction(model, val_loader1, device=device)
    auc1 = sklearn.metrics.roc_auc_score(targets, preds.reshape(-1, 1))
    preds, targets, meta = batch_prediction(model, val_loader2, device=device)
    auc2 = sklearn.metrics.roc_auc_score(targets, preds.reshape(-1, 1))
    print(f'auc1: {auc1:.3f}, auc2: {auc2:.3f}')
# -

# ### Training Loop

eval_intervall = 10
save_intervall = 500 #steps_per_epoch

# +
train_meta = False # evaluate meta or target prediction

accum_steps = 0
batch_loss, batch_samples = 0, 0
lr = p.opt['param']['lr']

train_setup = ledger.setdefault('train_setup', {})
train_setup[step] = {
    'setup': p.params,
    'real_batch_size': real_batch_size,
    'batch_split': batch_split
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
        if train_meta:
            y = m.to(device, non_blocking=True)
        else:
            y = y.to(device, non_blocking=True)

        logits = model(x)            
        loss, n_samples = crit(logits, y)
        if loss != 0:
            # Accumulate grads
            (loss / batch_split / n_samples).backward()

        batch_loss += float(loss.data.cpu().numpy())  # Also ensures a sync point.
        batch_samples += n_samples.cpu().numpy()

        accum_steps += 1

        # Update params
        if accum_steps == batch_split:
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
                if eval_meta:
                    preds, targets, meta = batch_prediction(model, val_loader, device=device)
                    auc = sklearn.metrics.roc_auc_score(meta, preds.reshape(-1, 1))
                    ledger['val_auc_meta'].append((step-1, auc))
                    print(f'step {step} ->, train: {train_loss:.3f}, auc meta: {auc:.3f}')
                else:
                    preds, targets, meta = batch_prediction(model, val_loader1, device=device)
                    auc1 = sklearn.metrics.roc_auc_score(targets, preds.reshape(-1, 1))
                    preds, targets, meta = batch_prediction(model, val_loader2, device=device)
                    auc2 = sklearn.metrics.roc_auc_score(targets, preds.reshape(-1, 1))
                    ledger['val_auc'].append((step-1, auc1, auc2))
                    print(f'step {step} ->, train: {train_loss:.3f},  auc1: {auc1:.3f}, auc2: {auc2:.3f}') # FULL: 

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
        "optim" : optim.state_dict(),
    }, 
    os.path.join(p.computation['model_out'], f'step{step:05d}.pt')
)
json.dump(ledger, open(os.path.join(p.computation['model_out'], 'train_ledger.json'), 'w'))






