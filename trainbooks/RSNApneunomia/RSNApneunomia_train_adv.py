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
import os, glob, copy
import json_tricks as json

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

# +
import sys
sys.path.append('../../utils/')
sys.path.append('/home/users/jsoelter/Code/big_transfer/')

import data_loader, evaluations, model_setup, sacred
# -

import train_utils
import multi_head_modules as multihead

# ### Pretrained setup

# Load a pretrained (biased) model

# +
model_checkpoint = '/home/users/jsoelter/models/rsna/bitm/new_exp/mixed_15000_1_5_it1/step00464.pt'

dirname = os.path.dirname(model_checkpoint)
ledger = json.load(open(os.path.join(dirname, 'train_ledger.json')))
# -

# ### Parameter managment

# + tags=["parameters"]
defaults = {} # default parameter, will be overwritten if set explicilty in this notebooke
overwrites = {} # parameter that OVERWRITE all parameters set in this notebook. Usefull if Notebook is executed externally 
# -

p_pretrain = sacred.ParameterStore(defaults=ledger['train_setup']['0']['setup'])

p = sacred.ParameterStore(overwrites=overwrites, defaults=defaults)

# ### Model Setup

device = torch.device("cuda:0")

# #### Pretrained model

# +
p.model_dict = copy.deepcopy(p_pretrain.model_dict)

p.model_dict['pretrained'] = model_checkpoint
p.model_dict['fresh_head_weights'] = False

model = model_setup.instantiate_model(**p.model_dict)
model = model.to(device)
# -

# #### Deconfounder Head

# +
p.deconfounder_head = {
    'features': {
        'link_layers': ['head.avg'], 
        'out_channels': [6144]
    },
    'model': {
        'class_name': 'multihead.ClassificationHead',
        'param_dict': {}
}}

feature_extractor = multihead.FeatureExtractor(backbone=model)
deconfounder = multihead.AttachedHead(feature_extractor, p.deconfounder_head)
_ = deconfounder.to(device)

# +
p.computation = {
    'model_out': '/home/users/jsoelter/models/rsna/bitm/new_exp/deconf_test2',
}

if not os.path.exists(p.computation['model_out']):
    os.makedirs(p.computation['model_out'])

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
# -

# ### Data Setup

# #### Data Setup of Pretrained Model

p.data_setup = copy.deepcopy(p_pretrain.data_setup)
p.data_setup = {
    'data': { 
        'include_meta_features': ['Sex']
}}
p.sampling_config = copy.deepcopy(p_pretrain.sampling_config)

p.data_setup

# #### Y-independent data

p.data_setup['data_y0'] = {
    'include_meta': [], 
    'include_meta_features': ['Sex'],
    'subset': {
        'Target': [0]
    },
    'val_conf': p.data_setup['data']['val_conf'].copy()
}

# #### Augmentation Pipeline

preprocess = data_loader.transform_pipeline_from_dict(p.data_setup['transforms'])

# #### Datasts

# +
train_data = data_loader.RSNAPneumoniaData(
    transform=preprocess, 
    sub_sampling = p.sampling_config, 
    validation=False,  
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/', 
    **p.data_setup['data']
)

valid_data = data_loader.RSNAPneumoniaData(
    transform=preprocess, 
    sub_sampling=p.sampling_config, 
    validation=True,
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/', 
    **p.data_setup['data']                                    
)

train_data_y0 = data_loader.RSNAPneumoniaData(
    transform=preprocess, 
    sub_sampling=p.sampling_config, 
    validation=False,
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data_y0']
)

valid_data_y0 = data_loader.RSNAPneumoniaData(
    transform=preprocess,
    sub_sampling=p.sampling_config,
    validation=True, 
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data_y0'])


# -
# ####  External Test Data

testset_sampling = {
    'meta_field': 'Sex',
    'meta_values': ['M', 'F'],
    'frac_meta0': 0.5,
    'frac_meta0_tar1': 0.3,
    'frac_meta1_tar1': 0.3,
}

test_data = data_loader.RSNAPneumoniaData(
    transform=preprocess, 
    sub_sampling=testset_sampling, 
    validation=True,
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/', 
    **p.data_setup['data']                                    
)

# ## Deconfounder Training

# ### Evaluation

# #### Validation data

# +
computational_setup = dict(
    num_workers = 8,
    batch_size = 16
)

valid_loader = torch.utils.data.DataLoader(valid_data, **computational_setup)
test_loader = torch.utils.data.DataLoader(test_data, **computational_setup)


# -

# #### Function to do validation prediction

def batch_prediction(head, loader, max_batch=-1, tta_ensemble = 1, device=None):
    
    head.eval()
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble, targets, meta = [], [], []
    for i in range(tta_ensemble):
        preds, targs, metas = [], [], []
        with torch.no_grad():
            for i, (x, t, m) in enumerate(loader):
                x, m = x.to(device), m.numpy()
                logits = head(x)
                preds.append(logits.to('cpu').numpy())
                meta.append(m)
                targs.append(t)
                if i == max_batch:
                    break

        ensemble.append(np.vstack(preds))
        targets.append(np.vstack(targs))
        metas.append(np.vstack(meta))
   
    return np.array(ensemble).squeeze(), targets[0], metas[0]


# #### Scores

import sklearn.metrics as skm

preds, targets, meta = batch_prediction(deconfounder, test_loader, device=device)    
print(f'AUC M vs F: {skm.roc_auc_score(meta>0, preds.reshape(-1, 1)):.3f}')
preds, targets, meta = batch_prediction(model, test_loader, device=device)    
print(f'AUC ext.: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}')
preds, targets, _ = batch_prediction(model, valid_loader, device=device)
print(f'AUC int.: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}')


# ### Confounder Training

def train_step_confounder_head(model, data_iter, optimizer, device=device):
    
    x, y, m = next(data_iter)
    # Schedule sending to GPU(s)
    x = x.to(device, non_blocking=True)
    m = (m>0).float().to(device, non_blocking=True)

    # update confounder prediction
    optimizer.zero_grad()
    logits = model(x)            
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, m)
    loss.backward()    
    optimizer.step()
        
    return float(loss.data.cpu().numpy())


# #### Confounder Optimizer

# Optimizer to train confounder head

# +
p.opt_conf = {
    'class': 'SGD',
    'param': dict(
        lr = 1E-4,
        momentum=0.9,
        nesterov = True
    )
}

optim_conf = getattr(torch.optim, p.opt_conf['class'])(deconfounder.head.parameters(), **p.opt_conf['param'])
# -

# #### Y-restricted data

# +
computational_setup_conf = dict(
    num_workers = 8,
    batch_size = 16
)

valid_y0_loader = torch.utils.data.DataLoader(valid_data_y0, **computational_setup_conf)

train_y0_loader = torch.utils.data.DataLoader(train_data_y0, **computational_setup_conf)
train_y0_iter = train_utils.EndlessIterator(train_y0_loader)
# -

# #### Train confounder head for 1 epoch

# +
num_epochs = 1
current_epoch = train_y0_iter.epochs
finish_epoch = current_epoch + num_epochs

losses = []
steps = 0
while current_epoch < finish_epoch:
    loss = train_step_confounder_head(deconfounder, train_y0_iter, optim_conf)
    losses.append(loss)
    steps += 1
    if train_y0_iter.epochs != current_epoch:
        current_epoch = train_y0_iter.epochs
        preds, targets, meta = batch_prediction(deconfounder, test_loader, device=device)
        auc = evaluations.eval_auc(preds.reshape(-1, 1), meta>0)
        print(f'Train: {np.mean(losses):.2f}, AUC: {auc:.3f}')
# -

preds, targets, meta = batch_prediction(deconfounder, test_loader, device=device)    
print(f'AUC M vs F: {skm.roc_auc_score(meta>0, preds.reshape(-1, 1)):.3f}')
preds, targets, meta = batch_prediction(model, test_loader, device=device)    
print(f'AUC ext.: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}')
preds, targets, _ = batch_prediction(model, valid_loader, device=device)
print(f'AUC int.: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}')

# ### Deconfounding Training

# +
computational_setup_deconf = dict(
    num_workers = 8,
    batch_size = 32
)

train_loader = torch.utils.data.DataLoader(train_data, **computational_setup_deconf)
train_iter = train_utils.EndlessIterator(train_loader)


# -


def train_step(model, deconfounder, data_iter, optimizer, batch_acc=1, alpha=0.5, meta_injection=False, device=device):
    
    optimizer.zero_grad()
    for i in range(batch_acc):
        
        x, y, m = next(data_iter)
        # Schedule sending to GPU(s)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # select y restricted data
        y_restriction = y == 0
        # skip update if 
        if y_restriction.sum() < 2:
            continue
        m = (m>0).float().to(device, non_blocking=True)

        # update predictor
        logits_pred = model(x)
        loss_pred = torch.nn.functional.binary_cross_entropy_with_logits(logits_pred, y)
        
        logits_conf = deconfounder()[y_restriction] 
        targets_conf = m[y_restriction]

        loss_conf = torch.abs(
            torch.nn.functional.cosine_similarity(
                logits_conf - logits_conf.mean(), 
                targets_conf - targets_conf.mean(), 
                dim=0))
        
        #loss_conf = torch.nn.functional.binary_cross_entropy_with_logits(
        #    logits_conf.squeeze(), 
        #    0.5*torch.ones_like(logits_conf.squeeze())
        #)
    
        loss_full = loss_pred + alpha*loss_conf
        (loss_full/batch_acc).backward()

    optimizer.step()
    losses = np.array([float(loss_pred.data.cpu().numpy()), float(loss_conf.data.cpu().numpy())])
    
    return losses


# +
p.opt_deconf = {
    'class': 'SGD',
    'param': dict(
        lr = 1E-4,
        momentum=0.5,
        nesterov = False
    )
}

optim_pred = getattr(torch.optim, p.opt_deconf['class'])(model.parameters(), **p.opt_deconf['param'])


# -

def create_checkpoint():

    torch.save({
            "step": step,
            "backbone": model.state_dict(),
            "deconf_head": deconfounder.head.state_dict(),
            "optim_model": optim_pred.state_dict(),
            "optim_conf" : optim_conf.state_dict(),
        }, 
        os.path.join(p.computation['model_out'], f'step{step:05d}.pt')
    )
    json.dump(ledger, open(os.path.join(p.computation['model_out'], 'train_ledger.json'), 'w'))
    print(f'saved at step {step}')


# +
max_steps = 100
save_intervall = 5
eval_intervall = 1


train_steps_target = 50
train_steps_confounder= 200
batch_acc = 1
loss_alpha = 0.5

p.train_loop = dict(
    train_steps_target = train_steps_target,
    train_steps_confounder = train_steps_confounder,
    batch_acc = batch_acc,
    loss_alpha = loss_alpha
)

train_setup = ledger.setdefault('train_setup', {})
train_setup[step] = {
    'setup': p.params
}

# save initial model (to better compare improvements)
create_checkpoint()

_ = model.train()
while step < max_steps:
        
    # train target + deconfounding
    losses = np.zeros(2)
    for i in range(train_steps_target):
        losses += train_step(model, deconfounder, train_iter, optim_pred, batch_acc = batch_acc, alpha = loss_alpha)
    ledger['train_losses_targets'].append(losses/train_steps_target)
        
    if (step % eval_intervall) == 0:
        print(f'======== {step} =========')

        preds, targets, meta = batch_prediction(deconfounder, test_loader, device=device)    
        score = skm.roc_auc_score(meta>0, preds.reshape(-1, 1))
        ledger['auc_val_deconf'].append((step, score))
        print(f'AUC M vs F: {score:.3f}')
        
        preds, targets, meta = batch_prediction(model, test_loader, device=device)
        score = skm.roc_auc_score(targets, preds.reshape(-1, 1))
        ledger['auc_val_tar_ext'].append(step, score)
        print(f'AUC ext.: {score:.3f}')
        
        preds, targets, _ = batch_prediction(model, valid_loader, device=device)
        score = skm.roc_auc_score(targets, preds.reshape(-1, 1))
        ledger['auc_val_tar_int'].append(step, score)
        print(f'AUC int.: {score:.3f}')

    
    # train confounder head
    loss = 0
    for i in range(train_steps_confounder):    
        loss += train_step_confounder_head(deconfounder, train_y0_iter, optim_conf)
    ledger['train_loss_conf'].append(loss/train_steps_confounder)

        
    if (step % eval_intervall) == 0:
        preds, targets, meta = batch_prediction(deconfounder, test_loader, device=device)
        score = skm.roc_auc_score(meta>0, preds.reshape(-1, 1))
        ledger['auc_val_conf'].append((step, score))
        print(f'AUC M vs F: {score:.3f}')
        
    
    if (step % save_intervall) == 0:
        create_checkpoint()
        
    step += 1


# +
#torch.cat((a,t[2].unsqueeze(2).unsqueeze(3).to('cuda')), 1).shape
# -

create_checkpoint()




