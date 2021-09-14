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
sys.path.append('../../../big_transfer/')

import data_loader, evaluations, model_setup, sacred
# -

import train_utils
import multi_head_modules as multihead

# ### Pretrained setup

# Load a pretrained (biased) model

# +
model_checkpoint = '/work/projects/covid19_dv/models/rsna/bitm/new_exp/new_exp/mixed_15000_1_5_it1/step00464.pt'

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
p.discrepancy_subhead = {
    'features': {
        'link_layers': ['head.avg'], 
        'out_channels': [6144]
    },
    'model': {
        'class_name': 'multihead.ClassificationHead',
        'param_dict': {}
}}

feature_extractor = multihead.FeatureExtractor(backbone=model)
discrepancy_subhead1 = multihead.AttachedHead(feature_extractor, p.discrepancy_subhead)
discrepancy_subhead0 = multihead.AttachedHead(feature_extractor, p.discrepancy_subhead)

_ = discrepancy_subhead1.to(device)
_ = discrepancy_subhead0.to(device)

# +
p.computation = {
    'model_out': '/home/users/jsoelter/models/rsna/bitm/new_exp/discrepancy_test3',
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

p.sampling_config

# #### C-dependent data

# +
p.data_setup['data_c0'] = {
    'include_meta': [], 
    'subset': {
        'Sex': [-1]
    },
    'val_conf': p.data_setup['data']['val_conf'].copy()
}

p.data_setup['data_c1'] = {
    'include_meta': [], 
    'subset': {
        'Sex': [1]
    },
    'val_conf': p.data_setup['data']['val_conf'].copy()
}
# -

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


# +
train_data_c1 = data_loader.RSNAPneumoniaData(
    transform=preprocess, 
    sub_sampling=p.sampling_config, 
    validation=False,
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data']
)
train_data_c1.meta_df = train_data_c1.meta_df[train_data_c1.meta_df['Sex'] == 1]
train_data_c1.meta_df = train_data_c1.meta_df.reset_index().drop(columns='index')


train_data_c0 = data_loader.RSNAPneumoniaData(
    transform=preprocess, 
    sub_sampling=p.sampling_config, 
    validation=False,
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data']
)
train_data_c0.meta_df = train_data_c0.meta_df[train_data_c0.meta_df ['Sex'] != 1]
train_data_c0.meta_df = train_data_c0.meta_df.reset_index().drop(columns='index')



val_data_c1 = data_loader.RSNAPneumoniaData(
    transform=preprocess, 
    sub_sampling=p.sampling_config, 
    validation=True,
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data']
)
val_data_c1.meta_df = val_data_c1.meta_df[val_data_c1.meta_df['Sex'] == 1]
val_data_c1.meta_df = val_data_c1.meta_df.reset_index().drop(columns='index')


val_data_c0 = data_loader.RSNAPneumoniaData(
    transform=preprocess, 
    sub_sampling=p.sampling_config, 
    validation=True,
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data']
)
val_data_c0.meta_df = val_data_c0.meta_df[val_data_c0.meta_df ['Sex'] != 1]
val_data_c0.meta_df = val_data_c0.meta_df.reset_index().drop(columns='index')
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

# ## Discrepancy Training

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
from scipy.special import expit

#preds, targets, meta = batch_prediction(deconfounder, test_loader, device=device)    
#print(f'AUC M vs F: {skm.roc_auc_score(meta>0, preds.reshape(-1, 1)):.3f}')
preds, targets, meta = batch_prediction(model, test_loader, device=device)    
print(f'AUC ext.: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}')
preds, targets, _ = batch_prediction(model, valid_loader, device=device)
print(f'AUC int.: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}')


# ### Confounder Training

def train_step_discrepancy_head(model_c1, model_c0, data_iter_c1, data_iter_c0, optimizer, alpha=1, device=device):

    optimizer.zero_grad()
    
    # part 1: maximize predictions in sub-cohort
    x, y, m = next(data_iter_c1)
    x = x.to(device, non_blocking=True)
    y_c1 = y.to(device, non_blocking=True)
    
    logits_c1onc1 = model_c1(x)
    logits_c0onc1 = model_c0()
    
    # part 2 : maximize discrepancy in opposite cohort
    x, y, m = next(data_iter_c0)
    x = x.to(device, non_blocking=True)
    y_c0 = y.to(device, non_blocking=True)

    logits_c1onc0 = model_c1(x)
    logits_c0onc0 = model_c0()    
    
    c_loss = torch.nn.functional.binary_cross_entropy_with_logits
    reweighting = torch.Tensor([9]).to(device)
    loss_classification = c_loss(logits_c1onc1, y_c1) + c_loss(logits_c0onc0, y_c0, pos_weight=reweighting) 

    d_loss = lambda logit1, logit2: torch.mean(torch.abs(torch.sigmoid(logit1)-torch.sigmoid(logit2)))       
    loss_discrepancy = d_loss(logits_c1onc0, logits_c0onc0) + d_loss(logits_c1onc1, logits_c0onc1)
    
    loss = loss_classification - alpha * loss_discrepancy
    loss.backward()    
    optimizer.step()
        
    return  float(loss_classification.data.cpu().numpy()), float(loss_discrepancy.data.cpu().numpy())


# #### Max Discrepancy

# Optimizer to train confounder head

# +
p.opt_conf = {
    'class': 'SGD',
    'param': dict(
        lr = 1E-4,
        momentum=0.9,
        nesterov = False
    )
}

trainable_params = list(discrepancy_subhead0.head.parameters()) + list(discrepancy_subhead1.head.parameters())
optim_conf = getattr(torch.optim, p.opt_conf['class'])(trainable_params, **p.opt_conf['param'])
# -

# #### Confounder seperated data

# +
computational_setup_conf = dict(
    num_workers = 8,
    batch_size = 16
)

train_c0_loader = torch.utils.data.DataLoader(train_data_c0, **computational_setup_conf)
train_c0_iter = train_utils.EndlessIterator(train_c0_loader)

train_c1_loader = torch.utils.data.DataLoader(train_data_c1, **computational_setup_conf)
train_c1_iter = train_utils.EndlessIterator(train_c1_loader)

val_c0_loader = torch.utils.data.DataLoader(val_data_c0, **computational_setup_conf)
val_c1_loader = torch.utils.data.DataLoader(val_data_c1, **computational_setup_conf)
# -

# #### (pre)Train discrepancy head

# +
num_steps = 200

losses = []
steps = 0
while steps < num_steps:
    loss = train_step_discrepancy_head(discrepancy_subhead1, discrepancy_subhead0, train_c1_iter, train_c0_iter, optim_conf, alpha=0.5)
    losses.append(loss)
    steps += 1
    if steps%50 == 0:
        print(f'========================{steps}=======================')
        print('Train: {0:.2f}, {1:.2f}'.format(*np.mean(np.array(losses)[-100:],0)))
        preds, targets, _ = batch_prediction(discrepancy_subhead0, val_c0_loader, device=device)    
        print(f'     cohort AUC: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}, logloss: {skm.log_loss(targets, expit(preds.reshape(-1, 1))):.3}')
        preds, targets, _ = batch_prediction(discrepancy_subhead0, val_c1_loader, device=device)
        print(f'anti-cohort AUC: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}, logloss: {skm.log_loss(targets, expit(preds.reshape(-1, 1))):.3}')
# -

if False:
    fig, ax = plt.subplots(1,2,figsize=(20,5))

    x, y, m = next(train_c0_iter)
    x = x.to(device, non_blocking=True)
    p_c0 = torch.sigmoid(discrepancy_subhead0(x))
    p_c1 = torch.sigmoid(discrepancy_subhead1())

    ax[0].plot(p_c0.detach().cpu().numpy(), label = 'model c0')
    ax[0].plot(p_c1.detach().cpu().numpy(), label = 'model c1')
    ax[0].plot(y, alpha=0.1, color='k')

    plt.legend()
    _ = ax[0].set_title(f'data c0')

    x, y, m = next(train_c1_iter)
    x = x.to(device, non_blocking=True)
    p_c0 = torch.sigmoid(discrepancy_subhead0(x))
    p_c1 = torch.sigmoid(discrepancy_subhead1())

    ax[1].plot(p_c0.detach().cpu().numpy(), label = 'model c0')
    ax[1].plot(p_c1.detach().cpu().numpy(), label = 'model c1')
    ax[1].plot(y, alpha=0.1, color='k')

    plt.legend()
    _ = plt.title(f'data c1')


# ### Deconfounding Training

# +
computational_setup_deconf = dict(
    num_workers = 8,
    batch_size = 32
)

train_loader = torch.utils.data.DataLoader(train_data, **computational_setup_deconf)
train_iter = train_utils.EndlessIterator(train_loader)


# -


def plot_models(train_iter, model, discrepancy_subhead0, discrepancy_subhead1):

        x, y, m = next(train_iter)
        x = x.to(device, non_blocking=True)

        pb = torch.sigmoid(model(torch.sigmoid(x)))
        p_c0 = torch.sigmoid(discrepancy_subhead0())
        p_c1 = torch.sigmoid(discrepancy_subhead1())

        plt.plot(p_c0.detach().cpu().numpy(), label = 'model c0')
        plt.plot(p_c1.detach().cpu().numpy(), label = 'model c1')
        plt.plot(pb.detach().cpu().numpy(), label = 'base')
        plt.plot(y, 'k', alpha=0.2)

        plt.legend()
        score = np.abs(p_c0.detach().cpu() - p_c1.detach().cpu()).mean() 
        
        return score


plot_models(train_iter, model, discrepancy_subhead0, discrepancy_subhead1)


def train_step(base_model, model_c1, model_c0, data_iter, optimizer, alpha=1, device=device):

    optimizer.zero_grad()
    
    # part 1: maximize prediction
    x, y, m = next(data_iter)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    logits = base_model(x)            
    loss_classification = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    
    # part 2 : maximize discrepancy in opposite cohort
    logits_c0 = model_c0()
    logits_c1 = model_c1()
    
    d_loss = lambda logit1, logit2: torch.mean(torch.abs(torch.sigmoid(logit1)-torch.sigmoid(logit2)))       
    loss_discrepancy = d_loss(logits_c0, logits_c1)
    
    loss = loss_classification + alpha * loss_discrepancy
    loss.backward()    
    optimizer.step()
        
    return  float(loss_classification.data.cpu().numpy()), float(loss_discrepancy.data.cpu().numpy())


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
            "discrepancy_subhead0": discrepancy_subhead0.head.state_dict(),
            "discrepancy_subhead1": discrepancy_subhead1.head.state_dict(),
            "optim_model": optim_pred.state_dict(),
            "optim_conf" : optim_conf.state_dict(),
        }, 
        os.path.join(p.computation['model_out'], f'step{step:05d}.pt')
    )
    json.dump(ledger, open(os.path.join(p.computation['model_out'], 'train_ledger.json'), 'w'))
    print(f'saved at step {step}')


# +
max_steps = 100
save_intervall = 100
eval_intervall = 10

train_steps_target = 50
train_steps_discrepancy = 200
loss_alpha = 1

p.train_loop = dict(
    train_steps_target = train_steps_target,
    train_steps_discrepancy = train_steps_discrepancy,
    loss_alpha = loss_alpha
)

train_setup = ledger.setdefault('train_setup', {})
train_setup[step] = {
    'setup': p.params
}

# save initial model (to better compare improvements)
#create_checkpoint()

_ = model.train()
while step < max_steps:
        
    # train target + min discrepancy
    losses = np.zeros(2)
    for i in range(train_steps_target):
        losses += train_step(model, discrepancy_subhead1, discrepancy_subhead0, train_iter, optim_pred, alpha = loss_alpha)
    ledger['train_losses_targets'].append(losses/train_steps_target)
        
    if (step % eval_intervall) == 0:
        print(f'======== {step} =========')

        print('Train: {0:.2f}, {1:.2f}'.format(*ledger['train_losses_targets'][-1]))
        #preds, targets, _ = batch_prediction(discrepancy_subhead0, val_c0_loader, device=device)    
        #print(f'     cohort AUC: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}, logloss: {skm.log_loss(targets, expit(preds.reshape(-1, 1))):.3}')
        #preds, targets, _ = batch_prediction(discrepancy_subhead0, val_c1_loader, device=device)
        #print(f'anti-cohort AUC: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}, logloss: {skm.log_loss(targets, expit(preds.reshape(-1, 1))):.3}')
        
        preds, targets, meta = batch_prediction(model, test_loader, device=device)
        score = skm.roc_auc_score(targets, preds.reshape(-1, 1))
        ledger['auc_val_tar_ext'].append([step, score])
        print(f'AUC ext.: {score:.3f}')
        
        preds, targets, _ = batch_prediction(model, valid_loader, device=device)
        score = skm.roc_auc_score(targets, preds.reshape(-1, 1))
        ledger['auc_val_tar_int'].append([step, score])
        print(f'AUC int.: {score:.3f}')

    
    # train confounder head
    losses = np.zeros(2)
    for i in range(train_steps_discrepancy):    
        losses += train_step_discrepancy_head(discrepancy_subhead1, discrepancy_subhead0, train_c1_iter, train_c0_iter, optim_conf, alpha=0.5)
    ledger['train_loss_disc'].append(losses/train_steps_discrepancy)

        
    if (step % eval_intervall) == 0:
        print('Train: {0:.2f}, {1:.2f}'.format(*ledger['train_loss_disc'][-1]))
        preds, targets, _ = batch_prediction(discrepancy_subhead0, val_c0_loader, device=device)    
        print(f'     cohort AUC: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}, logloss: {skm.log_loss(targets, expit(preds.reshape(-1, 1))):.3}')
        preds, targets, _ = batch_prediction(discrepancy_subhead0, val_c1_loader, device=device)
        print(f'anti-cohort AUC: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}, logloss: {skm.log_loss(targets, expit(preds.reshape(-1, 1))):.3}')
        
    
    if (step % save_intervall) == 0:
        create_checkpoint()
        
    step += 1


# +
#torch.cat((a,t[2].unsqueeze(2).unsqueeze(3).to('cuda')), 1).shape

# +
ledger['train_loss_disc'].append(losses/train_steps_discrepancy)

        
if (step % eval_intervall) == 0:
    print('Train: {0:.2f}, {1:.2f}'.format(*ledger['train_loss_disc'][-1]))
    preds, targets, _ = batch_prediction(discrepancy_subhead0, val_c0_loader, device=device)    
    print(f'     cohort AUC: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}, logloss: {skm.log_loss(targets, expit(preds.reshape(-1, 1))):.3}')
    preds, targets, _ = batch_prediction(discrepancy_subhead0, val_c1_loader, device=device)
    print(f'anti-cohort AUC: {skm.roc_auc_score(targets, preds.reshape(-1, 1)):.3f}, logloss: {skm.log_loss(targets, expit(preds.reshape(-1, 1))):.3}')
# -

plt.plot(np.vstack(ledger['train_losses_targets']))

plt.plot(np.vstack(ledger['train_loss_disc']))


