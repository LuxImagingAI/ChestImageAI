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
import os, glob
import json_tricks as json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import collections
import cv2
import tqdm
from PIL import Image
import torch
import torchvision.models

import sklearn
import sklearn.metrics
from scipy.special import expit

import scipy.special
import seaborn as sns
import sklearn.metrics as skm

import monai

# +
import sys
sys.path.append('/home/users/jsoelter/Code/ChestImageAI/utils/')
sys.path.append('/home/users/jsoelter/Code/big_transfer/')

import data_loader, evaluations, model_setup, sacred, train_utils
# -

# ## Model Performance Comparison

# + tags=["parameters"]
models = [
  #'/work/projects/covid19_dv/models/brixia/jan/local/att2/',
  #'/work/projects/covid19_dv/models/brixia/jan/local/att1/',
    #'/work/projects/covid19_dv/models/brixia/jan/local/att3/',
    #'/work/projects/covid19_dv/models/brixia/jan/local/att4/',
    #'/work/projects/covid19_dv/models/brixia/jan/local/att5/',
    #'/work/projects/covid19_dv/models/brixia/jan/local/att6/',
  '/work/projects/covid19_dv/models/brixia2/jan/may1',   
    #'/work/projects/covid19_dv/models/brixia/jan/local/att8/',    
    #'/work/projects/covid19_dv/models/brixia/jan/local/att9/',    
    #'/work/projects/covid19_dv/models/brixia/jan/local/att10/',    
    #'/work/projects/covid19_dv/models/brixia/jan/local/att12/',    

   #'/work/projects/covid19_dv/models/brixia/jan/global/plain1/',    
   #'/work/projects/covid19_dv/models/brixia/jan/global/onlyAP/',    

    
]

# -

for model in models:
    ledger = json.load(open(os.path.join(model, 'train_ledger.json')))
    
    fig = plt.figure(figsize=(20,8)) 
    gs = matplotlib.gridspec.GridSpec(3,1,height_ratios=(1,4,4), hspace=0)

    ax = plt.subplot(gs[0])
    plt.plot(ledger['lr'], 'k')
    plt.xticks([])
    plt.ylabel('lr')
    plt.yscale('log')
    plt.xlim([0, 4000])

    ax = plt.subplot(gs[1])
    plt.plot(ledger['train_loss_BrixiaDataD'], alpha=0.1) #, np.hstack([np.zeros(99), np.ones(100)/100]), mode = 'same'))
    plt.plot(np.convolve(ledger['train_loss_BrixiaDataD'], np.hstack([np.zeros(9), np.ones(10)/10]), mode = 'same'), color='b', label='train')
    plt.plot(*np.array([(k, np.mean(v)) for k, v in ledger['validation_BrixiaDataD']]).T, 'o-', label='val')
    #plt.yscale('log')
    plt.legend()
    plt.grid()
    #plt.yscale('log')
    plt.ylabel('cross entropy')
    #plt.xticks([])
    #plt.ylim([2, 7])
    plt.xlim([0, 3000])


# ## Deeper Model Evaluation

# +
model_checkpoint = '/work/projects/covid19_dv/models/brixia2/jan/may1/step03000.pt'
device = torch.device("cuda:0")

dirname = os.path.dirname(model_checkpoint)
ledger = json.load(open(os.path.join(dirname, 'train_ledger.json')))
# -

p = sacred.ParameterStore(defaults=ledger['train_setup']['0']['setup'])

p.transformations

# ### Data Setup

cached_aug = train_utils.transform_pipe_factory(p.transformations, ['cached'], pop_key = ['mask'])
train_aug = train_utils.transform_pipe_factory(p.transformations, ['train_aug']) #, pop_key = ['mask'])

train_data, val_data = {}, {}
for class_name, setup in p.datasets.items():
    
    transforms = {}
    if class_name == 'GeneralBlockchainData':
        transforms['transform'] = train_aug
        transforms['deterministic_transform'] = cached_aug
    else:
        transforms['transform'] = train_aug_nolabel
        transforms['deterministic_transform'] = cached_aug_nolabel

    train_data[class_name] = train_utils.instantiate_object(class_name=class_name,  validation=False, cache = {}, **setup, **transforms)
    val_data[class_name] = train_utils.instantiate_object(class_name=class_name, validation=True, cache = {}, **setup, **transforms)

# +
param = p.data_setup['data'].copy()
param['val_conf'] = {}

test_data = data_loader.BrixiaData(transform=test_aug, deterministic_transform=prepro, **param, cache={}, test=True)
# -

cache_folder = os.path.join('/work/projects/covid19_dv/models/brixia/cache')
train_data.preload(cache_folder)
val_data.preload(cache_folder)
test_data.preload(cache_folder)

computational_setup = dict(
    num_workers = 8,
    batch_size = 16
)
train_loader = torch.utils.data.DataLoader(train_data, **computational_setup)
valid_loader = torch.utils.data.DataLoader(val_data, **computational_setup)
valid_loader_tta = torch.utils.data.DataLoader(val_data_tta, **computational_setup)
test_loader = torch.utils.data.DataLoader(test_data, **computational_setup)

# ### Model setup

p.model_dict

import importlib
importlib.reload(model_setup)

# +
p.model_dict['pretrained'] = model_checkpoint
p.model_dict['fresh_head_weights'] = False

model = model_setup.instantiate_model(**p.model_dict)
model = model.to(device)

classp = torch.nn.Softmax(dim=-1)

is_global = p.data_setup['data'].get('global_score', False)

# +
base_loss = getattr(torch.nn, p.loss['class'])(**p.loss.get('param', {}), reduction='sum')
if is_global:
    crit = lambda x, y: (base_loss(x, y.long()), np.sum(y.shape))
else:
    crit = lambda x, y: (base_loss(torch.transpose(x, 1,2), y.long()), np.sum(y.shape))

print(f'Crit: {evaluations.eval_crit(model, train_loader, crit, device=device):.3f}')
print(f'Crit: {evaluations.eval_crit(model, valid_loader, crit, device=device):.3f}')
print(f'Crit: {evaluations.eval_crit(model, valid_loader_tta, crit, device=device):.3f}')
# -

pred_val, target_val = evaluations.batch_prediction(model, valid_loader, device=device)
pred_val_tta, target_val_tta = evaluations.batch_prediction(model, valid_loader_tta, tta_ensemble=10, device=device)
pred_train, target_train = evaluations.batch_prediction(model, train_loader, device=device)

# +
aggregate = 'expectation'

if aggregate == 'max':
    pred_val_bin = np.argmax(pred_val, -1)
    pred_train_bin = np.argmax(pred_train, -1)
    pred_valtta_bin = np.argmax(np.mean(pred_val_tta, 0), -1)
elif aggregate == 'expectation':
    pred_val_bin = (classp(torch.Tensor(pred_val)) * np.array([0,1,2,3])).sum(-1).numpy()
    pred_train_bin = (classp(torch.Tensor(pred_train)) * np.array([0,1,2,3])).sum(-1).numpy()
    pred_valtta_bin = (classp(torch.Tensor(np.mean(pred_val_tta, 0))) * np.array([0,1,2,3])).sum(-1).numpy()
elif aggregate == 'None':
    pred_val_bin = pred_val.squeeze()
    pred_train_bin = pred_train.squeeze()
    pred_valtta_bin = np.mean(pred_val_tta, 0).squeeze()
# -

if is_global:
    mae = np.mean(np.abs(np.round(pred_val_bin)-target_val.squeeze()), 0)
    print(f'Global MAE: {mae:0.2f} (vs. 1.83/1.73)')
else:
    mae = np.mean(np.abs(np.round(pred_val_bin)-target_val.squeeze()), 0)
    g_mae = np.mean(np.abs(pred_val_bin.sum(-1)-target_val.sum(-1)), 0)

    print(f'Global MAE: {g_mae:0.2f} (vs. 1.83/1.73)')
    print(f'Avg. MAE  : {np.mean(mae):0.2f} (vs. 0.47/0.44)\n')
    print(np.round(mae, 2).reshape(2,3).T)

if is_global:
    mae = np.mean(np.abs(np.round(pred_valtta_bin)-target_val.squeeze()), 0)
    print(f'Global MAE: {mae:0.2f} (vs. 1.83/1.73)')
else:
    mae = np.mean(np.abs(np.round(pred_valtta_bin)-target_val.squeeze()), 0)
    g_mae = np.mean(np.abs(pred_valtta_bin.sum(-1)-target_val.sum(-1)), 0)

    print(f'Global MAE: {g_mae:0.2f} (vs. 1.83/1.73)')
    print(f'Avg. MAE  : {np.mean(mae):0.2f} (vs. 0.47/0.44)\n')
    print(np.round(mae, 2).reshape(2,3).T)

if is_global:
    x = pred_valtta_bin
else:
    x = pred_valtta_bin.sum(-1)
o = sns.displot(x = x, y=target_val.sum(-1), bins=list(np.arange(-.5, 18)))
plt.plot([0,18], [0,18], ':', alpha=0.4)
_ = plt.xlabel('prediction')
_ = plt.ylabel('target')

if is_global:
    g_c = np.corrcoef(pred_val_bin, target_val.squeeze())[1,0]
    print(f'Global corr: {g_c:.2f} (vs. 0.85/0.86)')
else:

    c = []
    for i in range(6):
        c.append(np.corrcoef(pred_val_bin[:,i], target_val[:,i])[1,0])
    g_c = np.corrcoef(pred_val_bin.sum(-1), target_val.sum(-1))[1,0]

    print(f'Global corr: {g_c:.2f} (vs. 0.85/0.86)')
    print(f'Avg. corr:   {np.mean(c):.2f} (vs. 0.67/0.71) \n')
    print(np.round(c,2).reshape(2,3).T)

if is_global:
    g_c = np.corrcoef(pred_valtta_bin, target_val.squeeze())[1,0]
    print(f'Global corr: {g_c:.2f} (vs. 0.85/0.86)')
else:
    c = []
    for i in range(6):
        c.append(np.corrcoef(pred_valtta_bin[:,i], target_val_tta[:,i])[1,0])
    g_c = np.corrcoef(pred_valtta_bin.sum(-1), target_val_tta.sum(-1))[1,0]

    print(f'Global corr: {g_c:.2f} (vs. 0.85/0.86)')
    print(f'Avg. corr:   {np.mean(c):.2f} (vs. 0.67/0.71) \n')
    print(np.round(c,2).reshape(2,3).T)

# +
meta = val_data.meta_df.copy()


meta['score'] = pred_val_bin.sum(-1)
meta['target'] = target_val.sum(-1)

#meta['score'] = pred_val_bin
#meta['target'] = target_val

#meta['score'] = pred_val_bin[:,5]#.sum(-1)
#meta['target'] = target_val[:,5] #.sum(-1)

meta['error'] = meta.target - meta.score 
#meta['target2'] = target_val[:,1] 
#meta['diff'] = meta.target - meta.target2 

meta.View = meta.View.fillna('na')
meta.loc[meta.AcquisitionDeviceProcessingDescription == 'Thorax pa - HC', 'View'] = 'PA'

# +
#meta.groupby(['diff', 'View']).agg({'error':'mean'}).unstack('diff')

# +
#meta.groupby('Sex').agg({'Sex': 'count', 'score': 'mean', 'target':'mean', 'error':'mean'})

# +
#meta.groupby('View').agg({'View': 'count', 'score': 'mean', 'target':'mean', 'error':'mean'})

# +
#k, kv = 'Modality', ['CR', 'DX']
#k, kv = 'View', ['AP', 'PA'] #, 'na']
#k, kv = 'Sex', ['M', 'F']
k, kv = 'ManufacturerModelName', ['Fluorospot Compact FD', 'CR 75'] #, 'DRX-REVOLUTION'],

for v in kv:
    m = meta[k] == v
    sns.regplot(x = meta[m].score, y = meta[m].target, order=3, label=v)
#plt.ylim(-1,19)
#plt.xlim(-1,19)
plt.plot([0,18],[0,18], 'k:')
plt.legend()
plt.grid()
# -

# ### Paying Attention

# +
visualisation = {}

def hook_fn(m, i, o):
    visualisation[m._get_name()] = o 

model.global_pool.softmax.register_forward_hook(hook_fn)
model.global_pool.register_forward_hook(hook_fn)
# -

d = next(iter(valid_loader))
#d = next(iter(train_loader))

with torch.no_grad():
    logits = model(d[0].to(device))

att = visualisation['Softmax'].view(-1,6,12,12)
#b = visualisation['RegionalAttentionHead']

for j in range(15):

    plt.figure()
    plt.imshow(d[0][j,0], cmap='bone')
    plt.title(d[1][j])

    plt.figure(figsize=(20,5))
    for i in range(6):
        plt.subplot(1,6,i+1)
        plt.imshow(torch.nn.functional.interpolate(d[0], size=12)[j,0], cmap='gray', alpha=0.7)
        attention = att[j,i].detach().cpu().numpy()
        attention[attention<0.01] = np.nan
        plt.imshow(attention, vmin=0, vmax=1, alpha=0.9, cmap=plt.cm.spring_r)
        pred = classp(logits[j,i]).detach().cpu().numpy()
        plt.title([f'{i:.2f}' for i in pred])

pred_test, target_test = evaluations.batch_prediction(model, test_loader, device=device)

consensus_test = pd.read_csv('/work/projects/covid19_dv/raw_data/brixia/metadata_consensus_v1.csv').set_index('Filename')

consensus_test = consensus_test.loc[test_data.meta_df.Filename]

consensus_test.MeanGlobal.shape

pred_test_max = np.argmax(pred_test, -1)
pred_test_expec = (classp(torch.Tensor(pred_test)) * np.array([0,1,2,3])).sum(-1).numpy()

#sns.regplot(consensus_test.ModeGlobal, consensus_test.MeanGlobal)
sns.regplot(target_test.sum(-1), consensus_test.MeanGlobal, order=3)
sns.regplot(pred_test_max.sum(-1), consensus_test.MeanGlobal, order=3)
#sns.regplot(pred_test_expec.sum(-1), consensus_test.MeanGlobal, order=3)

np.abs(consensus_test.MeanGlobal - target_test.sum(-1)).mean()

np.abs(consensus_test.MeanGlobal - pred_test_max.sum(-1)).mean()

np.abs(consensus_test.MeanGlobal - pred_test_expec.sum(-1)).mean()

target_test
