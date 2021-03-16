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
import matplotlib
import collections
#import cv2
import tqdm
#from PIL import Image
import torch
#import torchvision.models

import sklearn
import sklearn.metrics
from scipy.special import expit

import scipy.special
import sklearn.metrics as skm

# +
import sys
sys.path.append('/home/users/jsoelter/Code/ChestImageAI/utils/')
sys.path.append('/home/users/jsoelter/Code/big_transfer/')

import data_loader, evaluations, model_setup, sacred
# -

# ### Model Setup

# + tags=["parameters"]
model_checkpoint = '/home/users/jsoelter/models/rsna/bitm/new_exp/mixed_15000_1_5_it1/step0046'
#model_checkpoint = '/home/users/jsoelter/models/rsna/bitm/new_exp/test2/step00464.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# +
dirname = os.path.dirname(model_checkpoint)
ledger = json.load(open(os.path.join(dirname, 'train_ledger.json')))

model_dict = ledger['train_setup']['0']['setup']['model_dict'].copy()
model_dict['pretrained'] = model_checkpoint
model_dict['fresh_head_weights'] = False
# -

model = model_setup.instantiate_model(**model_dict)
model = model.to(device)

# +
fig = plt.figure(figsize=(20,8)) 
gs = matplotlib.gridspec.GridSpec(3,1,height_ratios=(1,4,4), hspace=0)

ax = plt.subplot(gs[0])
plt.plot(ledger['lr'], 'k')
plt.xticks([])
plt.ylabel('lr')
plt.yscale('log')
#plt.xlim([0, 10000])

ax = plt.subplot(gs[1])
plt.plot(ledger['train_loss'], alpha=0.1) #, np.hstack([np.zeros(99), np.ones(100)/100]), mode = 'same'))
plt.plot(np.convolve(ledger['train_loss'], np.hstack([np.zeros(9), np.ones(10)/10]), mode = 'same'), color='b', label='train')
plt.plot(*np.array(ledger['internal']).T, '-', label='val')
#plt.yscale('log')
plt.legend()
plt.grid()
#plt.yscale('log')
plt.ylabel('cross entropy')
#plt.xticks([])
plt.ylim([0.1,0.8])
# -

# ### Data Setup

transforms = ledger['train_setup']['0']['setup']['data_setup']['transforms']
data_setup = ledger['train_setup']['0']['setup']['data_setup']['data']
sampling_config = ledger['train_setup']['0']['setup']['sampling_config']

sampling_config2 = {
    'meta_field': 'Sex',
    'meta_values': ['M', 'F'],
    'frac_meta0': 0.5,
    'frac_meta0_tar1': 0.3,
    'frac_meta1_tar1': 0.3,
}

# +
preprocess = data_loader.transform_pipeline_from_dict(transforms)

train_data = data_loader.RSNAPneumoniaData(transform=preprocess, sub_sampling=sampling_config, validation=False,  
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
     **data_setup)

valid_data = data_loader.RSNAPneumoniaData(transform=preprocess, sub_sampling=sampling_config, validation=True,
        datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/', **data_setup                                      
)

valid_data2 = data_loader.RSNAPneumoniaData(transform=preprocess, sub_sampling=sampling_config2, validation=True,
        datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/', **data_setup                                      
)
# -

computational_setup = dict(
    num_workers = 8,
    batch_size = 16
)
train_loader = torch.utils.data.DataLoader(train_data, **computational_setup)
valid_loader = torch.utils.data.DataLoader(valid_data, **computational_setup)
valid_loader2 = torch.utils.data.DataLoader(valid_data2, **computational_setup)


# ## Benchmark

def batch_prediction(model, loader, max_batch=-1, tta_ensemble = 1, device=None, overwrite=None):
    
    model.eval()
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble, targets = [], []
    for i in range(tta_ensemble):
        preds, targs = [], []
        with torch.no_grad():
            for i, (x, t, m) in enumerate(loader):
                x, t = x.to(device), t.numpy()
                if getattr(model, 'meta_injection', None):
                    if overwrite is not None:
                        m = torch.ones_like(m)*overwrite
                    m = m.to(device)
                    logits = model(x, m)
                else:
                    logits = model(x)
                preds.append(logits.to('cpu').numpy())
                targs.append(t)
                if i == max_batch:
                    break

        ensemble.append(np.vstack(preds))
        targets.append(np.vstack(targs))
    
    assert np.all(targets[0] == np.array(targets).mean(axis=0)), 'Targets across the ensemble do not match'
    
    return np.array(ensemble).squeeze(), targets[0]


def tta_predict(model, loader, data, tta_iter=1, overwrite=None):
    result_ens = []
    for i in range(tta_iter):
        preds, targets = batch_prediction(model, loader, device=device, overwrite=overwrite)
        results = data.meta_df.copy()
        results['tta'] = i
        results['p'] = scipy.special.expit(preds)
        result_ens.append(results)
    return pd.concat(result_ens)


# +
#results_train = tta_predict(model, train_loader, train_data, tta_iter=1)
# -

results_val = tta_predict(model, valid_loader, valid_data, tta_iter=1)
results_val2 = tta_predict(model, valid_loader2, valid_data2, tta_iter=1)

# +
#results_val2_negative = tta_predict(model, valid_loader2, valid_data2, tta_iter=1, overwrite=-1)
#results_val2_positive = tta_predict(model, valid_loader2, valid_data2, tta_iter=1, overwrite=1)
#results_val2_zero = tta_predict(model, valid_loader2, valid_data2, tta_iter=1, overwrite=0)
#results_val2_supermale = tta_predict(model, valid_loader2, valid_data2, tta_iter=1, overwrite=-200)
# -

if False:
    fig = plt.figure(figsize=(20,7))
    fig.suptitle('Trained on 0.1 male (-1) prevalence and 0.5 female (1) prevalence')

    plt.subplot(151)
    plt.grid()
    sns.boxenplot(data = results_val2, x='Target', y='p', hue='Sex')
    plt.title('true sex')

    plt.subplot(152)
    plt.grid()
    sns.boxenplot(data = results_val2_negative, x='Target', y='p', hue='Sex')
    plt.title('all male')

    plt.subplot(153)
    plt.grid()
    sns.boxenplot(data = results_val2_positive, x='Target', y='p', hue='Sex')
    plt.title('all female')

    plt.subplot(154)
    plt.grid()
    sns.boxenplot(data = results_val2_zero, x='Target', y='p', hue='Sex')
    plt.title('no sex (0)')

    plt.subplot(155)
    plt.grid()
    sns.boxenplot(data = results_val2_supermale, x='Target', y='p', hue='Sex')
    plt.title('super male (-200)')

#results_train_mean = results_train.groupby(['patientId', 'Modality', 'Sex', 'Age'])['p', 'Target'].mean().reset_index()
results_val_mean = results_val.groupby(['patientId', 'Modality', 'Sex', 'Age'])['p', 'Target'].mean().reset_index()
results_val2_mean = results_val2.groupby(['patientId', 'Modality', 'Sex', 'Age'])['p', 'Target'].mean().reset_index()

plt.figure(figsize=(20,7))
plt.subplot(131)
#sns.boxenplot(data = results_train_mean, x='Target', y='p', hue='Sex')
plt.subplot(132)
sns.boxenplot(data = results_val_mean, x='Target', y='p', hue='Sex')
plt.subplot(133)
sns.boxenplot(data = results_val2_mean, x='Target', y='p', hue='Sex')


# +
def score_functions(x, y):
    try : 
        tpr, fpr, _ = skm.roc_curve(y, x)
        auc = skm.auc(tpr, fpr)
    except ValueError:
        auc = np.nan
    try:
        bce = skm.log_loss(y, x)
    except ValueError:
        bce = np.nan
    return auc, bce

def get_scores(df, name=''):
    
    score_list = []
    for tta_ix, df in df.groupby('tta'):

        scores = {
            'name': name,
            'tta': tta_ix
        }

        m = df['Sex'] == -1
        f = df['Sex'] == 1
        
        scores['auc_m'], scores['bce_m'] = score_functions(df.p[m], df.Target[m])
        scores['auc_f'], scores['bce_f'] = score_functions(df.p[f], df.Target[f])
        scores['auc'], scores['bce'] = score_functions(df.p, df.Target)
        
        score_list.append(scores)

    return pd.DataFrame(score_list)


def plot_rocauc(df, scores):
    
        m = df['Sex'] == -1
        f = df['Sex'] == 1
        
        try:
            tpr, fpr, _ = skm.roc_curve(df.Target[m], df.p[m])
            auc = scores['auc_m'].mean()
            plt.plot(tpr, fpr, label=f'M AUC: {auc:.3f}')
        except ValueError:
            pass
        
        try:
            tpr, fpr, _ = skm.roc_curve(df.Target[f], df.p[f])
            auc = scores['auc_f'].mean()
            plt.plot(tpr, fpr, label=f'F AUC: {auc:.3f}')
        except ValueError:
            pass
        
        try:
            tpr, fpr, _ = skm.roc_curve(df.Target, df.p)
            auc = scores['auc'].mean()
            plt.plot(tpr, fpr, label=f'All AUC: {auc:.3f}')
        except ValueError:
            pass
        plt.legend(loc='lower right')


# +
evaluate = (
    #(results_train, 'train'),
    (results_val, 'val1'),
    (results_val2, 'val2'),
    #(results_val2_zero, 'val2_zero')
)

all_scores = []
fig = plt.figure(figsize=(20,5))
for ix, (df, name) in enumerate(evaluate):
    plt.subplot(1,len(evaluate),ix+1)
    scores = get_scores(df, name)
    all_scores.append(scores)
    plot_rocauc(df, scores)
    plt.title(name)
all_scores = pd.concat(all_scores)

fig.savefig(os.path.join(dirname, 'auc.png'))
# -

train_sampling = ledger['train_setup']['0']['setup']['sampling_config']

all_scores['max_samples'] = train_sampling['max_samples']
all_scores['frac_meta1_tar1'] = round(train_sampling['frac_meta1_tar1'],1)
all_scores['frac_meta0_tar1'] = round(train_sampling['frac_meta0_tar1'],1)
all_scores['frac_meta0'] = round(train_sampling['frac_meta0'],1)

all_scores

all_scores.to_csv(os.path.join(dirname, 'scores.csv'))

# +
fig = plt.figure(figsize=(20,5))

def rolling_mean_prob(x, y, nroll=50, label=''):
    plt.plot(x.rolling(nroll).mean(), y.rolling(nroll).mean(), label=label)


for ix, (df, name) in enumerate(evaluate):
    plt.subplot(1,3,ix+1)
    
    df = df.sort_values(by='p')
    
    m = df.Sex == -1
    rolling_mean_prob(df[m].p, df[m].Target, label='M')
    
    f = df.Sex == 1
    rolling_mean_prob(df[f].p, df[f].Target, label='F')

    rolling_mean_prob(df.p, df.Target, label='All', nroll=150)

    plt.ylabel('target mean')
    plt.xlabel('predicted probability mean')
    plt.plot([0,1], [0,1], 'k:')

    plt.legend()
    
fig.savefig(os.path.join(dirname, 'calibration.png'))
# -

if False:
    
    # historgram of errors
    plt.figure(figsize=(20,5))

    results = results.sort_values(by='p')

    diff = results.Target*np.log(results.p)+(1-results.Target)*np.log(1-results.p)


    plt.subplot(121)
    plt.hist([diff[results.Sex == 0], diff[results.Sex == 1]], label=['F', 'M'], bins=20, density=True, log=True)
    plt.legend()

    plt.subplot(122)
    plt.hist([diff[results['AP/PA'] == 'AP'], diff[results['AP/PA'] == 'PA']], label=['AP', 'PA'], bins=100, density=True)

    plt.legend()


# ## Explainability

# +
#import gradcam
#gradcam_model = gradcam.GradCAM(model, model._modules['head']._modules['relu'])
#gradcam_model = gradcam.GradCAM(model, model._modules['features']._modules['norm5'])
# -

def explain(image_ids):
    for i in image_ids:
        plt.figure(figsize=(20, 4.5))
        img, tar = valid_data[i]

        plt.subplot(131)
        plt.imshow(img[0], cmap=plt.cm.viridis)
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(img[0], cmap=plt.cm.bone)
        plt.axis('off')

        #labels = [external_valid_data.targets[q] for q in np.where(external_tar[i,:]>0.5)[0]]
        #plt.title('\n'.join(labels))

        plt.subplot(133)
        mask, _ = gradcam_model(img.unsqueeze(0).to(device), class_idx=0)
        heatmap, result = gradcam.utils.visualize_cam(mask, img[0])
        plt.imshow((result.T.numpy().swapaxes(0,1)+1)/2) #, cmap=plt.cm.bone)
        plt.axis('off')


def gradcam_analysis(valid_data, i):  
    plt.figure(figsize=(20, 4.5))
    img, tar, _ = valid_data[i]
    print(tar)
    
    plt.subplot(131)
    plt.imshow(img[0], cmap=plt.cm.viridis)
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(img[0], cmap=plt.cm.bone)
    plt.axis('off')

    #labels = [external_valid_data.targets[q] for q in np.where(external_tar[i,:]>0.5)[0]]
    #plt.title('\n'.join(labels))

    plt.subplot(133)
    mask, _ = gradcam_model(img.unsqueeze(0).to(device), class_idx=0)
    heatmap, result = gradcam.utils.visualize_cam(mask, img[0])
    plt.imshow((result.T.numpy().swapaxes(0,1)+1)/2) #, cmap=plt.cm.bone)
    plt.axis('off')

    return heatmap

# +
#ix = 11
#h = gradcam_analysis(valid_data, ix)
# -




