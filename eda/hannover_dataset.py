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
import os, json, glob
import pandas as pd

import numpy as np
import pydicom
import scipy
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
# -

import sys
sys.path.append('/home/users/jsoelter/Code/ChestImageAI/utils/')
import data_loader, evaluations #model_setup

datafolder = '/work/projects/covid19_dv/raw_data/hanover_dataset/'

# ### Meta Information

meta_df = pd.read_csv(os.path.join(datafolder, 'data.csv')).sort_values(by=['patient_id', 'admission_offset'])
meta_df.head(2)

patients = meta_df.groupby('patient_id').agg({
    'image_id': 'count',
    'icu_admission_offset': lambda x: x.notnull().sum() > 0,
    'death_offset': lambda x: x.notnull().sum() > 0
})

patients.icu_admission_offset.mean()

patients.death_offset.mean()

meta_df[meta_df.patient_id == '48bbd9e5']

# +
import torch

from torch.utils.data import Dataset
from monai.transforms import LoadImage

import cv2

class HannoverMLData(Dataset):
    
    def __init__(self,
            meta_csv = '/work/projects/covid19_dv/raw_data/hanover_dataset/data.csv',
            image_path = '/work/projects/covid19_dv/raw_data/hanover_dataset/png/',
            transform = None,
            deterministic_transform = None
        ):
      
        self.transform = transform
        self.deterministic_transform = deterministic_transform
        
        meta_df = pd.read_csv(meta_csv)
        meta_df['Path'] = meta_df.image_id.apply(lambda x: os.path.join(image_path, x+'.png'))
        self.meta_df = meta_df
        
        #self.data_reader = LoadImage()
        

    def __getitem__(self, ix):
        
        row = self.meta_df.iloc[ix]
        image = cv2.imread(row.Path, 0)[np.newaxis] #read Grayscale to CWH
        data = {'image': image}
        
        if self.deterministic_transform:
            data = self.deterministic_transform(data)
        if self.transform:
            data = self.transform(data)
        
        return data
    
    
    def __len__(self):
        return len(self.meta_df)
# -

data = HannoverMLData()

plt.imshow(data[8]['image'][0])

# +
import timm
import monai

import sys
sys.path.append('/home/users/jsoelter/Code/ChestImageAI/utils/')
sys.path.append('/home/users/jsoelter/Code/big_transfer/')

import data_loader, evaluations, model_setup, sacred
import multi_head_modules as multihead
import train_utils

# +
#model_checkpoint = '/work/projects/covid19_dv/models/brixia/jan/local/att2/step03600.pt'
#model_checkpoint = '/work/projects/covid19_dv/models/brixia/jan/local/att7/step03000.pt'


model_checkpoint = '/work/projects/covid19_dv/models/brixblock/jan/test1/'
device = torch.device("cpu")

dirname = os.path.dirname(model_checkpoint)
ledger = json.load(open(os.path.join(dirname, 'train_ledger.json')))
# -

p = sacred.ParameterStore(defaults=ledger['train_setup']['0']['setup'])

# +
import copy

backbone = timm.create_model(**p.model['backbone'])
heads = copy.deepcopy(p.model['heads'])
for name, setup in heads.items():
    setup['features'] = multihead.FeatureExtractor(backbone, **setup['features'])
    setup['model'] = train_utils.instantiate_object(**setup['model'], encoder_channels = setup['features'].out_channels,)
# -

for x in ['cached', 'train_aug']:
    for k,v in p.transformations[x]:
        for k2, v2 in v.items():
            if k2 != 'keys' and k2 != 'interpolation' and (type(v2) == list):
                v[k2] = tuple(v2)

# Transformations with mask
cached_aug = train_utils.transform_pipe_factory(p.transformations, ['cached'], pop_key = ['mask'])
train_aug = train_utils.transform_pipe_factory(p.transformations, ['train_aug'], pop_key = ['mask'])

data = HannoverMLData(
    transform=train_aug, 
    deterministic_transform=cached_aug
)

plt.imshow(data[7]['image'][0])


def predictions(backbone, head, batch, device):

    with torch.no_grad():

        f = head['features']
        m = head['model']
        m.eval()
        backbone.eval()
        preds, targets = [], []
        x = torch.Tensor(batch[head['input']][np.newaxis])
        #y = batch[head['target']]
        _ = backbone(x)
        pred = m(list(f.get_features().values()))
        preds.append(pred.to('cpu').numpy())
        #targets.append(y)

    return np.vstack(preds) #, np.vstack(targets)


preds = []
for k in data.meta_df.index:
    try:
        pred = predictions(backbone, heads['BrixiaDataD'], data[k], device)
    except:
        pred = np.zeros((1,4,6))*np.nan
    preds.append(pred)
preds = np.vstack(preds)

plt.figure(figsize=(20,5))
plt.imshow(np.argmax(preds, -2).T, vmin=0, vmax=3, aspect='auto')

global_score = np.argmax(preds, -2).sum(1)

data.meta_df['global_score'] = global_score

data.meta_df.columns

data.meta_df.groupby('icu_admission_offset').global_score.mean().plot(marker = '.')

tmp = data.meta_df.copy()
tmp.death_offset = tmp.death_offset.fillna(-10)
s = tmp.groupby('death_offset').global_score.agg(['mean', 'count'])
plt.scatter(s.index, s['mean'], s=10*np.sqrt(s['count']))



tmp = data.meta_df.copy()
tmp.icu_admission_offset = tmp.icu_admission_offset.fillna(-80)
s = tmp.groupby('icu_admission_offset').global_score.agg(['mean', 'count'])
plt.scatter(s.index, s['mean'], s=10*np.sqrt(s['count']))
plt.grid()


