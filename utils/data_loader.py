import os, glob, json
import pandas as pd
import numpy as np
import collections
import cv2
import torch
import torchvision

from torch.utils.data import Dataset


class ChexpertData(Dataset):
    
    def __init__(self,
            meta_csv, # file to load
            datapath = '/work/projects/covid19_dv/heavy_datasets/chexpert_stanford/',
            subset = {}, # Define subsetting of data
            labels = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 
                      'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
                      'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'],
            include_meta = [], # meta-data to include in targets
            label_value_map = {
                'nan': 0.,
                -1: 0.5,
            },
            fill_hierachy = {}, 
            transform = None,
            equalize = 'hist'
        ):
                
        self.labels = labels
        self.transform = transform
        self.equalize = True
        
        meta_df = pd.read_csv(os.path.join(datapath, meta_csv))
        m = meta_df.Path.notnull()
        meta_df.Path = meta_df.Path.apply(lambda x: os.path.join(datapath, x))

        
        # Subset the data
        for k,v in subset.items():
            m &= meta_df[k].isin(v)
        print(f'Removed {sum(np.logical_not(m))} entries')
        meta_df = meta_df[m].reset_index().drop(columns='index')
        
        meta_df.Sex = meta_df.Sex.map({'Male': 0, 'Female': 1}).fillna(-1)
        meta_df['AP/PA'] = meta_df['AP/PA'].map({'AP': 0, 'PA': 1}).fillna(-1)
        meta_df['Frontal/Lateral'] =  meta_df['Frontal/Lateral'].map({'Frontal': 0, 'Lateral':1}).fillna(-1)
        
        # mapping of labels
        for l in labels:
            for k, v in label_value_map.items():
                if k != 'nan':
                    m = meta_df[l] == k
                    meta_df.loc[m, l] = v
            meta_df[l].fillna(label_value_map['nan'], inplace=True)
        
        # propagate hierachical labels
        for k, v_list in fill_hierachy.items():
            for v in v_list:
                less = meta_df[k] < meta_df[v]
                meta_df.loc[less, k] = meta_df.loc[less, v]
         
        self.meta_df = meta_df
        self.targets = labels + include_meta
                    
        
    def __getitem__(self, ix):
        
        labels = self.meta_df.iloc[ix][self.targets]
        image = cv2.imread(self.meta_df.iloc[ix].Path, cv2.IMREAD_GRAYSCALE)
        
        if self.equalize == 'hist':
            image = cv2.equalizeHist(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels.values.astype('double')
    
    def __len__(self):
        return len(self.meta_df)
    
    
class BIMCVCovid(Dataset):
    
    def __init__(self,
            meta_tsv,
            datapath = '/work/projects/covid19_dv/snapshots/bimcv_covid19',
            subset = {
                'modality': ['cr', 'dx'],
                'view': ['vp-pa', 'vp-pa'],
                "Patient's Sex": ['M', 'F']
                     },
            include_meta = [],
            label_smooth = 0,
            transform = None,
            labels = ['Covid']
        ):

        self.labels = labels
        self.transform = transform
        self.label_smooth = label_smooth
        
        meta_df = pd.read_csv(os.path.join(datapath, meta_tsv), delimiter='\t')
        labels = pd.read_csv(os.path.join(datapath, meta_tsv).replace('data', 'labels'), delimiter='\t').set_index(['PatientID', 'ReportID'])

        m = meta_df.path.notnull()
        # Subset the data
        for k,v in subset.items():
            m &= meta_df[k].isin(v)
        print(f'Removed {sum(np.logical_not(m))} entries')
        meta_df = meta_df[m].set_index(['PatientID', 'ReportID'])

        meta_df["Sex"] = meta_df["Patient's Sex"].map({'M': 0, 'F': 1})
        meta_df['AP/PA'] = meta_df['view'].map({'vp-ap': 0, 'vp-pa': 1})
        meta_df['manu'] = meta_df['Manufacturer'].map({'SIEMENS': 0, 'GE Healthcare':1})      
        
        meta_df.path = meta_df.path.apply(lambda x: os.path.join(datapath, x))

        t = labels[(labels.normal ==1) & (labels.n_labels==1) & (labels.exclude==0)].index
        data_normal = meta_df.loc[pd.Index(set(t).intersection(meta_df.index))]
        data_normal['Covid'] = 0

        t = labels[(labels['COVID 19'] ==1) & (labels.exclude==0)].index
        data_covid = meta_df.loc[pd.Index(set(t).intersection(meta_df.index))]
        data_covid['Covid'] = 1
        data_covid = data_covid[data_covid.n_images == 1]

        self.meta_df = pd.concat([data_normal, data_covid])
        self.targets = self.labels + include_meta
                    
        
    def __getitem__(self, ix):
        
        labels = self.meta_df.iloc[ix][self.targets]
        p = os.path.join(self.meta_df.iloc[ix].path, self.meta_df.iloc[ix].new_filename)
        
        #img_pre = Image.open(p)
        img_pre = cv2.imread(p, cv2.IMREAD_ANYDEPTH)
        img_pre = img_pre - img_pre.min()
        img_pre = img_pre/img_pre.max()
        if self.meta_df.iloc[ix]['PhotometricInterpretation'] == 'MONOCHROME1':
            img_pre = 1 - img_pre
        img_pre = (img_pre *  255).astype(np.uint8)
        img_pre = cv2.equalizeHist(img_pre)
        img_pre = cv2.cvtColor(img_pre, cv2.COLOR_GRAY2BGR)
        
        if self.transform:
            img_pre = self.transform(img_pre)
        
        labels = labels.values.astype('float')
        if self.label_smooth:
            labels = np.abs(labels - self.label_smooth)
            
        return img_pre, labels
    
    def __len__(self):
        return len(self.meta_df)
    
    
    
# -------------------- utils -----------------------------------------------------------------------


def equalize16bit(img):
    
    assert (min(img)>=0) & (max(img)<=1), 'Image should be between 0,1' 

    h, _ = np.histogram(img.flatten(), bins=np.linspace(0,1, 2**16))
    cs = np.cumsum(h)

    nj = (cs - cs.min()) * 2**16
    N = cs.max() - cs.min()

    # re-normalize the cdf
    cs = (nj / N).astype('uint16')

    img_new = cs[(f*2**16).astype('uint16')]
    
    return img_new


def create_transform(name, param_dict):
    ''' creates transform object from name and parameter dict 
    
        Example:
          name = 'Normalize'
          param_dict ={'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}        
    '''
    trans_class = getattr(torchvision.transforms, name)
    if param_dict:
        trans_func = trans_class(**param_dict)
    else:
        trans_func = trans_class()
    return trans_func


def transform_pipeline_from_dict(transform_dict):
    ''' creates transfrom pipeline from dict
    
        Example:
          {
            'RandomRotation': {
                    'degrees': 5
            },
            'RandomCrop': {
                    'size': (480,480)
            },
            'ToTensor': {}
          }
    '''    
    transform_objects = [create_transform(k, v) for k, v in transform_dict.items()]
    return torchvision.transforms.Compose(transform_objects)
