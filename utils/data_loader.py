import os, glob, json
import hashlib
import collections

import cv2
import pydicom

import torch
import torchvision

import pandas as pd
import numpy as np

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
    
    
class RSNAPneumoniaData(Dataset):
    
    
    def __init__(self,
            meta_csv = 'stage_2_train_labels_extended.csv', # file to load
            datapath = '/work/projects/covid19_dv/rsna_pneunomia/',
            image_folder = 'stage_2_train_images',
            subset = {}, # Define subsetting of data
            sub_sampling = {},
            include_meta = [], # meta-data to include in targets
            include_meta_features  = [],
            transform = None,
            equalize = 'hist_cv',
            val_conf = {
                'salt': '42',
                'fraction': 0.03,
            },
            validation = True

        ):
        
        self.meta_fields = {'ViewPosition': 'AP/PA', 'PatientSex': 'Sex', 'PatientAge': 'Age'}
        self.meta_mapping = {
            'Sex': {'M': 0, 'F': 1},
            'AP/PA': {'AP': 0, 'PA': 1}
        }
        self.transform = transform
        self.equalize = equalize
        
        meta_df = pd.read_csv(os.path.join(datapath, meta_csv))
        self.boxes = meta_df.loc[meta_df.Target>0, ['patientId', 'x', 'y', 'width', 'height']].set_index('patientId')
        meta_df = meta_df.drop(columns=['x', 'y', 'width', 'height']).drop_duplicates()
        
        meta_df['Path'] = meta_df.patientId.apply(lambda x: os.path.join(datapath, image_folder, x+'.dcm'))
        
        # Subset the data
        m = meta_df.Path.notnull()
        for k,v in subset.items():
            m &= meta_df[k].isin(v)
        print(f'Removed {sum(np.logical_not(m))} entries')
        meta_df = meta_df[m]
        
        # Train-Validation Split
        if val_conf:
            meta_df['val'] = meta_df.patientId.apply(lambda x: hash_sample(val_conf['salt'] + x, val_conf['fraction']))
            meta_df = meta_df[meta_df.val] if validation else meta_df[np.logical_not(meta_df.val)]
        
        # Subsample the data
        if sub_sampling:
            meta_df = add_sample_factor(meta_df, **sub_sampling)
            meta_df['selection'] = meta_df.apply(lambda x: hash_sample('saltSampling' + x['patientId'], x['sampling_factor']), axis=1)
            meta_df = meta_df[meta_df.selection]
        
        for meta in self.meta_fields.values():
            if meta not in meta_df.columns:
                meta_df[meta] = -1
        for col, mapping in self.meta_mapping.items():
            meta_df[col] = meta_df[col].map(mapping)
                
        self.meta_df = meta_df.reset_index().drop(columns=['index'])
        self.include_meta = include_meta                    
        self.include_meta_features = include_meta_features
        
    
    def __getitem__(self, ix):
        
        dcm = pydicom.dcmread(self.meta_df.loc[ix].Path)
        
        labels = self.meta_df.loc[ix, ['Target'] + self.include_meta].to_list()
        labels = np.array(labels).astype('float32')
        
        if self.include_meta_features:
            meta_features = self.meta_df.loc[ix, self.include_meta_features].to_list()
            meta_features = np.array(meta_features).astype('float32')
        else:
            meta_features = np.array(np.nan).astype('float32')
            
        image = dcm.pixel_array
        if self.equalize == 'hist_cv':
            image = equalize_cv(image, dcm.BitsStored, dcm.PhotometricInterpretation)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels, meta_features
    
    
    def __len__(self):
        return len(self.meta_df)

    
# -------------------- utils -----------------------------------------------------------------------

def equalize(img, bit=16, photometric_interpretation=None):
    
    h, _ = np.histogram(img.flatten(), bins=2**bit)
    cs = np.cumsum(h)

    img_new = cs[img]
    img_new -= img_new.min()
    img_new = img_new / img_new.max()
    if photometric_interpretation == 'MONOCHROME1':
        img_new = 1 - img_new
    
    return img_new


def equalize_cv(img, bit= 16, photometric_interpretation=None):
    
    img_new = cv2.equalizeHist((255*(img/img.max())).astype('uint8'))
    if photometric_interpretation == 'MONOCHROME1':
        img_new = 255 - img_new
    
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


def transform_pipeline_from_dict(transform_list):
    """ creates transfrom pipeline from list
    
        Example:
          [
            ('RandomRotation', {
                    'degrees': 5
            }),
            ('RandomCrop', {
                    'size': (480,480)
            }),
            ('ToTensor', {})
          ]
    """    
    transform_objects = [create_transform(k, v) for k, v in transform_list]
    return torchvision.transforms.Compose(transform_objects)


def hash_sample(id_string, fraction=0.5, resolution=100):
    a = hashlib.md5(id_string.encode('utf-8'))
    b = a.hexdigest()
    return int(b, 16) % resolution < (fraction * resolution)

    
def add_sample_factor(meta_df, meta_field, meta_values, frac_meta0, frac_meta0_tar1, frac_meta1_tar1, max_samples=None):

    counts = meta_df.groupby([meta_field, 'Target']).Target.count().unstack('Target')
    total_samples = counts.sum().sum()
    actual_fractions = counts/total_samples

    ratio_meta = np.array([frac_meta0, 1-frac_meta0])
    ratios_target = np.array([[1-frac_meta0_tar1, frac_meta0_tar1], [1-frac_meta1_tar1, frac_meta1_tar1]])

    desired_fractions = pd.DataFrame((ratio_meta.reshape(-1,1) * ratios_target), index=meta_values, columns=[0,1])
    desired_fractions = desired_fractions.loc[counts.index, counts.columns]

    factors = actual_fractions/desired_fractions
    factors = factors.min().min() / factors

    num_samples = (counts * factors).sum().sum()
    print(f'{num_samples:.0f} samples could be drawn from {total_samples}', end='; ')

    if max_samples and (max_samples < num_samples):
        factors *= max_samples/num_samples
        num_samples = (counts * factors).sum().sum()

    print(f'about {num_samples:.0f} will be drawn')

    meta_df['sampling_factor'] = meta_df.apply(lambda x: factors.loc[x[meta_field], x['Target']], axis=1)
    
    return meta_df