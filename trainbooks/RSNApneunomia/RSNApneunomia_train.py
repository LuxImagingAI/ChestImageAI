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

# +
import sys
sys.path.append('/home/users/jsoelter/Code/ChestImageAI/utils/')
sys.path.append('/home/users/jsoelter/Code/big_transfer/')

import data_loader, evaluations, model_setup, sacred
# -

# ### Parameter managment

# + tags=["parameters"]
defaults = {} # default parameter, will be overwritten if set explicilty in this notebooke
overwrites = {} # parameter that OVERWRITE all parameters set in this notebook. Usefull if Notebook is executed externally 
# -

p = sacred.ParameterStore(overwrites=overwrites)

# ### Model Setup

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
    'model_out': '/home/users/jsoelter/models/rsna/densenet/test3',
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

# ### Data Setup

# +
p.data_setup = dict(
    data =  dict(
        include_meta = [],
        #subset = {'Sex': ['M']}
        include_meta_features = [], #['Sex'],
        #include_meta = ['Sex', 'AP/PA', 'Frontal/Lateral'],
        val_conf = {
                'salt': '42',
                'fraction': 0.05,
            }
    ),
    transforms = [
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
        }),
])

p.sampling_config = dict(
    #meta_field = 'Sex',
    #meta_values = ['M', 'F'],
    #frac_meta0 = 0.5,
    #frac_meta0_tar1 = 0.3,
    #frac_meta1_tar1 = 0.3,
    #max_samples = 15000
)

sampling_config2 = p.sampling_config.copy()
sampling_config2['frac_meta0'] = 0.5
sampling_config2['frac_meta0_tar1'] = 0.2
sampling_config2['frac_meta1_tar1'] = 0.2

p.sampling_config2 = sampling_config2

# +
preprocess = data_loader.transform_pipeline_from_dict(p.data_setup['transforms'])

train_data = data_loader.RSNAPneumoniaData(
    transform=preprocess, 
    #sub_sampling=p.sampling_config, 
    validation=False, 
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data']
)

valid_data = data_loader.RSNAPneumoniaData(
    transform=preprocess,
    #sub_sampling=p.sampling_config,
    validation=True, 
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data'])

valid_data2 = data_loader.RSNAPneumoniaData(
    transform=preprocess,
    #sub_sampling=p.sampling_config2,
    validation=True, 
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data'])

# +
real_batch_size = 64 #
batch_split = 4 #number of forward pathes before optimization is performed 

train_loader = torch.utils.data.DataLoader(train_data, batch_size=int(real_batch_size/batch_split), num_workers=8, shuffle=True, drop_last=False)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=16, num_workers=8)
valid_loader2 = torch.utils.data.DataLoader(valid_data2, batch_size=16, num_workers=8)

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

p.scheduler = dict(
    supports = [int(0.5*steps_per_epoch), 1*steps_per_epoch, 2*steps_per_epoch, 3*steps_per_epoch, 4*steps_per_epoch]
)
#supports = [2*steps_per_epoch, 3*steps_per_epoch, 4*steps_per_epoch, 6*steps_per_epoch, 8*steps_per_epoch]#[3000, 7000, 9000, 10000]

# ### Loss

# +
p.loss_param = {
    #'pos_weight': 2.3
}

crit = model_setup.maskedBCE(**p.loss_param)
# -

# ### Initial errors

preds, targets = evaluations.batch_prediction(model, valid_loader, device=device)
print(f'AUC: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
print(f'Crit: {evaluations.eval_crit(model, valid_loader, crit, device=device):.3f}')

# ### Training Loop

eval_intervall = 20
save_intervall = 500#steps_per_epoch

# +
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
        y = y.to(device, non_blocking=True)
        if getattr(model, 'meta_injection', None):
            m = m.to(device, non_blocking=True)
            logits = model(x, m)
        else:
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
                preds, targets = evaluations.batch_prediction(model, valid_loader, device=device)
                auc_selection = evaluations.eval_auc(preds.reshape((-1,1)), targets)
                preds, targets = evaluations.batch_prediction(model, valid_loader2, device=device)
                auc_selection2 = evaluations.eval_auc(preds.reshape((-1,1)), targets)
                val = evaluations.eval_crit(model, valid_loader, crit, device=device)
                ledger['internal'].append((step-1, val))
                ledger['val_auc'].append((step-1, auc_selection, auc_selection2))

                print(f'step {step} ->, train: {train_loss:.3f},  auc: {auc_selection:.3f}, auc2: {auc_selection2:.3f}') # FULL: 

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




