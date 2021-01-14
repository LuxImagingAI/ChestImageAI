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
    architecture = 'BiTX-M-R50x3',
    #architecture = 'densenet121',
    num_classes = 1,
    #pretrained = 'imagenet', 
    pretrained = '/home/users/jsoelter/models/chexpert/fullmeta_503_consolidation_new/step00200.pt', #None, #'imagenet','imagenet', #
    fresh_head_weights = True,
    num_meta = 0
)

# +
p.computation = {
    'model_out': '/home/users/jsoelter/models/rsna/bitm/new_exp/test2',
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
        include_meta_features = ['Sex'],
        #include_meta = ['Sex', 'AP/PA', 'Frontal/Lateral'],
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
            #'mean': [0.485, 0.456, 0.406], 
            'mean': (0.5, 0.5, 0.5),
            #'std': [0.229, 0.224, 0.225]  
            'std': (0.5, 0.5, 0.5)
        }),
])

p.sampling_config = dict(
    meta_field = 'Sex',
    meta_values = ['M', 'F'],
    frac_meta0 = 0.5,
    frac_meta0_tar1 = 0.1,
    frac_meta1_tar1 = 0.5,
    max_samples = 15000
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
    sub_sampling=p.sampling_config, 
    validation=False,
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data']
)

valid_data = data_loader.RSNAPneumoniaData(
    transform=preprocess,
    sub_sampling=p.sampling_config,
    validation=True, 
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data'])

valid_data2 = data_loader.RSNAPneumoniaData(
    transform=preprocess,
    sub_sampling=p.sampling_config2,
    validation=True, 
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data'])

# +
real_batch_size = 128 #
batch_split = 8 #number of forward pathes before optimization is performed 

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

#optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
#optim = torch.optim.SGD([p for n,p in model.named_parameters() if 'head' in n], lr=0.003, momentum=0.9)
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

p.loss_param = {
    #'pos_weight': 2.3
}
crit = model_setup.maskedBCE(**p.loss_param)

# ### Initial errors

preds, targets = evaluations.batch_prediction(model, valid_loader, device=device)
print(f'AUC: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
print(f'Crit: {evaluations.eval_crit(model, valid_loader, crit, device=device):.3f}')

# ### Training Loop

eval_intervall = 10
save_intervall = steps_per_epoch

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
        print(f'{loss.data.cpu().numpy():.3f}')
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
                val = evaluations.eval_crit(model, valid_loader, crit, device=device)
                ledger['internal'].append((step-1, val))
                print(f'step {step} ->, train: {train_loss:.3f}, val: {val:.3f}, auc: {auc_selection:.3f}') # FULL: 

            if (step % save_intervall) == 0:
                torch.save({
                        "step": step,
                        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        "optim" : optim.state_dict(),
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

# +
import torch.nn as nn
from contextlib import nullcontext

class ConfounderPredictor(torch.nn.Module):
    
    def __init__(self, backbone, num_meta_data=0):
        super().__init__()

        self.backbone = backbone
        img_feat = self.backbone.body.block4.unit03.conv3.out_channels 
        self.classifier = nn.Conv2d(img_feat + num_meta_data, 1, kernel_size=(1, 1), stride=(1, 1))
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, im, meta_data=None, backbone_grad=False):
        
        # switch gradient calculation on/off
        gradient_context = nullcontext if backbone_grad else torch.no_grad 
        with gradient_context():
            feat = self.backbone(im) 
        pred = self.classifier(feat)
        assert pred.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return pred[...,0,0]


# +
import torch.nn as nn
from contextlib import nullcontext

class ConfounderPredictor(torch.nn.Module):
    
    def __init__(self, model, num_meta_data=0):
        super().__init__()

        self.model = model
        img_feat = self.backbone.body.block4.unit03.conv3.out_channels 
        self.classifier = nn.Conv2d(img_feat + num_meta_data, 1, kernel_size=(1, 1), stride=(1, 1))
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, im, meta_data=None, backbone_grad=False):
        
        # switch gradient calculation on/off
        gradient_context = nullcontext if backbone_grad else torch.no_grad 
        with gradient_context():
            feat = self.backbone(im)
            
        pred = self.classifier(feat)
        assert pred.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return pred[...,0,0]


# -

cp = ConfounderPredictor(backbone=model.backbone)
_ = cp.to(device)

# +
p.opt2 = {
    'class': 'SGD',
    'param': dict(
        lr = 2E-4,
        momentum=0.9,
        nesterov = True
    )
}

optim2 = getattr(torch.optim, p.opt2['class'])(cp.parameters(), **p.opt2['param'])

# +
p.data_setup['data2'] = {
    'include_meta': [], 
    'include_meta_features': ['Sex'],
    'subset': {
        'Target': [0]
    }
}

train_data_y0 = data_loader.RSNAPneumoniaData(
    transform=preprocess, 
    sub_sampling=p.sampling_config, 
    validation=False,
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data2']
)

valid_data_y0 = data_loader.RSNAPneumoniaData(
    transform=preprocess,
    sub_sampling=p.sampling_config,
    validation=True, 
    datapath = '/work/projects/covid19_dv/raw_data/rsna_pneunomia/',                                       
    **p.data_setup['data2'])



train_loader_y0 = torch.utils.data.DataLoader(train_data_y0, batch_size=int(real_batch_size/batch_split), num_workers=8, shuffle=True, drop_last=False)
valid_loader_y0 = torch.utils.data.DataLoader(valid_data_y0, batch_size=16, num_workers=8)
# -

if True:

    batch_loss = []
    _ = cp.train()

    for x, y, m in train_loader_y0:

        # Schedule sending to GPU(s)
        x = x.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        logits = cp(x)            
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, m)
        loss.backward()

        optim2.step()
        optim2.zero_grad()

        batch_loss.append(float(loss.data.cpu().numpy()))
    print(np.mean(batch_loss))


def batch_prediction(model, loader, max_batch=-1, tta_ensemble = 1, device=None):
    
    model.eval()
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ensemble, targets = [], []
    for i in range(tta_ensemble):
        preds, targs = [], []
        with torch.no_grad():
            for i, (x, t, m) in enumerate(loader):
                x, m = x.to(device), m.numpy()
                logits = model(x)
                preds.append(logits.to('cpu').numpy())
                targs.append(m)
                if i == max_batch:
                    break

        ensemble.append(np.vstack(preds))
        targets.append(np.vstack(targs))
    
    assert np.all(targets[0] == np.array(targets).mean(axis=0)), 'Targets across the ensemble do not match'
    
    return np.array(ensemble).squeeze(), targets[0]


preds, targets = batch_prediction(cp, valid_loader2, device=device)
print(f'AUC: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')

# +
p.opt_all = {
    'class': 'SGD',
    'param': dict(
        lr = 1E-4,
        momentum=0.9,
        nesterov = True
    )
}


p.opt_all1 = {
    'class': 'SGD',
    'param': dict(
        lr = 1E-5,
        momentum=0.5,
        nesterov = False
    )
}


optim_pred = getattr(torch.optim, p.opt_all['class'])(model.parameters(), **p.opt_all1['param'])
optim_conf = getattr(torch.optim, p.opt_all['class'])(cp.classifier.parameters(), **p.opt_all['param'])
optim_deconf = getattr(torch.optim, p.opt_all['class'])(cp.backbone.parameters(), **p.opt_all['param'])
# -

len(train_loader)

# +
predictor_loss, conf_loss, deconf_loss = [], [], [] 

update_conf, update_deconf = True, True

train_loader2 = torch.utils.data.DataLoader(
    train_data, 
    batch_size=8, 
    num_workers=8, 
    shuffle=True, 
    drop_last=False
)


# +
class EndlessIter():

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(loader)
        
    def __next__(self):
        try:
            out = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            out = next(self.iter)
        return out
    
dli = EndlessIter(train_loader2)
dlcp = EndlessIter(train_loader_y0)


# -

def train_step(model, dli, b_acc=1, alpha=0.5, device=device):
    
    optim_pred.zero_grad()
    for i in range(b_acc):
        
        x, y, m = next(dli)
        # Schedule sending to GPU(s)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_restriction = y == 0
        if y_restriction.sum() < 2:
            continue
        m = m.to(device, non_blocking=True)

        # update predictor
        logits_pred = model(x)
        loss_pred = torch.nn.functional.binary_cross_entropy_with_logits(logits_pred, y)
        
        try:
            logits_conf = cp(x[y_restriction.squeeze()], backbone_grad=True)            
            loss_conf = torch.nn.functional.binary_cross_entropy_with_logits(logits_conf.squeeze(), 1-m[y_restriction])
        except ValueError:
            print(y_restriction, y_restriction.sum())
            
        loss_full = loss_pred + alpha*loss_conf
        (loss_full/b_acc).backward()

    optim_pred.step()



def train_step2(cp, optim_conf, dlcp, device=device):
    
    x,y,m = next(dlcp)
    # Schedule sending to GPU(s)
    x = x.to(device, non_blocking=True)
    m = m.to(device, non_blocking=True)
    
    # update confounder prediction
    optim_conf.zero_grad()
    logits = cp(x)            
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, m)
    loss.backward()    
    optim_conf.step()
    conf_loss.append(float(loss.data.cpu().numpy()))
    


preds, targets = batch_prediction(cp, valid_loader2, device=device)    
print(f'AUC M vs F: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
preds, targets = evaluations.batch_prediction(model, valid_loader, device=device)
print(f'AUC: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
preds, targets = evaluations.batch_prediction(model, valid_loader2, device=device)
print(f'AUC ext.: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
print('===================')

# +
_ = cp.train()
_ = model.train()

b_acc = 4
alpha = 0.8

count = 0

while True:
    
    if count % 25 == 10:
        preds, targets = batch_prediction(cp, valid_loader2, device=device)    
        print(f'AUC M vs F: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
        preds, targets = evaluations.batch_prediction(model, valid_loader, device=device)
        print(f'AUC: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
        preds, targets = evaluations.batch_prediction(model, valid_loader2, device=device)
        print(f'AUC ext: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
        print('===================')
    count += 1

    for i in range(15):    
        train_step2(cp, optim_conf, dlcp)
    train_step(model, dli, b_acc = b_acc)
# -

torch.save({
        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
    }, 
    os.path.join(p.computation['model_out'], f'deconf.pt'))

1+1

# +
_ = cp.train()
_ = model.train()

for ix, (x, y, m) in enumerate(train_loader):
    if ix %10 == 0: print(ix, end=', ')
    if ix == 100: break
    
    # Schedule sending to GPU(s)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    y_restriction = y == 0
    m = m.to(device, non_blocking=True)
    
    # update predictor
    optim_pred.zero_grad()
    logits = model(x)
    
    #loss, n_samples = crit(logits, y)
    #loss /= n_samples
    #loss.backward()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()    
    
    optim_pred.step()
    predictor_loss.append(float(loss.data.cpu().numpy()))
    
    if update_conf:
        # update confounder prediction
        optim_conf.zero_grad()
        logits = cp(x[y_restriction.squeeze()])            
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), m[y_restriction])
        loss.backward()    
        optim_conf.step()
        conf_loss.append(float(loss.data.cpu().numpy()))
    
    if update_deconf:
        # decorrelate confounding    
        optim_deconf.zero_grad()
        logits = cp(x[y_restriction.squeeze()], backbone_grad=True)            
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), 1-m[y_restriction])
        loss.backward()
        optim_deconf.step()
        deconf_loss.append(float(loss.data.cpu().numpy()))
    

print()
    
preds, targets = batch_prediction(cp, valid_loader2, device=device)
print(f'AUC M vs F: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')

preds, targets = evaluations.batch_prediction(model, valid_loader, device=device)
print(f'AUC: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
# -

plt.figure(figsize=(20,5))
plt.plot(predictor_loss, '-', label='predictor')
plt.plot(conf_loss, '-', label='conf')
plt.plot(-1*np.array(deconf_loss), '-', label='deconf', )
plt.legend()
plt.yscale('log')



# +
preds, targets = batch_prediction(cp, valid_loader2, device=device)
print(f'AUC M vs F: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')

preds, targets = evaluations.batch_prediction(model, valid_loader, device=device)
print(f'AUC: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
# -

preds, targets = evaluations.batch_prediction(model, valid_loader, device=device)
print(f'AUC: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
print(f'Crit: {evaluations.eval_crit(model, valid_loader, crit, device=device):.3f}')

preds, targets = evaluations.batch_prediction(model, train_loader, device=device)
print(f'AUC: {evaluations.eval_auc(preds.reshape(-1, 1), targets):.3f}')
print(f'Crit: {evaluations.eval_crit(model, valid_loader, crit, device=device):.3f}')

plt.plot(np.convolve(predictor_loss, np.ones(20)/20, 'valid'))



plt.plot(np.convolve(deconf_loss, np.ones(20)/20, 'valid'))

plt.plot(np.convolve(conf_loss, np.ones(20)/20, 'valid'))



loss =  torch.nn.functional.binary_cross_entropy_with_logits(logits, m)

loss.backward()

optim2.step()

logits


