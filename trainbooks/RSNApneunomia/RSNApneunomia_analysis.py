# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import glob, os, json
import numpy
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
# -

# ## Gather Data
# Collect data produced by the `RSNApneunomia_eval` Notebook.

result_files = glob.glob('/home/users/jsoelter/models/rsna/bitm/new_exp/*/*.csv')

# +
results = []
for f in result_files:
    
    dat = pd.read_csv(f)
    ledger = json.load(open(os.path.join(os.path.dirname(f), 'train_ledger.json')))
    dat['model'] = ledger['train_setup']['0']['setup']['model_dict']['architecture']
    dat['salt'] = ledger['train_setup']['0']['setup']['data_setup']['data']['val_conf']['salt']
    results.append(dat)
    
results = pd.concat(results)
results = results.rename(columns={'name': 'dataset'})
results = results.drop(columns=['Unnamed: 0'])

results.head(2)
# -

# Results contain:
#   - `dataset` evaluation dataset (val1 internal, val2 external)
#   - `tta` test time augmentation id
#   - `auc` area under the curve (for `_m`: male, `_f`: female)
#   - `bce` binary cross entropy (for `_m`: male, `_f`: female)
#   - `max_samples` number of training samples
#   - `frac_meta0` fraction of male training samples
#   - `frac_meta0_tar1` fraction of target samples in male population 
#   - `frac_meta1_tar1` fraction of target samples in female population
#   - `model` name of trained model

# Add `frac_tar1`, the overal fraction of target values in the full population

results['frac_tar1'] = np.round(results.frac_meta0 * results.frac_meta0_tar1 + (1-results.frac_meta0 )* results.frac_meta1_tar1, 1)

# Get average auc across experiments

# +
aggregation = 'mean'
salt_values = ['0', '1', '2']

ix = ['dataset', 'frac_meta0', 'max_samples', 'model', 'frac_tar1', 'frac_meta0_tar1', 'frac_meta1_tar1']
auc = results[results.salt.isin(salt_values)].groupby(ix)[['auc', 'auc_m', 'auc_f']].agg([aggregation, 'std', 'count'])
bce = results[results.salt.isin(salt_values)].groupby(ix)[['bce', 'bce_m', 'bce_f']].agg([aggregation, 'std', 'count'])
# -

# experiments with same target fraction in both populations
mask_equal_frac = auc.index.get_level_values('frac_meta0_tar1') == auc.index.get_level_values('frac_meta1_tar1')


# +
what = 'auc_f'
model = 'BiT-M-R50x3'

tmp = results.set_index(ix).loc[('val2', 0, 7500, model, 0.3, 0.3)].sort_values(by='salt')
plt.scatter([0]*3, tmp[what].iloc[:3])

tmp = results.set_index(ix).loc[('val2', 0.5, 7500, model, 0.3, 0.3)].sort_values(by='salt')
plt.scatter([1]*3, tmp[what], alpha=0.5)

tmp = results.set_index(ix).loc[('val2', 0.5, 15000, model, 0.3, 0.3)].sort_values(by='salt')
plt.scatter([2]*3, tmp[what],  alpha=0.5)

plt.legend(loc='upper right', bbox_to_anchor=(1.6,1))
_ = plt.xticks([0,1,2], ['7.5k 100% female', '7.5k 50% female, 50% male', '15k  50% female, 50% male'], rotation=45, ha='right')
plt.grid()
plt.ylabel('AUC')

# +
model = 'BiT-M-R50x3'
what = ['auc', 'auc_m', 'auc_f']
#what = ['bce', 'bce_m', 'bce_f']
df = auc

plt.figure(figsize=(20, 12))

d = df[mask_equal_frac].loc[('val2', 0, 7500, model)]
for ix, c in enumerate(what):
    plt.subplot(3,2, 2*(ix+1))
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('frac_tar1'),
        y = dd[aggregation].values, 
        yerr = dd['std'].values, 
        #'o:',
        ls = '-',
        mfc='none', 
        label='7.5k female',
        alpha=0.5
    )
#plt.gca().set_prop_cycle(None)


d = df[mask_equal_frac].loc[('val2', 0.5, 7500, model)]
for ix, c in enumerate(what):
    plt.subplot(3,2, 2*(ix+1))
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('frac_tar1'),
        y = dd[aggregation].values, 
        yerr = dd['std'].values, 
        #'o:',
        ls = '-',
        mfc='none', 
        label='7.5k male + female',
        alpha=0.5
    )
    
d = df[mask_equal_frac].loc[('val2', 0.5, 15000, model)]
for ix, c in enumerate(what):
    plt.subplot(3,2, 2*(ix+1))
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('frac_tar1'),
        y = dd[aggregation].values, 
        yerr = dd['std'].values, 
        ls = '-',
        mfc='none', 
        label='15k male + female',
        alpha=0.5
    )    
    
    plt.legend()
    plt.ylim([0.85, 0.95])
    plt.yticks([0.85,0.9,0.95])
    plt.grid()
    plt.title(c)
# -
model2 = 'BiTX-M-R50x3'
#c = 'auc'
plt.figure(figsize=(20,5))
for i, c in enumerate(['auc', 'auc_m', 'auc_f']):
    plt.subplot(1,3,i+1)
    d = auc.loc[('val1', 0.5, 15000, model, 0.3)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label = 'confounded test data'
    )
    
    d = auc.loc[('val2', 0.5, 15000, model, 0.3)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label = 'external balanced test data'
    )
    
    only_female_model_score = auc.loc[('val2', 0, 7500, 'BiT-M-R50x3', 0.3), (c, aggregation)]
    plt.hlines(only_female_model_score, 0., 0.6, 'k', ls=':')
    
    # balanced_model_score1 = auc.loc[('val2', 0.5, 7500, 'BiT-M-R50x3', 0.3), (c, aggregation)]
    # balanced_model_score2 = auc.loc[('val2', 0.5, 15000, 'BiT-M-R50x3', 0.3, 0.3), (c, aggregation)]
    # plt.hlines(balanced_model_score1, 0., 0.6, 'c', ls='--', alpha=0.5)
    # plt.hlines(balanced_model_score2, 0., 0.6, 'm', ls='--', alpha=0.5)

    
    plt.title(['All', 'Subset Male', 'Subset Female'][i])
    plt.legend()
    plt.grid()
    plt.ylim([0.75, 0.97])
    plt.ylabel('AUC')
    plt.xlabel('Disease-Fraction Male')
# +
model2 = 'BiTX-M-R50x3'
#c = 'auc'
plt.figure(figsize=(20,5))
for i, c in enumerate(['auc', 'auc_m', 'auc_f']):
    plt.subplot(1,3,i+1)
    d = auc.loc[('val1', 0.5, 15000, model, 0.3)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label = 'confounded test data'
    )
    
    d = auc.loc[('val2', 0.5, 15000, model, 0.3)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label = 'external balanced test data'
    )
    plt.gca().set_prop_cycle(None)   

#c = 'auc'
for i, c in enumerate(['auc', 'auc_m', 'auc_f']):
    plt.subplot(1,3,i+1)
    d = auc.loc[('val1', 0.5, 15000, model2, 0.3)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label = 'meta injection',
        ls = ':',
    )
    
    d = auc.loc[('val2', 0.5, 15000, model2, 0.3)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label = 'meta injection',
        ls = ':',
    )
    plt.title(['All', 'Subset Male', 'Subset Female'][i])
    plt.legend(loc = 'lower left')
    plt.grid()
    plt.ylim([0.75, 0.97])
    plt.ylabel('AUC')
    plt.xlabel('Disease-Fraction Male')
# -


