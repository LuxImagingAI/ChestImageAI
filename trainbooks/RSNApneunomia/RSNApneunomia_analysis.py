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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import glob
import numpy
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
# -

result_files = glob.glob('/home/users/jsoelter/models/rsna/bitm/new_exp/*/*.csv')

# +
results = []
for f in result_files:
    dat = pd.read_csv(f)
    dat['mx'] = '_mf' in f 
    results.append(dat)
    
results = pd.concat(results)
# -

results['tar1'] = np.round(results.frac_meta0 * results.frac_meta0_tar1 + (1-results.frac_meta0 )* results.frac_meta1_tar1, 1)

ix = ['name', 'frac_meta0', 'max_samples', 'tar1', 'frac_meta0_tar1', 'frac_meta1_tar1', 'mx']

tmp = results.loc[results.name != 'train']
tmp = tmp.groupby(ix)['auc', 'auc_m', 'auc_f'].agg(['mean', 'std', 'count'])



# +
m1 = tmp.index.get_level_values('frac_meta0_tar1') == tmp.index.get_level_values('frac_meta1_tar1')
m2 = tmp.index.get_level_values('mx').values

m = m1 & np.logical_not(m2)


# +
plt.figure(figsize=(20,5))


d = tmp[m].loc[('val2', 0, 7500)]
for c in ['auc', 'auc_m', 'auc_f']:
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('tar1'),
        y = dd['mean'].values, 
        yerr = dd['std'].values, 
        #'o:',
        ls = ':',
        mfc='none', 
        label=c,
        alpha=0.5
    )
plt.gca().set_prop_cycle(None)


d = tmp[m].loc[('val2', 0.5, 7500)]
for c in ['auc', 'auc_m', 'auc_f']:
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('tar1'),
        y = dd['mean'].values, 
        yerr = dd['std'].values, 
        #'o:',
        ls = '-',
        mfc='none', 
        label=c,
        alpha=0.5
    )
#d = tmp[m].loc[(0.5, 7500)]
#_ = plt.plot(d.values, 'o-')
#_ = plt.xticks(range(8), d.index.get_level_values('tar1'))
plt.legend()
plt.grid()

plt.figure(figsize=(20,5))

m = m1 & np.logical_not(m2)

d = tmp[m].loc[('val1', 0, 7500)]
for c in ['auc', 'auc_m', 'auc_f']:
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('tar1'),
        y = dd['mean'].values, 
        yerr = dd['std'].values, 
        #'o:',
        ls = ':',
        mfc='none', 
        label=c,
        alpha=0.5
    )
plt.gca().set_prop_cycle(None)


d = tmp[m].loc[('val1', 0.5, 7500)]
for c in ['auc', 'auc_m', 'auc_f']:
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('tar1'),
        y = dd['mean'].values, 
        yerr = dd['std'].values, 
        #'o:',
        ls = '-',
        mfc='none', 
        label=c,
        alpha=0.5
    )
#d = tmp[m].loc[(0.5, 7500)]
#_ = plt.plot(d.values, 'o-')
#_ = plt.xticks(range(8), d.index.get_level_values('tar1'))
plt.legend()
plt.grid()
# -



# +
plt.figure(figsize=(20,5))

m = m1 & np.logical_not(m2)

d = tmp[m].loc[('val2', 0.5, 7500)]
for c in ['auc', 'auc_m', 'auc_f']:
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('tar1'),
        y = dd['mean'].values, 
        yerr = dd['std'].values, 
        #'o:',
        ls = ':',
        mfc='none', 
        label=c,
        alpha=0.5
    )
plt.gca().set_prop_cycle(None)


d = tmp[m].loc[('val2', 0.5, 15000)]
for c in ['auc', 'auc_m', 'auc_f']:
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('tar1'),
        y = dd['mean'].values, 
        yerr = dd['std'].values, 
        #'o:',
        ls = '-',
        mfc='none', 
        label=c,
        alpha=0.5
    )
plt.legend()


plt.figure(figsize=(20,5))

m = m1 & np.logical_not(m2)

d = tmp[m].loc[('val1', 0.5, 7500)]
for c in ['auc', 'auc_m', 'auc_f']:
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('tar1'),
        y = dd['mean'].values, 
        yerr = dd['std'].values, 
        #'o:',
        ls = ':',
        mfc='none', 
        label=c,
        alpha=0.5
    )
plt.gca().set_prop_cycle(None)


d = tmp[m].loc[('val1', 0.5, 15000)]
for c in ['auc', 'auc_m', 'auc_f']:
    dd = d[c]
    _ = plt.errorbar(
        x = dd.index.get_level_values('tar1'),
        y = dd['mean'].values, 
        yerr = dd['std'].values, 
        #'o:',
        ls = '-',
        mfc='none', 
        label=c,
        alpha=0.5
    )
plt.legend()

# +
plt.figure(figsize=(20,5))
plt.subplot(121)
for c in ['auc', 'auc_m', 'auc_f']:
    d = tmp.loc[('val2', 0.5, 15000, 0.2)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label=c
    )
plt.legend()

plt.subplot(122)
for c in ['auc', 'auc_m', 'auc_f']:
    d = tmp.loc[('val1', 0.5, 15000, 0.2)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label=c
    )
plt.legend()
# -

for c in ['auc', 'auc_m', 'auc_f']:
    d = tmp.loc[('val2', 0.5, 15000, 0.3)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label=c
    )
plt.legend()

# +
#c = 'auc'
plt.figure(figsize=(20,5))
for i, c in enumerate(['auc', 'auc_m', 'auc_f']):
    plt.subplot(1,3,i+1)
    d = tmp.loc[('val1', 0.5, 15000, 0.3)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label = 'confounded test data'
    )
    
    d = tmp.loc[('val2', 0.5, 15000, 0.3)]
    _ = plt.errorbar(
        x = d.index.get_level_values('frac_meta0_tar1'), 
        y = d[(c, 'mean')].values, 
        yerr = d[(c, 'std')].values,
        label = 'external balanced test data'
    )
    plt.title(['All', 'Subset Male', 'Subset Female'][i])
    plt.legend()
    plt.grid()
    plt.ylim([0.75, 0.97])
    plt.ylabel('AUC')
    plt.xlabel('Disease-Fraction Male')
    
    
# -


