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

result_files

# +
results = []
for f in result_files:
    dat = pd.read_csv(f)
    dat['mx'] = '_mf' in f 
    results.append(dat)
    
results = pd.concat(results)
# -

results['tar1'] = np.round(results.frac_meta0 * results.frac_meta0_tar1 + (1-results.frac_meta0 )* results.frac_meta1_tar1, 1)

ix = ['frac_meta0', 'max_samples', 'tar1', 'frac_meta0_tar1', 'frac_meta1_tar1', 'mx']

tmp = results.loc[results.name == 'val2']
tmp = tmp.groupby(ix)['auc', 'auc_m', 'auc_f'].mean()

m1 = tmp.index.get_level_values('frac_meta0_tar1') == tmp.index.get_level_values('frac_meta1_tar1')
m2 = tmp.index.get_level_values('mx').values

m = m1 & m2


tmp[m1].loc[0]

tmp[m1].loc[(0.5, 7500)]

# +
m = m1 & np.logical_not(m2)

for c in tmp.columns:
    d = tmp[m].loc[0]
    _ = plt.plot(d[c].values, 'o:', mfc='none', label=c)
plt.gca().set_prop_cycle(None)

d = tmp[m].loc[(0.5, 7500)]
_ = plt.plot(d.values, 'o-')
_ = plt.xticks(range(8), d.index.get_level_values('tar1'))
plt.legend()
# -

for c in tmp.columns:
    d = tmp[m].loc[(0.5, 15000)]
    _ = plt.plot(d[c].values, 'o-', label=c)
plt.gca().set_prop_cycle(None)
_ = plt.plot(tmp[m].loc[(0.5, 7500)].values, 'o-', mfc='none', ls=':')
_ = plt.xticks(range(3), d.index.get_level_values('tar1'))
plt.legend()



for c in tmp.columns:
    d = tmp.loc[(0.5, 15000, 0.2)]
    _ = plt.plot(d[c].values, 'o-', label=c)

d = tmp.loc[(0.5, 15000, 0.3)]
for c in tmp.columns:
    _ = plt.plot(d[c].values, 'o-', label=c)
_ = plt.xticks(range(len(d)), d.index.get_level_values('frac_meta0_tar1').values)
