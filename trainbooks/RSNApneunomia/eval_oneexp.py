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
import papermill as pm
import torch
import multiprocessing as mp
import nest_asyncio
nest_asyncio.apply()

import pwd, os, glob, shutil
import time
import collections

import numpy as np
# -

folder = glob.glob('/home/users/jsoelter/models/rsna/bitm/new_exp/*_it*')

# +
to_eval = []
last = []

for f in folder:
    cp = glob.glob(os.path.join(f, '*.pt'))
    if len(cp) == 0:
        print(f'Warning: {f}')
        #shutil.rmtree(f)
        continue
    cp = np.sort(cp)
    last_cp = cp[-1]
    previous_cp = cp[:-1]
    for old in previous_cp:
        print('remove', old)
        #os.remove(old)
    #print('keep', last_cp)
    
    last.append(int(last_cp.split('step')[1].split('.')[0]))
    to_eval.append(last_cp)
# -

eval_notebook = './RSNApneunomia_eval.ipynb'

#available_gpu = torch.cuda.device_count()
available_gpu = [0]#range(4)
processes = {k: None for k in available_gpu}

# +
redo = True

for cp in to_eval:
    
    param = {
        'model_checkpoint': cp,
        'device': 'cuda:0'

    }
    output_path = os.path.dirname(cp)
    output_file = os.path.join(output_path, 'evalbook.ipynb')
    scores = os.path.join(output_path, 'scores.csv')
    
    if os.path.exists(output_file) and not(redo):
        print(f'skipped {scores}')
    else:
        _ = pm.execute_notebook(eval_notebook, output_file, param)
# -


