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

folder = glob.glob('/home/users/jsoelter/models/rsna/bitm/new_exp/*')

# +
to_eval = []

for f in folder:
    cp = glob.glob(os.path.join(f, '*.pt'))
    if len(cp) == 0:
        print(f'Warning: {f}')
        continue
    cp = np.sort(cp)
    last_cp = cp[-1]
    previous_cp = cp[:-1]
    for old in previous_cp:
        print('remove', old)
        #os.remove(old)
    print('keep', last_cp)
    to_eval.append(last_cp)
# -

eval_notebook = './RSNApneunomia_eval.ipynb'

#available_gpu = torch.cuda.device_count()
available_gpu = [0]#range(4)
processes = {k: None for k in available_gpu}

for cp in to_eval:
    
    #gpu_id = fold%available_gpu
    proc_started = False
    param = {
        'model_checkpoint': cp
    }
    output_path = os.path.dirname(cp)
    output_file = os.path.join(output_path, 'evalbook.ipynb')
    if os.path.exists(output_file):
        print(f'skipped {output_path}')
        continue
        
    while not(proc_started):
        for k, v in processes.items():
            if not(v) or not(v.is_alive()):
                if v: v.close()
                param['device'] = f'cuda:{k}'
                print(f"\n Start {param['model_checkpoint']} on {param['device']}")
                p = mp.Process(target = pm.execute_notebook, args = (eval_notebook, output_file, param))
                p.start()
                processes[k] = p
                proc_started = True
                break
        if not(proc_started):
            print('*', end='')
            time.sleep(60)



1+1


