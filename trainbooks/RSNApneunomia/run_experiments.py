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

import pwd, os
import time
import collections
# -

#available_gpu = torch.cuda.device_count()
available_gpu = [0]#range(4)
processes = {k: None for k in available_gpu}

Experiment = collections.namedtuple('Experiment', ['frac_meta0', 'frac_meta0_tar1', 'frac_meta1_tar1', 'max_samples', 'name'])

# +
experiments = []

for iteration in range(0,5):
    for frac_meta1_tar1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        experiments.append(Experiment(
            frac_meta0=0, 
            frac_meta0_tar1=frac_meta1_tar1, 
            frac_meta1_tar1=frac_meta1_tar1, 
            max_samples=7500,
            name = f'pure_7500_None_{int(frac_meta1_tar1*10):d}_it{iteration:d}'
        ))

    for frac_meta1_tar1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        experiments.append(Experiment(
            frac_meta0=0.5, 
            frac_meta0_tar1=frac_meta1_tar1, 
            frac_meta1_tar1=frac_meta1_tar1, 
            max_samples=7500,
            name = f'mixed_7500_{int(frac_meta1_tar1*10):d}_{int(frac_meta1_tar1*10):d}_it{iteration:d}'
        ))

    for frac_meta1_tar1 in [0.1, 0.2, 0.3, 0.4, 0.5]:
        experiments.append(Experiment(
            frac_meta0=0.5, 
            frac_meta0_tar1=frac_meta1_tar1, 
            frac_meta1_tar1=frac_meta1_tar1, 
            max_samples=15000,
            name = f'mixed_15000_{int(round(frac_meta1_tar1*10)):d}_{int(round(frac_meta1_tar1*10)):d}_it{iteration:d}'
        ))

    for frac_meta1_tar1 in [0., 0.1, 0.3, 0.4]:
        experiments.append(Experiment(
            frac_meta0=0.5, 
            frac_meta0_tar1=0.4-frac_meta1_tar1, 
            frac_meta1_tar1=frac_meta1_tar1, 
            max_samples=15000,
            name = f'mixed_15000_{int(round((0.4-frac_meta1_tar1)*10)):d}_{int(round(frac_meta1_tar1*10)):d}_it{iteration:d}'
        ))

    for frac_meta1_tar1 in [0, 0.1, 0.2, 0.4, 0.5, 0.6]:
        experiments.append(Experiment(
            frac_meta0=0.5, 
            frac_meta0_tar1=0.6-frac_meta1_tar1, 
            frac_meta1_tar1=frac_meta1_tar1, 
            max_samples=15000,
            name = f'mixed_15000_{int(round((0.6-frac_meta1_tar1)*10)):d}_{int(round(frac_meta1_tar1*10)):d}_it{iteration:d}'
        ))
# -

train_notebook = './RSNApneunomia_train.ipynb'

out_folder = '/home/users/jsoelter/models/rsna/bitm/new_exp/'
for experiment in experiments:
    
    #gpu_id = fold%available_gpu
    proc_started = False
    param = {
    'overwrites': {
        'sampling_config': {
                'frac_meta0': experiment.frac_meta0,
                'frac_meta0_tar1': experiment.frac_meta0_tar1,
                'frac_meta1_tar1': experiment.frac_meta1_tar1,
                'max_samples': experiment.max_samples
        },
        'computation': {
            'model_out': os.path.join(out_folder, experiment.name),
            'device': None
        }
    }}
    output_path = param['overwrites']['computation']['model_out']
    output_file = os.path.join(output_path, 'runbook.ipynb')
    if os.path.exists(output_path): 
        print(f'skip {output_path}')
        continue
    else:
        os.makedirs(output_path)
    
    while not(proc_started):
        for k, v in processes.items():
            if not(v) or not(v.is_alive()):
                if v: v.close()
                param['overwrites']['computation']['device'] = f'cuda:{k}'
                print(f"\n Start {param['overwrites']['computation']['model_out']} on {param['overwrites']['computation']['device']}")
                p = mp.Process(target = pm.execute_notebook, args = (train_notebook, output_file, param))
                p.start()
                processes[k] = p
                proc_started = True
                break
        if not(proc_started):
            print('*', end='')
            time.sleep(5*60)


