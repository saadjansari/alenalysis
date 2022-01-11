import yaml
import pdb
import os

# Read yaml file for sims to analyze
with open('./sims_list.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# get sim_names if required
if not cfg['custom_sims']:
    sims = []
    for jconf in cfg['Confinement']:
        for jpf in cfg['PackingFraction']:
            for jx in cfg['NumXlinks']:
                for kval in cfg['k_val']: 
                    for wp in cfg['walk_pause']:
                        sims.append('{0}_PF{1}_X{2}_{3}_k{4}'.format(jconf, jpf, jx, wp, kval) )
else:
    sims = cfg['sims']
print(sims)

# Time on summit
time = '01:00:00'

# check sims exist
for isim in sims:
    simpath = cfg['path'] + '/' +  isim

    # JobString
    jobStringDef = """#!/bin/bash

#SBATCH --job-name=alenalysis
#SBATCH --qos=condo
#SBATCH --partition=shas
#SBATCH --account=ucb-summit-smr

#SBATCH -o {0}/alenalysis.out  # send stdout to outfile
#SBATCH -e {0}/alenalysis.err # send stderr to errfile
#SBATCH --time={1}

#SBATCH --nodes=1      # nodes requested
#SBATCH --cpus-per-task=1

source /home/${USER}/.bashrc
#module load python
conda activate pysos
python --version

export PYTHONUNBUFFERED=1
python main.py {0}
"""

    jobString = jobStringDef.format( simpath, time)
    with open('jobscript.sh', 'w') as f:
        f.write( jobString)

    # Launch sbatch
    os.system('sbatch jobscript.sh')
