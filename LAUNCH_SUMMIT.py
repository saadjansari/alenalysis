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

# Filament correlations mode
correlation_mode = cfg['filament_correlations_mode']
if correlation_mode:
    filename = 'correlations.py'
else:
    filename = 'main.py'

print(sims)

# Time on summit
time = '05:00:00'

# check sims exist
for isim in sims:
    simpath = cfg['path'] + '/' +  isim

    # JobString
    jobStringDef = """#!/bin/bash

#SBATCH --job-name=alenalysis
#SBATCH --qos=condo
#SBATCH --partition=shas
#SBATCH --account=ucb-summit-smr
#SBATCH --mem=14G

#SBATCH -o {0}/alenalysis.out  # send stdout to outfile
#SBATCH -e {0}/alenalysis.err # send stderr to errfile
#SBATCH --time={1}

#SBATCH --nodes=1      # nodes requested
#SBATCH --cpus-per-task=3

source /home/${2}/.bashrc
#module load python
conda activate pysos
python --version

export PYTHONUNBUFFERED=1
python {3} {0}
"""

    jobString = jobStringDef.format( simpath, time, 'USER', filename)
    with open('jobscript.sh', 'w') as f:
        f.write( jobString)

    # Launch sbatch
    os.system('sbatch jobscript.sh')
