import yaml
import pdb
import os

# Read yaml file for sims to analyze
with open('./sims_list.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Time on summit
time = '01:00:00'

# check sims exist
for isim in cfg['sims']:
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
