import yaml
import pdb
import os

# Read yaml file for sims to analyze
with open('./sims_list.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# launch sims individually
for sim in cfg['sims']:
    simpath = cfg['path'] + '/' + sim
    print('python test.py {0}'.format(simpath))
    errorCode = os.system('python test.py {0}'.format(simpath))
    # errorCode = os.system('python main.py {0}'.format(simpath))
    if errorCode != 0:
        print('Error Code = {0}'.format(errorCode))
        raise Exception('Analysis crashed...')
