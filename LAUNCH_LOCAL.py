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

            

# launch sims individually
for sim in sims:
    simpath = cfg['path'] + '/' + sim
    print('python test.py {0}'.format(simpath))
    errorCode = os.system('python test.py {0}'.format(simpath))
    # errorCode = os.system('python main.py {0}'.format(simpath))
    if errorCode != 0:
        print('Error Code = {0}'.format(errorCode))
        raise Exception('Analysis crashed...')
