import yaml
import pdb
import os

# Read yaml file for sims to analyze
with open('./sims_list.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

with open('./config.yaml', 'r') as f:
    cfg2 = yaml.safe_load(f)

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
correlation_mode = cfg2['CorrelationMode']

# Print sims names
print("-"*50)
print("-"*50)
print('\n')
print("   ###    ##       ######## ##    ##    ###    ##       ##    ##  ######  ####  ######")
print("  ## ##   ##       ##       ###   ##   ## ##   ##        ##  ##  ##    ##  ##  ##    ##")
print(" ##   ##  ##       ##       ####  ##  ##   ##  ##         ####   ##        ##  ##")
print("##     ## ##       ######   ## ## ## ##     ## ##          ##     ######   ##   ######")
print("######### ##       ##       ##  #### ######### ##          ##          ##  ##        ##")
print("##     ## ##       ##       ##   ### ##     ## ##          ##    ##    ##  ##  ##    ##")
print("##     ## ######## ######## ##    ## ##     ## ########    ##     ######  ####  ######")
print('\n')
print('Sims to analyze :')
for idx in range(len(sims)):
    print('\t{0}. {1}'.format(idx, sims[idx]))
print('\n')

# launch sims individually
for sim in sims:
    simpath = cfg['path'] + '/' + sim
    # errorCode = os.system('python test.py {0}'.format(simpath))
    # os.system('python main.py {0}'.format(simpath))
    if correlation_mode:
        errorCode = os.system('python correlations.py {0}'.format(simpath))
    else:
        errorCode = os.system('python main.py {0}'.format(simpath))
    if errorCode != 0:
        print('Error Code = {0}'.format(errorCode))
        raise Exception('Analysis crashed...')
