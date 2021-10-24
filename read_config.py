import numpy as np
import yaml

def get_boxsize(spath):
    # Read yaml config file to extract information about space
    # If confinement is True, use a cylindrical confinement in XY plane, but full box size in Z
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    return np.array( rconfig['simBoxHigh'])

def get_boxsize_confined(spath, conf='U'):

    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    box_size = np.array( rconfig['simBoxHigh'])
    if conf == 'U': # Unconfined
        return box_size
    elif conf == 'C': # Cylindrical (free dimension is 3rd)
        box_size[0] = 10*box_size[0]
        box_size[1] = 10*box_size[1]
    if conf == 'S': # Spherical
        box_size[0] = 10*box_size[0]
        box_size[1] = 10*box_size[1]
        box_size[2] = 10*box_size[2]
    if conf == 'P': # Planar (free dimension is 2nd, 3rd)
        box_size[2] = 10*box_size[2]

    return box_size

def get_timeSnap(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    return rconfig['timeSnap']

def get_timestep(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    return rconfig['dt']

def get_viscosity(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    return rconfig['viscosity']

def get_spring_constant(spath, idx=0):
    # Read yaml config file to extract information about crosslinker 
    with open( spath / 'ProteinConfig.yaml') as yf:
        pconfig = yaml.safe_load( yf)
    return pconfig['proteins'][idx]['kappa']

def get_rest_length(spath, idx=0):
    # Read yaml config file to extract information about crosslinker 
    with open( spath / 'ProteinConfig.yaml') as yf:
        pconfig = yaml.safe_load( yf)
    return pconfig['proteins'][idx]['freeLength']

def get_kT(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    return rconfig['KBT']

def get_diameter_sylinder(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    return rconfig['sylinderDiameter']
