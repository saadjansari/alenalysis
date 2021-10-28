import numpy as np
import yaml
import pdb

def get_config(spath):
    # Read and store useful configuration values from RunConfig and ProteinConfig.yaml
    print('Configuration :') 
    print('--------------\n') 
    config = {}
    config['geometry'] = get_confining_geometry(spath)
    config['box_size'] = get_boxsize_confined(spath, config['geometry']['type'])
    config['time_snap'] = get_timeSnap(spath)
    config['dt'] = get_timestep(spath)
    config['diameter_fil'] = get_diameter_sylinder(spath)
    config['visc'] = get_viscosity(spath)
    config['kT'] = get_kT(spath)
    config['kappa'] = get_spring_constant(spath)
    config['rest_length'] = get_rest_length(spath)
    print('--------------\n') 
    return config

def get_boxsize(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    return np.array( rconfig['simBoxHigh'])

def get_boxsize_confined(spath, conf='unconfined'):

    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    box_size = np.array( rconfig['simBoxHigh'])
    if conf == 'unconfined': # Unconfined
        return box_size
    elif conf == 'cylindrical': # Cylindrical (free dimension is 3rd)
        box_size[0] = 1000*box_size[0]
        box_size[1] = 1000*box_size[1]
    if conf == 'spherical': # Spherical
        box_size[0] = 1000*box_size[0]
        box_size[1] = 1000*box_size[1]
        box_size[2] = 1000*box_size[2]
    if conf == 'planar': # Planar (free dimension is 1st, 2nd)
        box_size[2] = 1000*box_size[2]
    print('box_size = {0}'.format(box_size) )
    return box_size

def get_timeSnap(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    timeSnap = rconfig['timeSnap']
    print('timeSnap = {0}'.format(timeSnap) )
    return timeSnap

def get_timestep(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    dt = rconfig['dt']
    print('dt = {0}'.format(dt) )
    return dt 

def get_viscosity(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    visc = rconfig['viscosity']
    print('viscosity = {0}'.format(visc) )
    return visc 

def get_spring_constant(spath, idx=0):
    # Read yaml config file to extract information about crosslinker 
    with open( spath / 'ProteinConfig.yaml') as yf:
        pconfig = yaml.safe_load( yf)
    kappa = np.zeros( len(pconfig['proteins']) )
    for idx in range(len(kappa)):
        kappa[idx] = pconfig['proteins'][idx]['kappa']
    print('kappa = {0}'.format(kappa) )
    return kappa 

def get_rest_length(spath, idx=0):
    # Read yaml config file to extract information about crosslinker 
    with open( spath / 'ProteinConfig.yaml') as yf:
        pconfig = yaml.safe_load( yf)
    rest_len = np.zeros( len(pconfig['proteins']) )
    for idx in range(len(rest_len)):
        rest_len[idx] = pconfig['proteins'][idx]['freeLength']
    print('freeLength = {0}'.format(rest_len) )
    return rest_len

def get_kT(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    kT = rconfig['KBT']
    print('kT = {0}'.format(kT) )
    return kT

def get_diameter_sylinder(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)
    d = rconfig['sylinderDiameter']
    print('sylinderDiameter = {0}'.format(d) )
    return d 

def get_confining_geometry(spath):
    # Read yaml config file to extract information about space
    with open( spath / 'RunConfig.yaml') as yf:
        rconfig = yaml.safe_load( yf)

    geometry = {}
    # boundaries
    if 'boundaries' not in rconfig.keys():
        geometry['type'] = 'unconfined'
        print('Geometry = {0}'.format(geometry['type']))
        return geometry

    bd = rconfig['boundaries']
    # Determine confinement type
    if len(bd) == 0:
        geometry['type'] = 'unconfined'
    elif len(bd) == 2:
        geometry['type'] = 'planar'
    elif len(bd) == 1:
        if bd[0]['type'] == 'tube':
            geometry['type'] = 'cylindrical'
        elif bd[0]['type'] == 'sphere':
            geometry['type'] = 'spherical'

    print('Geometry = {0}'.format(geometry['type']))

    # Find confinement parameters
    if geometry['type'] == 'spherical':
        geometry['center'] = bd[0]['center']
        geometry['radius'] = bd[0]['radius']
        print('  Center = {0}'.format(geometry['center']))
        print('  Radius = {0}'.format(geometry['radius']))
    elif geometry['type'] == 'cylindrical':
        print('  Assumption: Cylindrical confinement dimensions are X and Y')
        geometry['center'] = bd[0]['center']
        geometry['axis'] = bd[0]['axis']
        geometry['radius'] = bd[0]['radius']
        print('  Center = {0}'.format(geometry['center']))
        print('  Axis = {0}'.format(geometry['axis']))
        print('  Radius = {0}'.format(geometry['radius']))
    elif geometry['type'] == 'planar':
        print('  Assumption: Planar confinement dimension is Z')
        geometry['range'] = [ bd[0]['center'][-1] , bd[1]['center'][-1] ]
        print('  Range = {0}'.format(geometry['range']))
    return geometry
