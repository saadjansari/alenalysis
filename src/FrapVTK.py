import numpy as np
import pdb
from src.write2vtk import add_array_to_vtk

def WriteFrapIntensity(FData, bleachFrame, params):

    # Calculate frap boolean array
    BleachedInt = CalcBleachedIntensity( FData.pos_minus_, FData.pos_plus_,
            bleachFrame)

    add_array_to_vtk(BleachedInt, 'Frap', params['sim_path'])


def CalcBleachedIntensity( pos0, pos1, bleachFrame):
    # Filaments with false value have been photobleached

    nFil = pos0.shape[1]
    nFrame = pos0.shape[-1]
    
    if bleachFrame > nFrame:
        raise Exception('Frap frame is greater than the total frames in the sim')
    
    arr = np.ones( (nFil, nFrame), dtype=int)

    # Figure out which filaments to photobleach
    p_minus = pos0[-1,:,bleachFrame]
    p_plus = pos1[-1,:,bleachFrame]

    # bleach filaments that pass through the mean Z plane
    bleachZ = (np.mean(p_minus) + np.mean(p_plus) )/2
    Fils2Bleach = np.where( ( (p_minus < bleachZ) & (p_plus > bleachZ) ) | 
            ( (p_minus > bleachZ) & (p_plus < bleachZ) ) )[0]

    nFilBleach = len(Fils2Bleach)
    print('Photobleaching {0} / {1} filaments'.format(nFilBleach, nFil))
    pdb.set_trace()

    arr[ Fils2Bleach, bleachFrame:] = False
    return arr

