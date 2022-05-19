from src.CalcOverlaps import minDistBetweenAllFilaments
from src.DataHandler import *
import numpy as np
from pathlib import Path

def CalcSaveMinDist(FData, simpath):

    # Define mindist file path
    md_path = simpath / 'min_dist.npy'

    # Check if data already exists. If yes, load it
    if Path.exists(md_path):
        with open(str(md_path), 'rb') as f:
            MD = np.load(f)
        
        # how many new frames to compute this for
        n_new = int(FData.nframe_ - MD.shape[-1])
        print('Computing minimum distances for {0} frames'.format(n_new))

        if n_new > 0:
            MD_new = np.zeros( (FData.nfil_, FData.nfil_, n_new) )
            MD = np.concatenate( (MD, MD_new), axis=-1)
        else:
            return MD
    else:
        # New frames
        n_new = int(FData.nframe_)
        MD = np.zeros((FData.nfil_, FData.nfil_, n_new))

    # Process unprocessed frames
    for cframe in range(FData.nframe_-n_new,FData.nframe_):

        print('Frame = {0}/{1}'.format(cframe,FData.nframe_), 
                end='\r', flush=True )
    
        # overlap matrix (normalized by filament diameter)
        MD[:,:,cframe] = minDistBetweenAllFilaments( 
                FData.pos_minus_[:,:,cframe], 
                FData.pos_plus_[:,:,cframe], 
                FData.pos_minus_[:,:,cframe], 
                FData.pos_plus_[:,:,cframe]) / FData.config_['diameter_fil']
    print('Frame = {0}/{0}'.format(FData.nframe_))

    # Save min dist data
    with open(str(md_path), 'wb') as f:
        np.save(f, MD)

    return MD
