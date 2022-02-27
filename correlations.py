import os 
from pathlib import Path
import pdb 
import tracemalloc
import gc
import numpy as np
import sys
from src.read_files import find_all_frames, read_dat_sylinder, read_dat_protein, count_sylinders
from src.read_config import get_config
from src.CalcOverlaps import minDistBetweenAllFilaments
from src.CalcOrderParametersLocal import calc_local_order_frame

def main( params):

    print('######### FILAMENT CORRELATIONS MODE #########')
    tracemalloc.start()
    gc.collect()
    
    sim_name = params['sim_name']
    simpath = params['sim_path']
    savepath = params['plot_path']
    if not Path.exists( savepath):
        os.mkdir( savepath)

    print('_'*50)
    print('\nSim Name : {0}'.format(sim_name))

    # Get files to load
    files_s, files_p, _ = find_all_frames(simpath)

    # number of frames
    nframe_ = len(files_s)

    # number of filaments
    nfil_ = count_sylinders(files_s[0])

    # Load config 
    cfg = get_config(simpath)

    # Initialize variables
    # Minimum Distance, Contact Number, Local Polar Order, Local Nematic Order
    MD = np.zeros((nfil_, nfil_, nframe_))
    CN = np.zeros((nfil_, nframe_))
    LPO = np.zeros((nfil_, nframe_))
    LNO = np.zeros((nfil_, nframe_))

    # Check if data exists . If yes, load and fill out arrays
    md_path = simpath / 'min_dist.npy'
    cn_path = simpath / 'contact_number.npy'
    lo_path = simpath / 'local_order.npy'

    # Check if data already exists. If yes, load it
    if Path.exists(md_path):
        with open(str(md_path), 'rb') as f:
            MD_pre = np.load(f)
        # Find how many frames processed, and fill out
        n_done_md = MD_pre.shape[-1]
        print('Min Dist data exists: {0}/{1} frames'.format(n_done_md, nframe_))
        MD[:,:,:n_done_md] = MD_pre
    else:
        n_done_md = 0 
        print('No MD data exists')
    if Path.exists(cn_path):
        with open(str(cn_path), 'rb') as f:
            CN_pre = np.load(f)
        # Find how many frames processed, and fill out
        n_done_cn = CN_pre.shape[-1]
        print('Contact Number data exists: {0}/{1} frames'.format(n_done_md, nframe_))
        CN[:,:n_done_md] = CN_pre
    else:
        n_done_cn = 0 
        print('No CN data exists')
    if Path.exists(lo_path):
        with open(str(lo_path), 'rb') as f:
            LPO_pre = np.load(f)
            LNO_pre = np.load(f)
        # Find how many frames processed, and fill out
        n_done_lo = LPO_pre.shape[-1]
        print('Local Order data exists: {0}/{1} frames'.format(n_done_md, nframe_))
        LPO[:,:n_done_md] = LPO_pre
        LNO[:,:n_done_md] = LNO_pre
    else:
        n_done_lo = 0 
        print('No LO data exists')
    n_done = min([n_done_lo, n_done_cn, n_done_md])

    # Loop over frames and process
    for jframe in range(nframe_):

        if jframe < n_done:
            continue
        
        print('Frame = {0}/{1}'.format(jframe, nframe_) )
        
        # Read sylinder data
        _,_,pos_minus_,pos_plus_, orientation_ = read_dat_sylinder(files_s[jframe])
    
        # Min Dist
        if jframe >= n_done_md:
            MD[:,:,jframe] = minDistBetweenAllFilaments( 
                pos_minus_, 
                pos_plus_, 
                pos_minus_, 
                pos_plus_, cfg['box_size']) / cfg['diameter_fil']

        # Contact Number 
        if jframe >= n_done_cn:
            CN[:,jframe] = np.sum( np.exp(-1*(MD[:,:,jframe]**2)), axis=1)

        # Local Order
        if jframe >= n_done_lo:
            LPO[:,jframe], LNO[:,jframe] = calc_local_order_frame( 
                    orientation_, 
                    MD[:,:,jframe])
    # Save data
    with open(str(md_path), 'wb') as f:
        np.save(f, MD)

    # Save data
    with open(str(cn_path), 'wb') as f:
        np.save(f, CN)

    # Save data
    with open(str(lo_path), 'wb') as f:
        np.save(f, LPO)
        np.save(f, LNO)
    
    gc.collect()
    print(tracemalloc.get_traced_memory())
    # stopping the library
    tracemalloc.stop()

if __name__ == '__main__':
    simpath = Path(sys.argv[1])
    params = {
            'sim_name': simpath.name,
            'sim_path': simpath,
            'plot_path': simpath / 'plots',
            }
    main(params)
