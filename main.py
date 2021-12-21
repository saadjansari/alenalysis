import numpy as np
import os, pdb 
from pathlib import Path
import h5py
from DataHandler import FilamentSeries, CrosslinkerSeries
from read_config import *
from read_files import *
from CalcNumXlinks import *
from CalcOrderParameters import *
from CalcOrderParametersLocal import *
from CalcOverlaps import *
from CalcMSD import *
from local_order.CalcLocalMaps import *
from CalcCondensationFilaments import PlotFilamentCondensation, PlotFilamentClusters
from CalcCondensationXlinks import PlotXlinkClusters
from CalcMobility import *
import pickle

attemptFastLoad = False
attemptFastSave = False

# Define sims to analyze
mainpath = Path('/scratch/summit/saan8193/alens/conf_scan/sims')
sim_names = [
        'C1_PF4_X1_w1_p1_k1',
        'C2_PF4_X1_w1_p1_k1',
        'C3_PF4_X1_w1_p1_k1',
        'C4_PF4_X1_w1_p1_k1',
        'C1_PF8_X1_w1_p1_k1',
        'C2_PF8_X1_w1_p1_k1',
        'C3_PF8_X1_w1_p1_k1',
        'C4_PF8_X1_w1_p1_k1',
        'C1_PF16_X1_w1_p1_k1',
        'C2_PF16_X1_w1_p1_k1',
        'C3_PF16_X1_w1_p1_k1',
        'C4_PF16_X1_w1_p1_k1',
        'C1_PF4_X2_w1_p1_k1',
        'C2_PF4_X2_w1_p1_k1',
        'C3_PF4_X2_w1_p1_k1',
        'C4_PF4_X2_w1_p1_k1',
        'C1_PF8_X2_w1_p1_k1',
        'C2_PF8_X2_w1_p1_k1',
        'C3_PF8_X2_w1_p1_k1',
        'C4_PF8_X2_w1_p1_k1',
        'C1_PF16_X2_w1_p1_k1',
        'C2_PF16_X2_w1_p1_k1',
        'C3_PF16_X2_w1_p1_k1',
        'C4_PF16_X2_w1_p1_k1',
        'C1_PF4_X3_w1_p1_k1',
        'C2_PF4_X3_w1_p1_k1',
        'C3_PF4_X3_w1_p1_k1',
        'C4_PF4_X3_w1_p1_k1',
        'C1_PF8_X3_w1_p1_k1',
        'C2_PF8_X3_w1_p1_k1',
        'C3_PF8_X3_w1_p1_k1',
        'C4_PF8_X3_w1_p1_k1',
        'C1_PF16_X3_w1_p1_k1',
        'C2_PF16_X3_w1_p1_k1',
        'C3_PF16_X3_w1_p1_k1',
        'C4_PF16_X3_w1_p1_k1',
        'C1_PF4_X4_w1_p1_k1',
        'C2_PF4_X4_w1_p1_k1',
        'C3_PF4_X4_w1_p1_k1',
        'C4_PF4_X4_w1_p1_k1',
        'C1_PF8_X4_w1_p1_k1',
        'C2_PF8_X4_w1_p1_k1',
        'C3_PF8_X4_w1_p1_k1',
        'C4_PF8_X4_w1_p1_k1',
        'C1_PF16_X4_w1_p1_k1',
        'C2_PF16_X4_w1_p1_k1',
        'C3_PF16_X4_w1_p1_k1',
        'C4_PF16_X4_w1_p1_k1',
        ]

for sim_name in sim_names:
    simpath = mainpath / sim_name
    savepath = simpath / 'plots'
    if not Path.exists(savepath):
        os.mkdir(savepath)
        
    # analysis data h5py
    h5path = simpath / 'data.hdf5'
    if Path.exists(h5path):
        os.system('rm {0}'.format(str(h5path)) )
    data_filestream = h5py.File(h5path, "w")

    # paths
    params = {
            'sim_name': sim_name,
            'sim_path': simpath,
            'plot_path': savepath,
            'data_path': h5path,
            'data_filestream': data_filestream,
            }

    print('_'*50)
    print('\nSim Name : {0}'.format(sim_name))
    
    # Read sim into FilamentData and CrosslinkerData {{{
    fastpath = simpath / 'FXdata.pickle'
    if attemptFastLoad and Path.exists(fastpath):
        with open(fastpath, "rb") as fp:
            FData, XData = pickle.load(fp)
    else:
        FData, XData = read_sim(simpath, sim_name[0])
        if attemptFastSave:
            with open(fastpath, "wb") as fp:
                pickle.dump([FData, XData], fp)
    # }}}

    # Plot Trajectories
    # FData.plot_trajectories(savepath / 'traj_filament.pdf')
    # XData.plot_trajectories(savepath / 'traj_xlink.pdf')

    # Analysis things
    # 1. Number of crosslinkers per filament
    # PlotStateCounts(FData, XData, savepath / 'xlink_states.pdf')
    # PlotXlinkPerFilamentVsTime( FData, XData, savepath / 'xlinks_per_fil_img.pdf')
    # PlotXlinkPerFilament( FData, XData, savepath / 'xlinks_per_fil_hist.pdf')
    # PlotXlinkPerFilamentVsTimeMax( FData, XData, savepath / 'xlinks_max.pdf', 5)

    # # 2. Crosslinker length vs time
    # XData.plot_length_mean_vs_time( simpath / 'xlink_length_vs_time.pdf')
    # XData.plot_energy_mean_vs_time( simpath / 'xlink_energy_vs_time.pdf')
    
    # # 3. Order parameters
    # PlotNematicOrder( FData, savepath/ 'nematic_order.pdf')     
    # PlotPolarOrder( FData, savepath / 'polar_order.pdf')     
    # PlotNematicAndPolarOrder( FData, savepath / 'nematic_polar_order.pdf')     
    # PlotZOrder( FData, savepath / 'z_order.pdf')     

    # # 4. Filament overlaps, and effective diameter
    # PlotEffectiveDiameter(FData, savepath / 'effective_diameter.pdf')

    # 5. Local Order Parameters
    # FData.CalcLocalStructure()
    # PlotLocalPolarOrderVsTime( FData, savepath / 'local_polar_order_img.pdf')     
    # PlotLocalPolarOrderHistogram( FData, savepath / 'local_polar_order_hist.pdf')     
    # PlotLocalNematicOrderVsTime( FData, savepath / 'local_nematic_order_img.pdf')     
    # PlotLocalNematicOrderHistogram( FData, savepath / 'local_nematic_order_hist.pdf')     
    # PlotLocalPackingFractionVsTime( FData, savepath / 'local_packing_fraction_img.pdf')     
    # PlotLocalPackingFractionHistogram( FData, savepath / 'local_packing_fraction_hist.pdf')     

    # 6. Dynamics
    # PlotMSD(FData, savepath / 'msd.pdf')

    # # 7. Structure
    # PlotRDF(FData, savepath / 'rdf.pdf', rcutoff=1.0)
    # PlotRDF(FData, savepath / 'rdf_shortrange.pdf', rcutoff=0.1)
    # PlotFilamentDensityLastNframes(FData, savepath / 'filament_density.pdf')
    # PlotFilamentDensityMovie(FData, savepath / 'density_movie', frame_gap=1)
    # PlotPackingFraction(FData, savepath / 'map_packing_fraction.pdf')
    # PlotSOrder(FData, savepath / 'map_nematic_order.pdf')
    # PlotPOrder(FData, savepath / 'map_polar_order.pdf')
    # PlotFlux(FData, savepath / 'map_flux.pdf')
    # PlotOrientation(FData, savepath / 'map_orientation.pdf')
    # PlotNematicDirector(FData, savepath / 'map_nematic_director.pdf')

    # Mobility
    # PlotMobilityFilamentVsTime(FData, savepath / 'mobility.pdf')

    # Filament Condensation / Clustering
    PlotFilamentCondensation(FData, XData, params)
    PlotFilamentClusters(FData, params)

    # Crosslinker Clusters
    PlotXlinkClusters(XData, params)
