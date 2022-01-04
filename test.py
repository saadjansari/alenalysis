import os 
from pathlib import Path
import pdb 
import h5py
import pickle
import tracemalloc
import gc
import numpy as np
from DataHandler import FilamentSeries, CrosslinkerSeries
from read_files import read_sim
from CalcNumXlinks import PlotStateCounts, PlotXlinkPerFilament, PlotXlinkPerFilamentVsTime, PlotXlinkPerFilamentVsTimeMax
from CalcOrderParameters import PlotNematicOrder, PlotPolarOrder, PlotNematicAndPolarOrder, PlotZOrder
from CalcOverlaps import PlotEffectiveDiameter
from CalcMSD import PlotMSD
from CalcDensityMaps import PlotFilamentDensityMovie, PlotFilamentDensityLastNframes
from local_order.CalcLocalMaps import PlotPackingFraction, PlotSOrder, PlotPOrder, PlotOrientation, PlotNematicDirector
from CalcCondensationFilaments import PlotFilamentCondensation, PlotFilamentClusters
from CalcCondensationXlinks import PlotXlinkClusters
from CalcMobility import PlotMobilityFilamentVsTime
from CalcOrderParametersLocal import *

attemptFastLoad = True
attemptFastSave = True

# Define sims to analyze
mainpath = Path('/Users/saadjansari/Documents/Projects/Results/SampleConf/Fig2')
sim_names = [
        # 'C3_PF16_X3_w1_p1_k1',
        'C1_PF8_X3_w1_p1_k1',
        # 'C2_PF8_X3_w1_p1_k1',
        # 'C3_PF8_X3_w1_p1_k1',
        # 'P1_PF8_X3_w1_p1_k1',
        # 'P2_PF8_X3_w1_p1_k1',
        # 'P3_PF8_X3_w1_p1_k1',
        # 'S1_PF8_X3_w1_p1_k1',
        # 'S2_PF8_X3_w1_p1_k1',
        # 'S3_PF8_X3_w1_p1_k1',
        ]

tracemalloc.start()

for sim_name in sim_names:
    gc.collect()

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
    FData.plot_trajectories( params['plot_path']/ 'trajectory_filament.pdf', alpha=0.3)
    XData.plot_trajectories( params['plot_path']/ 'trajectory_xlinker.pdf', alpha=0.3)

    # Analysis things
    # 1. Number of crosslinkers per filament
    # PlotStateCounts(FData, XData, params['plot_path']/ 'graph_xlinker_num_per_state.pdf')
    # PlotXlinkPerFilamentVsTime( FData, XData, params['plot_path']/ 'heatmap_xlinker_per_filament.pdf')
    # PlotXlinkPerFilament( FData, XData, params['plot_path']/ 'hist_xlinker_per_filament.pdf')
    # PlotXlinkPerFilamentVsTimeMax( FData, XData, params['plot_path']/ 'graph_xlinker_per_filament_max.pdf', 5)

    # 2. Crosslinker length/energy vs time
    # XData.plot_length_mean_vs_time( savepath / 'graph_xlinker_length_vs_time.pdf')
    # XData.plot_energy_mean_vs_time( savepath / 'graph_xlinker_energy_vs_time.pdf')
    
    # 3. Order parameters
    # PlotNematicOrder( FData, savepath/ 'graph_nematic_order.pdf')     
    # PlotPolarOrder( FData, savepath / 'graph_polar_order.pdf')     
    # PlotNematicAndPolarOrder( FData, savepath/'graph_nematic_polar_order.pdf' )
    # PlotZOrder( FData, savepath / 'graph_z_order.pdf')     

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
    PlotMSD(FData, savepath / 'graph_msd.pdf')

    # 7. Structure
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

    # displaying the memory
    print(tracemalloc.get_traced_memory())

    # Filament Condensation / Clustering
    # PlotFilamentCondensation(FData, XData, params)
    # PlotFilamentClusters(FData, params)

    # Crosslinker Clusters
    # PlotXlinkClusters(XData, params)

    # delete variables
    del FData
    del XData
    # close data stream
    data_filestream.close()
    # displaying the memory
    gc.collect()
    print(tracemalloc.get_traced_memory())

# stopping the library
tracemalloc.stop()
