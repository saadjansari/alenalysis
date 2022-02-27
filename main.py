import os 
from pathlib import Path
import pdb 
import h5py
# import pickle
import tracemalloc
import gc
import numpy as np
import sys
from src.DataHandler import FilamentSeries, CrosslinkerSeries
from src.read_files import read_sim
from src.CalcNumXlinks import PlotStateCounts, PlotXlinkPerFilament, PlotXlinkPerFilamentVsTime, PlotXlinkPerFilamentVsTimeMax
from src.CalcOrderParameters import PlotNematicOrder, PlotPolarOrder, PlotNematicAndPolarOrder, PlotZOrder
from src.CalcOverlaps import PlotEffectiveDiameter
from src.CalcMSD import PlotMSD
from src.CalcDensityMaps import PlotFilamentDensityMovie, PlotFilamentDensityLastNframes
from src.CalcLocalMaps import PlotPackingFraction, PlotSOrder, PlotPOrder, PlotOrientation, PlotNematicDirector, PlotNumberDensity, PlotNumberDensityXlink
from src.CalcCondensationFilaments import PlotFilamentCondensation, PlotFilamentClusters
from src.CalcCondensationXlinks import PlotXlinkClusters
from src.CalcMobility import PlotMobilityFilamentVsTime
from src.CalcOrderParametersLocal import *
from src.unpickler import renamed_load
from src.CalcPackingFraction import PlotLocalPackingFractionHistogram, PlotLocalPackingFractionVsTime
from src.CalcContactNumber import PlotContactNumberHistogram, PlotContactNumberVsTime, WriteContactNumber2vtk, CalcSaveContactNumber
from src.CalcCrosslinkerOrientations import PlotXlinkOrientations
from src.CalcRDF import PlotRDF, PlotRDF_PAP
from src.Write3DOrientation2vtk import Write3DOrientation2vtk
from src.CalcCrosslinkerPositionOnFilament import PlotCrosslinkerPositionOnFilament
from src.CalcMinDist import CalcSaveMinDist

def main( params):

    attemptFastLoad = True
    attemptFastSave = True
    tracemalloc.start()
    gc.collect()
    
    sim_name = params['sim_name']
    simpath = params['sim_path']
    savepath = params['plot_path']
    datapath = params['data_path']
    if not Path.exists( savepath):
        os.mkdir( savepath)
        
    # analysis data h5py
    if Path.exists(datapath):
        os.system('rm {0}'.format(str(datapath)) )
    data_filestream = h5py.File( datapath, "w")
    params['data_filestream'] = data_filestream

    print('_'*50)
    print('\nSim Name : {0}'.format(sim_name))
    
    # Read sim into FilamentData and CrosslinkerData {{{
    FData, XData = read_sim(simpath, sim_name[0])
    # fastpath_f = simpath / 'Fdata.pickle'
    # fastpath_x = simpath / 'Xdata.pickle'
    # if attemptFastLoad and Path.exists(fastpath_f) and Path.exists(fastpath_x):
        # with open(fastpath_f, "rb") as fp:
            # FData = pickle.load(fp)
        # with open(fastpath_x, "rb") as fp:
            # XData = pickle.load(fp)
    # else:
        # FData, XData = read_sim(simpath, sim_name[0])
        # if attemptFastSave:
            # with open(fastpath_f, "wb") as fp:
                # pickle.dump(FData, fp)
            # with open(fastpath_x, "wb") as fp:
                # pickle.dump(XData, fp)
    # }}}

    # Plot Trajectories
    # FData.plot_trajectories( params['plot_path']/ 'trajectory_filament.pdf', alpha=0.3)
    # XData.plot_trajectories( params['plot_path']/ 'trajectory_xlinker.pdf', alpha=0.3)

    # Analysis things
    # 1. Number of crosslinkers per filament
    # PlotStateCounts(FData, XData, params['plot_path']/ 'graph_xlinker_num_per_state.pdf')
    # PlotXlinkPerFilamentVsTime( FData, XData, params['plot_path']/ 'heatmap_xlinker_per_filament.pdf')
    # PlotXlinkPerFilament( FData, XData, params['plot_path']/ 'hist_xlinker_per_filament.pdf')
    # PlotXlinkPerFilamentVsTimeMax( FData, XData, params['plot_path']/ 'graph_xlinker_per_filament_max.pdf', 5)

    # # 2. Crosslinker length vs time
    # XData.plot_length_mean_vs_time( savepath / 'graph_xlinker_length_vs_time.pdf')
    # XData.plot_energy_mean_vs_time( savepath / 'graph_xlinker_energy_vs_time.pdf')
    
    # # 3. Order parameters
    # PlotNematicOrder( FData, savepath/ 'graph_nematic_order.pdf')     
    # PlotPolarOrder( FData, savepath / 'graph_polar_order.pdf')     
    # PlotNematicAndPolarOrder( FData, savepath/'graph_nematic_polar_order.pdf' )
    # PlotZOrder( FData, savepath / 'graph_z_order.pdf')     

    # # 4. Filament overlaps, and effective diameter
    # PlotEffectiveDiameter(FData, savepath / 'hist_effective_diameter.pdf')

    # 5. Local Order Parameters
    # Calculate Minimum Distance
    FData.MD = CalcSaveMinDist(FData, params['sim_path'])
    # Contact Number
    FData.CN = CalcSaveContactNumber(FData, params['sim_path'])
    # Calculate Local Order 
    FData.LPO, FData.LNO = CalcSaveLocalOrder(FData, params['sim_path'])

    # FData.CalcLocalStructure(N=50)
    # PlotContactNumberVsTime(FData, savepath / 'heatmap_contact_number_per_filament.pdf')     
    # PlotContactNumberHistogram(FData, savepath / 'hist_contact_number_per_filament.pdf')     
    # PlotLocalPolarOrderVsTime( FData, savepath / 'heatmap_local_polar_order.pdf')     
    # PlotLocalPolarOrderHistogram( FData, savepath / 'hist_local_polar_order.pdf')     
    # PlotLocalNematicOrderVsTime( FData, savepath / 'heatmap_local_nematic_order.pdf')     
    # PlotLocalNematicOrderHistogram( FData, savepath / 'hist_local_nematic_order.pdf')     
    # PlotLocalNematicOrderContactNumberHistogram( FData, savepath / 'hist2_local_nematic_order_contact_number.pdf')     
    # PlotLocalPolarOrderContactNumberHistogram( FData, savepath / 'hist2_local_polar_order_contact_number.pdf')     
    # PlotLocalNematicOrderLocalPolarOrderHistogram( FData, savepath / 'hist2_local_nematic_order_local_polar_order.pdf')     
    # WriteLocalOrder2vtk(FData, params)
    # WriteContactNumber2vtk(FData, params)
    # PlotLocalPackingFractionVsTime( FData, savepath / 'local_packing_fraction_img.pdf')     
    # PlotLocalPackingFractionHistogram( FData, savepath / 'local_packing_fraction_hist.pdf')     

    # 6. Dynamics
    # PlotMSD(FData, savepath / 'graph_msd.pdf')

    # # 7. Structure
    # PlotRDF(FData, params)
    # PlotRDF_PAP(FData, params)
    # PlotFilamentDensityLastNframes(FData, savepath / 'filament_density.pdf')
    # PlotFilamentDensityMovie(FData, savepath / 'density_movie', frame_gap=1)
    # PlotPackingFraction(FData, savepath / 'map_packing_fraction.pdf', N=20)
    # PlotNumberDensity(FData, savepath / 'map_number_density.pdf', N=20)
    # PlotNumberDensityXlink(FData, XData, savepath / 'map_number_density_xlink.pdf', N=20)
    # PlotSOrder(FData, savepath / 'map_nematic_order.pdf')
    # PlotPOrder(FData, savepath / 'map_polar_order.pdf')
    # PlotFlux(FData, savepath / 'map_flux.pdf')
    # PlotOrientation(FData, savepath / 'map_orientation.pdf')
    # PlotNematicDirector(FData, savepath / 'map_nematic_director.pdf')

    # Mobility
    # PlotMobilityFilamentVsTime(FData, savepath / 'mobility.pdf')

    # Crosslinker Distribution on MTs
    # PlotCrosslinkerPositionOnFilament(FData, XData, params)

    # Crosslinker Clusters
    # PlotXlinkClusters(XData, params)
    
    # Write Filament Orientation to VTK
    # Write3DOrientation2vtk(FData, XData, params)

    # Filament Condensation / Clustering
    # PlotFilamentCondensation(FData, XData, params, write2vtk=False)
    # PlotFilamentClusters(FData, params, N=400)

    # Crosslinker Filament orientation angle
    # PlotXlinkOrientations(FData, XData, savepath / 'hist_crosslinker_angles.pdf')

    # Close data file
    data_filestream.close()
    del FData
    del XData
    
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
            'data_path': simpath / 'data.hdf5',
            }
    main(params)
