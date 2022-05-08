import os 
from pathlib import Path
import pdb 
import h5py
# import pickle
import tracemalloc
import gc
import numpy as np
import sys
import yaml
from src.DataHandler import FilamentSeries, CrosslinkerSeries, AddCorrelationsToDataSeries
from src.read_files import read_sim
from src.CalcNumXlinks import PlotStateCounts, PlotXlinkPerFilament, PlotXlinkPerFilamentVsTime, PlotXlinkPerFilamentVsTimeMax
from src.CalcOrderParameters import PlotNematicOrder, PlotPolarOrder, PlotNematicAndPolarOrder, PlotZOrder
from src.CalcOverlaps import PlotEffectiveDiameter
from src.CalcMSD import PlotMSD
from src.CalcDensityMaps import PlotFilamentDensityMovie, PlotFilamentDensityLastNframes
from src.CalcLocalMaps import PlotPackingFraction, PlotSOrder, PlotPOrder, PlotOrientation, PlotNematicDirector, PlotNumberDensity, PlotNumberDensityXlink
from src.CalcCondensationFilaments import PlotFilamentCondensation, PlotFilamentClusters
from src.CalcCondensationXlinks import PlotXlinkClusters
from src.CalcMobility import PlotMobilityFilamentVsTime, PlotMobilityCrosslinkerHist
from src.CalcOrderParametersLocal import *
from src.unpickler import renamed_load
from src.CalcPackingFraction import PlotLocalPackingFractionHistogram, PlotLocalPackingFractionVsTime
from src.CalcContactNumber import PlotContactNumberHistogram, PlotContactNumberVsTime, WriteContactNumber2vtk
from src.CalcCrosslinkerOrientations import PlotXlinkOrientations
from src.CalcRDF import PlotRDF, PlotRDF_PAP
from src.Write3DOrientation2vtk import Write3DOrientation2vtk
from src.CalcCrosslinkerPositionOnFilament import PlotCrosslinkerPositionOnFilament, PlotCrosslinkerLength
from src.CalcExtensileMotors import PlotExtensileFilamentPairs
from src.CalcCrosslinkerDwellTime import PlotCrosslinkerDwellTime

def main( params):

    # Preamble {{{
    tracemalloc.start()
    gc.collect()
    
    simname = params['sim_name']
    simpath = params['sim_path']
    savepath = params['plot_path']
    datapath = params['data_path']
    if not Path.exists( savepath):
        os.mkdir( savepath)
        
    # analysis data h5py
    if Path.exists(datapath):
        os.system('rm {0}'.format(str(datapath)) )
    data_filestream = h5py.File( datapath, "a")
    params['data_filestream'] = data_filestream

    print('_'*50)
    print('\nSim Name : {0}'.format(simname))

    # Read configuration  yaml file 
    with open('./config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    params['cfg'] = cfg
    # }}}
    
    # Read Sim
    FData, XData = read_sim(simpath, simname[0])

    # Analysis

    # General {{{

    # Plot Trajectories
    if cfg['PlotTrajectories']:
        FData.plot_trajectories( params['plot_path']/ 'trajectory_filament.pdf', alpha=0.3)
        XData.plot_trajectories( params['plot_path']/ 'trajectory_xlinker.pdf', alpha=0.3)

    # Crosslinker Stats 
    if cfg['PlotCrosslinkerStats']:
        PlotStateCounts(FData, XData, params['plot_path']/ 'graph_xlinker_num_per_state.pdf')
        PlotXlinkPerFilamentVsTime( FData, XData, params['plot_path']/ 'heatmap_xlinker_per_filament.pdf')
        PlotXlinkPerFilament( FData, XData, params['plot_path']/ 'hist_xlinker_per_filament.pdf')
        PlotXlinkPerFilamentVsTimeMax( FData, XData, params['plot_path']/ 'graph_xlinker_per_filament_max.pdf', 5)
        PlotCrosslinkerPositionOnFilament(FData, XData, params)
        PlotXlinkOrientations(FData, XData, savepath / 'hist_crosslinker_angles.pdf')

    # Crosslinker Energy 
    if cfg['PlotCrosslinkerLength']:
        # XData.plot_length_mean_vs_time( savepath / 'graph_xlinker_length_vs_time.pdf')
        # XData.plot_energy_mean_vs_time( savepath / 'graph_xlinker_energy_vs_time.pdf')
        PlotCrosslinkerLength(XData, params, savepath / 'graph_xlinker_length_vs_time.pdf')

    
    # Order parameters
    if cfg['PlotGlobalOrderParameters']:
        PlotNematicOrder( FData, params, savepath/ 'graph_nematic_order.pdf')     
        PlotPolarOrder( FData, params, savepath / 'graph_polar_order.pdf')     

    # Fraction Extensile Motors
    if cfg['PlotExtensileMotors']:
        PlotExtensileFilamentPairs(FData, XData, params, savepath / 'graph_extensile_motors.pdf')
    # }}}

    # Correlations {{{
    if cfg['PlotCorrelations']:
        FData = AddCorrelationsToDataSeries(FData, simpath)
        PlotContactNumberVsTime(FData, savepath / 'heatmap_contact_number_per_filament.pdf')     
        PlotContactNumberHistogram(FData, savepath / 'hist_contact_number_per_filament.pdf')     
        PlotLocalPolarOrderVsTime( FData, savepath / 'heatmap_local_polar_order.pdf')     
        PlotLocalPolarOrderHistogram( FData, savepath / 'hist_local_polar_order.pdf')     
        PlotLocalNematicOrderVsTime( FData, savepath / 'heatmap_local_nematic_order.pdf')     
        PlotLocalNematicOrderHistogram( FData, savepath / 'hist_local_nematic_order.pdf')     
        PlotLocalNematicOrderContactNumberHistogram( FData, savepath / 'hist2_local_nematic_order_contact_number.pdf')     
        PlotLocalPolarOrderContactNumberHistogram( FData, savepath / 'hist2_local_polar_order_contact_number.pdf')     
        PlotLocalNematicOrderLocalPolarOrderHistogram( FData, savepath / 'hist2_local_nematic_order_local_polar_order.pdf')     
        # PlotLocalPackingFractionVsTime( FData, savepath / 'local_packing_fraction_img.pdf')     
        # PlotLocalPackingFractionHistogram( FData, savepath / 'local_packing_fraction_hist.pdf') 
    # Effective diameter
    if cfg['PlotEffectiveDiameter']:
        PlotEffectiveDiameter(FData, savepath / 'hist_effective_diameter.pdf')
    # }}}

    # Dynamics {{{
    if cfg['PlotDynamics']:
        # PlotMSD(FData, params, savepath / 'graph_msd.pdf',N=200)
        # PlotMobilityFilamentVsTime(FData, savepath / 'mobility.pdf')
        # PlotMobilityCrosslinkerVsTime(XData, savepath / 'mobility_xlinker.pdf')
        PlotMobilityCrosslinkerHist(XData, savepath / 'mobility_xlinker.pdf')
        # PlotCrosslinkerDwellTime(XData, savepath / 'dwell_time_xlinker.pdf')
    # }}}

    # RDF {{{
    if cfg['PlotRDF']:
        PlotRDF(FData, params)
        PlotRDF_PAP(FData, params)
    # }}}

    # Local Maps {{{
    if cfg['PlotLocalMaps']:
        PlotFilamentDensityLastNframes(FData, savepath / 'filament_density.pdf')
        PlotFilamentDensityMovie(FData, savepath / 'density_movie', frame_gap=1)
        PlotPackingFraction(FData, savepath / 'map_packing_fraction.pdf', N=20)
        PlotNumberDensity(FData, savepath / 'map_number_density.pdf', N=20)
        PlotNumberDensityXlink(FData, XData, savepath / 'map_number_density_xlink.pdf', N=20)
        PlotSOrder(FData, savepath / 'map_nematic_order.pdf')
        PlotPOrder(FData, savepath / 'map_polar_order.pdf')
        PlotFlux(FData, savepath / 'map_flux.pdf')
        PlotOrientation(FData, savepath / 'map_orientation.pdf')
        PlotNematicDirector(FData, savepath / 'map_nematic_director.pdf')
    # }}}

    # Condensation/Clustering {{{
    if cfg['ComputeFilamentCondensates']:
        PlotFilamentCondensation(FData, XData, params, write2vtk=cfg['WriteFilamentCondensatesToVTK'])

    # Filament Condensation / Clustering
    if cfg['ComputeFilamentClusters']:
        PlotFilamentClusters(FData, params, N=1000)
        
    # Crosslinker Clusters
    if cfg['ComputeCrosslinkerClusters']:
        PlotXlinkClusters(XData, params)
    # }}}

    # Write2VTK {{{

    # Write Angle to VTK
    if cfg['WriteAngleToVTK']:
        Write3DOrientation2vtk(FData, XData, params)

    # Write Correlations to VTK
    if cfg['WriteCorrelationsToVTK']:
        WriteLocalOrder2vtk(FData, params)
        WriteContactNumber2vtk(FData, params)
    # }}}

    # Cleanup {{{
    # Close data file
    data_filestream.close()
    del FData
    del XData
    
    gc.collect()
    print(tracemalloc.get_traced_memory())
    # stopping the library
    tracemalloc.stop()
    # }}}

if __name__ == '__main__':
    simpath = Path(sys.argv[1])
    params = {
            'sim_name': simpath.name,
            'sim_path': simpath,
            'plot_path': simpath / 'plots',
            'data_path': simpath / 'data.hdf5',
            }
    main(params)
