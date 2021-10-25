import numpy as np
import os, pdb 
from pathlib import Path
from DataHandler import FilamentSeries, CrosslinkerSeries
from read_config import *
from read_files import *
from CalcNumXlinks import *
from CalcOrderParameters import *
from CalcOverlaps import *

# Define sims to analyze
mainpath = Path('/scratch/summit/saan8193/alens/conf_scan/sims')
sim_names = [
        'U1_PF4_X3_w1_p1_k1',
        ]
mainpath = Path('/Users/saadjansari/Documents/Projects/Results/Tactoids')
sim_names = [
        # 'tactoid_vol_exclusion/test',
        # 'tactoid_vol_exclusion/test_lambda0',
        # 'length150_nosteric',
        # 'Usteric/U1.6',
        # 'Usteric/U3.2',
        # 'Usteric/U6.4'
        'fine',
        ]

for sim_name in sim_names:
    simpath = mainpath / sim_name
    savepath = simpath / 'plots'
    if not Path.exists(savepath):
        os.mkdir(savepath)

    print('Sim: {0}'.format(sim_name))
    
    # Read sim into FilamentData and CrosslinkerData
    FData, XData = read_sim(simpath, sim_name[0])

    # Plot Trajectories
    FData.plot_trajectories(savepath / 'traj_filament.pdf')
    XData.plot_trajectories(savepath / 'traj_xlink.pdf')

    # Analysis things
    # 1. Number of crosslinkers per filament
    PlotStateCounts(FData, XData, savepath / 'xlink_states.pdf')
    PlotXlinkPerFilamentVsTime( FData, XData, savepath / 'xlinks_per_fil_img.pdf')
    PlotXlinkPerFilament( FData, XData, savepath / 'xlinks_per_fil_hist.pdf')
    PlotXlinkPerFilamentVsTimeMax( FData, XData, savepath / 'xlinks_max.pdf', 5)

    # 2. Crosslinker length vs time
    XData.plot_length_mean_vs_time( simpath / 'xlink_length_vs_time.pdf')
    XData.plot_energy_mean_vs_time( simpath / 'xlink_energy_vs_time.pdf')
    
    # 3. Order parameters
    PlotNematicOrder( FData, savepath/ 'nematic_order.pdf')     
    PlotPolarOrder( FData, savepath / 'polar_order.pdf')     
    PlotNematicAndPolarOrder( FData, savepath / 'nematic_polar_order.pdf')     
    PlotZOrder( FData, savepath / 'z_order.pdf')     

    # 4. Filament overlaps, and effective diameter
    PlotEffectiveDiameter(FData, savepath / 'effective_diameter.pdf')
