import numpy as np
import pdb
from src.write2vtk import add_array_to_vtk

def Write3DOrientation2vtk(FData, XData, params):

    # Calculate filament orientations
    Phi,Theta = CalcXYZOrientation( FData.pos_minus_, FData.pos_plus_)

    add_array_to_vtk(Phi, 'Phi', params['sim_path'])
    add_array_to_vtk(Theta, 'Theta', params['sim_path'])


def CalcXYZOrientation( pos0, pos1):

    # XY orientation
    xi = pos1-pos0
    Phi = np.arctan2( xi[1,:,:], xi[0,:,:])
    Phi[ Phi < 0] += 2*np.pi

    # Z orientation
    xi = pos1-pos0
    r_xy = np.linalg.norm( xi[:2,:,:], axis=0)
    Theta = np.arctan( xi[2,:,:]/r_xy)

    return Phi,Theta

