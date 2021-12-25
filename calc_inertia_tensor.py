import numpy as np
import pdb

def calc_inertia_tensor3d_pbc(pos, box_size):
    """ calulate inertia tensor of pos (N x 3) with pbc """

    # find center of mass
    com = calc_com_pbc(pos, box_size)

    # find periodic distance to com for each point
    xyz = np.zeros_like(pos)

    for jd in range(pos.shape[1]):
        cc = pos[:,jd]-com[jd]
        Lc = box_size[jd]
        cc[ cc >= Lc/2] -= Lc
        cc[ cc < -Lc/2] += Lc
        xyz[:,jd] = cc

    # sum over outer products
    outs = np.zeros((3,3))
    for jc in range(pos.shape[0]):
        outs+=np.outer(xyz[jc,:], xyz[jc,:])
    
    # tensor components
    # Ixx: y^2 + z^2
    Ixx = outs[1,1] + outs[2,2]
    # Iyy: x^2 + z^2
    Iyy = outs[0,0] + outs[2,2]
    # Izz: x^2 + y^2
    Izz = outs[0,0] + outs[1,1]
    # Ixy = Iyx : -xy
    Ixy = -outs[0,1]
    Iyx = -outs[0,1]
    # Ixz = Izx : -xz
    Ixz = -outs[0,2]
    Izx = -outs[0,2]
    # Iyz = Izy : -yz
    Iyz = -outs[1,2]
    Izy = -outs[1,2]
    I = np.array([
        [Ixx,Ixy,Ixz],
        [Iyx,Iyy,Iyz],
        [Izx,Izy,Izz]])
    return I

def calc_com_pbc(pos, box_size):
    """ calulate com of pos (N x 3) with pbc """
    
    # Map each dimension from a line to a circle.
    com = np.zeros(pos.shape[1])
    for jd in range(pos.shape[1]):
        angle = 2*np.pi*pos[:,jd]/box_size[jd]

        # Get coordinates for a point on a circle for each particle in this dimension
        costheta = np.cos(angle)
        sintheta = np.sin(angle)

        # Get mean coordinate in this dimension
        costheta_com = np.mean(costheta)
        sintheta_com = np.mean(sintheta)

        # Find an angle associated with this point
        angle_com = np.arctan2(-1*sintheta_com, -1*costheta_com) + np.pi
        
        # Transform back to line
        com[jd] = angle_com*box_size[jd]/(2*np.pi)

    return com
