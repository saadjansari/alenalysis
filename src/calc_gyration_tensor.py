import numpy as np
import pdb

def calc_gyration_tensor3d_pbc(pos, box_size):
    """ calulate gyration tensor of pos (N x 3) with pbc """

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
    outs = outs/pos.shape[0]

    # diagonalize
    vals, vecs = np.linalg.eig(outs)

    # sort vals (ascending order)
    inds = vals.argsort()
    vals = vals[inds]
    vecs = vecs[:,inds]

    Rg2 = np.sum( vals)
    asphericity = vals[2] - 0.5*( vals[0] + vals[1])
    acylindricity = vals[1] - vals[0]
    shape_anisotropy = (asphericity**2 + (3/4)*acylindricity**2) / Rg2**2

    # print('Radius of Gyration Squared = {0:3f}'.format(Rg2))
    # print('Asphericity = {0:3f}'.format(asphericity))
    # print('Acylindricity = {0:3f}'.format(acylindricity))
    # print('Shape Anisotropy K^2= {0:3f}'.format(shape_anisotropy))
    G_data = {
            'Rg2': Rg2,
            'asphericity': asphericity,
            'acylindricity': acylindricity,
            'shape_anisotropy': shape_anisotropy,
            'eigenvalues': vals,
            'eigenvectors': vecs,
            'G': outs
            }
    
    return outs, G_data

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
