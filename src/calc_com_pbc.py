import numpy as np

# calc_com_pbc {{{
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
# }}}
