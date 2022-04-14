import numpy as np
from scipy.spatial.distance import pdist, squareform

# pdist_pbc {{{
def pdist_pbc(pos, box_size):
    """ pos is dimension N x 3 """
    for jd in range(pos.shape[1]):
        pd = pdist(pos[:,jd].reshape(-1,1))
        pd[ pd > 0.5*box_size[jd] ] -= box_size[jd]
        try:
            total += pd**2
        except:
            total = pd**2
    total = squareform( np.sqrt(total) )
    return total
# }}}

# pdist_pbc_xyz_max {{{
def pdist_pbc_xyz_max(pos, box_size):
    """ pos is dimension N x 3 """
    dist_xyz = np.zeros(3)
    for jd in range(pos.shape[1]):
        pd = pdist(pos[:,jd].reshape(-1,1))
        pd[ pd > 0.5*box_size[jd] ] -= box_size[jd]
        dist_xyz[jd] = np.max( np.abs(pd) )
    return dist_xyz 
# }}}

