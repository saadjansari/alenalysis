import os 
from pathlib import Path
import pdb 
import tracemalloc
import gc
import numpy as np
from numba import njit
import sys
import src.decorators
from src.read_config import get_config

def main( params):

    print('######### FILAMENT CORRELATIONS MODE #########')
    tracemalloc.start()
    gc.collect()
    
    sim_name = params['sim_name']
    simpath = params['sim_path']
    savepath = params['plot_path']
    if not Path.exists( savepath):
        os.mkdir( savepath)

    print('_'*50)
    print('\nSim Name : {0}'.format(sim_name))

    # Get files to load
    files_s, files_p, _ = find_all_frames(simpath)

    # number of frames
    nframe_ = len(files_s)

    # number of filaments
    nfil_ = count_sylinders(files_s[0])

    # Load config 
    cfg = get_config(simpath)

    # Initialize variables
    # Minimum Distance, Contact Number, Local Polar Order, Local Nematic Order
    CN = np.zeros((nfil_, nframe_))
    LPO = np.zeros((nfil_, nframe_))
    LNO = np.zeros((nfil_, nframe_))
    # LPO_com = np.zeros((nfil_, nframe_))
    # LNO_com = np.zeros((nfil_, nframe_))

    # Check if data exists . If yes, load and fill out arrays
    cn_path = simpath / 'contact_number.npy'
    lo_path = simpath / 'local_order.npy'
    # lo_com_path = simpath / 'local_order_com.npy'

    if Path.exists(cn_path):
        with open(str(cn_path), 'rb') as f:
            CN_pre = np.load(f)
        # Find how many frames processed, and fill out
        n_done_cn = CN_pre.shape[-1]
        print('Contact Number data exists: {0}/{1} frames'.format(n_done_cn, nframe_))
        if n_done_cn>nframe_:
            CN = CN_pre
        else:
            CN[:,:n_done_cn] = CN_pre
    else:
        n_done_cn = 0 
        print('No CN data exists')
    if Path.exists(lo_path):
        with open(str(lo_path), 'rb') as f:
            LPO_pre = np.load(f)
            LNO_pre = np.load(f)
        # Find how many frames processed, and fill out
        n_done_lo = LPO_pre.shape[-1]
        print('Local Order data exists: {0}/{1} frames'.format(n_done_lo, nframe_))
        if n_done_lo>nframe_:
            LPO = LPO_pre
            LNO = LNO_pre
        else:
            LPO[:,:n_done_lo] = LPO_pre
            LNO[:,:n_done_lo] = LNO_pre
    else:
        n_done_lo = 0 
        print('No LO data exists')
    # if Path.exists(lo_com_path):
        # with open(str(lo_com_path), 'rb') as f:
            # LPO_com_pre = np.load(f)
            # LNO_com_pre = np.load(f)
        # # Find how many frames processed, and fill out
        # n_done_lo_com = LPO_com_pre.shape[-1]
        # print('Local Order data exists: {0}/{1} frames'.format(n_done_lo_com, nframe_))
        # if n_done_lo_com>nframe_:
            # LPO_com = LPO_com_pre
            # LNO_com = LNO_com_pre
        # else:
            # LPO_com[:,:n_done_lo] = LPO_com_pre
            # LNO_com[:,:n_done_lo] = LNO_com_pre
    # else:
        # n_done_lo_com = 0 
        # print('No LO com data exists')
    # n_done = min([n_done_lo, n_done_lo_com, n_done_cn])
    n_done = min([n_done_lo, n_done_cn])

    # Loop over frames and process
    for jframe in range(nframe_):

        if jframe < n_done:
            continue
        
        print('Frame = {0}/{1}'.format(jframe+1, nframe_) )
        
        # Read sylinder data
        _,_,pos_minus_,pos_plus_, orientation_ = read_dat_sylinder(files_s[jframe])
    
        # Min Dist
        MD = minDistBetweenAllFilaments( 
            pos_minus_, 
            pos_plus_, 
            pos_minus_, 
            pos_plus_, cfg['box_size']) / cfg['diameter_fil']

        # Contact Number 
        if jframe >= n_done_cn:
            CN[:,jframe] = np.sum( np.exp(-1*(MD**2)), axis=1)

        # Local Order
        if jframe >= n_done_lo:
            LPO[:,jframe], LNO[:,jframe] = calc_local_order_frame( 
                    orientation_, 
                    MD)

        # # Local Order COM
        # if jframe >= n_done_lo_com:
            # LPO_com[:,jframe], LNO_com[:,jframe] = calc_local_order_frame_com( 
                    # orientation_, 0.5*(pos_plus_+pos_minus_), cfg['box_size'], cfg['diameter_fil'], scale=2)

        # Save every 10th frame
        if not jframe%10:
            print('Saving data')
            # Save data
            with open(str(cn_path), 'wb') as f:
                np.save(f, CN)

            # Save data
            with open(str(lo_path), 'wb') as f:
                np.save(f, LPO)
                np.save(f, LNO)

    # Save data
    with open(str(cn_path), 'wb') as f:
        np.save(f, CN)

    # Save data
    with open(str(lo_path), 'wb') as f:
        np.save(f, LPO)
        np.save(f, LNO)
        
    # # Save data
    # with open(str(lo_com_path), 'wb') as f:
        # np.save(f, LPO_com)
        # np.save(f, LNO_com)
    
    gc.collect()
    print(tracemalloc.get_traced_memory())
    # stopping the library
    tracemalloc.stop()

# READ {{{
def find_all_frames(spath):
    def fileKey(f):
        return int( f.parts[-1].split('_')[-1].split('.dat')[0] )
    def fileKey_c(f):
        return int( f.parts[-1].split('_')[-1].split('.pvtp')[0] )
    file_s = sorted( list(spath.glob('**/SylinderAscii*.dat')), key=fileKey)
    file_p = sorted( list(spath.glob('**/ProteinAscii*.dat')), key=fileKey)
    file_c = sorted( list(spath.glob('**/ConBlock*.pvtp')), key=fileKey_c)
    return file_s, file_p, file_c

def read_dat_sylinder(fname):
    # Read a SylinderAscii_X.dat file

    # open the file and read the lines
    with open(fname, 'r') as file1:
        filecontent = file1.readlines()

        # Delete the first two lines because they dont have any data
        filecontent[0:2] = []

        # Initialize numpy arrays for data
        gid = np.zeros(len(filecontent), dtype=int)
        radius = np.zeros(len(filecontent))
        pos_minus = np.zeros((3, len(filecontent)))
        pos_plus = np.zeros((3, len(filecontent)))
        orientation = np.zeros((3, len(filecontent)))

        for idx, line in enumerate(filecontent):

            # Split the string with space-delimited and convert strings into
            # useful data types
            data = line.split()
            gid[idx] = int(data[1])

            dat = np.array(list(map(float, data[2::])))
            radius[idx] = dat[0]
            pos_minus[:,idx] = dat[1:4]
            pos_plus[:,idx] = dat[4:7]
            xi = pos_plus[:,idx] - pos_minus[:,idx]
            orientation[:,idx] = xi / np.sqrt(xi.dot(xi))
    return gid, radius, pos_minus, pos_plus, orientation

def read_dat_protein(fname):
    # Read a ProteinAscii_X.dat file

    # open the file and read the lines
    with open(fname, 'r') as file1:
        filecontent = file1.readlines()

        # Delete the first two lines because they dont have any data
        filecontent[0:2] = []

        # Initialize numpy arrays for data
        gid = np.zeros(len(filecontent), dtype=int)
        pos0 = np.zeros((3, len(filecontent)))
        pos1 = np.zeros((3, len(filecontent)))
        link0 = np.zeros(len(filecontent), dtype=int)
        link1 = np.zeros(len(filecontent), dtype=int)

        for idx, line in enumerate(filecontent):

            # Split the string with space-delimited and convert strings into
            # useful data types
            data = line.split()
            # pdb.set_trace()
            gid[idx] = int(data[1])
            link0[idx] = int(data[9])
            link1[idx] = int(data[10])
            dat = np.array(list(map(float, data[2:9])))
            pos0[:,idx] = dat[1:4]
            pos1[:,idx] = dat[4::]
    return gid, pos0, pos1, link0, link1

def count_sylinders(fname):
    # count sylinders in a file
    with open(fname, 'r') as file1:
        filecontent = file1.readlines()
    return len(filecontent)-2
# }}}

# Min Dist {{{
@src.decorators.timer
@njit
def minDistBetweenAllFilaments(a0,a1, b0,b1, box_size):

    n_fil = a0.shape[1]
    dmat = np.zeros((n_fil,n_fil))
    for idx1 in np.arange( n_fil):
        for idx2 in np.arange( n_fil):
            if idx1 == idx2:
                dmat[idx1,idx2] = 1000
            elif idx1 > idx2:
                continue
            else:
                dmat[idx1,idx2] = minDistBetweenTwoFil( a0[:,idx1],a1[:,idx1], b0[:,idx2], b1[:,idx2], box_size)
                dmat[idx2,idx1] = dmat[idx1,idx2]
                
    return dmat

@njit
def minDistBetweenTwoFil(p1, p2, p3, p4, box_size):
    """
    Find Minimum Distance between two filaments

    https://www.mathworks.com/matlabcentral/fileexchange/32487-shortest-distance-between-two-line-segments
    which adapted this from Dan Sunday's Geometry Algorithms originally written in C++
    http://softsurfer.com/Archive/algorithm_0106/algorithm_0106.htm#dist3D_Segment_to_Segment

    p1 = f1.pos_start
    p2 = f1.pos_end
    p3 = f2.pos_start
    p4 = f2.pos_end
    Computes the minimum distance between two line segments. Code
    is adapted for Matlab from Dan Sunday's Geometry Algorithms originally
    written in C++
    http://softsurfer.com/Archive/algorithm_0106/algorithm_0106.htm#dist3D_Segment_to_Segment
    Usage: Input the start and end x,y,z coordinates for two line segments.
    p1, p2 are [x,y,z] coordinates of first line segment and p3,p4 are for
    second line segment.
    Output: scalar minimum distance between the two segments.
     Example:
            P1 = [0 0 0];     P2 = [1 0 0];
      P3 = [0 1 0];     P4 = [1 1 0];
            dist = minDistBetweenTwoFil(P1, P2, P3, P4)
    """

    u = p1 - p2
    v = p3 - p4
    w = p2 - p4

    # Apply PBC
    for jdim in np.arange(box_size.shape[0]):
        bs=box_size[jdim]
        if u[jdim] > 0.5*bs:
            u[jdim]-=bs
        elif u[jdim] <= -0.5*bs:
            u[jdim]+=bs
        if v[jdim] > 0.5*bs:
            v[jdim]-=bs
        elif v[jdim] <= -0.5*bs:
            v[jdim]+=bs
        if w[jdim] > 0.5*bs:
            w[jdim]-=bs
        elif w[jdim] <= -0.5*bs:
            w[jdim]+=bs

    a = np.dot(u,u)
    b = np.dot(u,v)
    c = np.dot(v,v)
    d = np.dot(u,w)
    e = np.dot(v,w)
    D = a*c - b*b
    sD = D
    tD = D

    SMALL_NUM = 0.00000001

    # compute the line parameters of the two closest points
    if D < SMALL_NUM: # the lines are almost parallel
        sN = 0.0     # force using point P0 on segment S1
        sD = 1.0     # to prevent possible division by 0.0 later
        tN = e
        tD = c
    else:             # get the closest points on the infinite lines
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:   # sc < 0 => the s=0 edge is visible
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:  # sc > 1 => the s=1 edge is visible
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:            # tc < 0 => the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:       # tc > 1 => the t=1 edge is visible
        tN = tD
        # recompute sc for this edge
        if -d + b < 0.0:
            sN = 0
        elif -d + b > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    # finally do the division to get sc and tc
    if  np.absolute(sN) < SMALL_NUM:
        sc = 0.0
    else:
        sc = sN / sD

    if np.absolute(tN) < SMALL_NUM:
        tc = 0.0
    else:
        tc = tN / tD

    # get the difference of the two closest points
    dP = w + (sc * u) - (tc * v);  # = S1(sc) - S2(tc)
    distance = np.linalg.norm(dP);
    # outV = dP;

    # outV = outV      # vector connecting the closest points
    # cp_1 = p2+sc*u  # Closest point on object 1
    # cp_2 = p4+tc*v  # Closest point on object 2
    return distance
# }}}

# calc_local_order_frame {{{
@src.decorators.timer
def calc_local_order_frame( orients, min_dist):
    """
    Calculate the local polar order via min dist
    Inputs: 
        orient_array : 3 x N 
        min_dist : N x N
    """

    Porder = np.zeros( orients.shape[1])
    Sorder = np.zeros( orients.shape[1])

    # Cos theta^2 of all filaments (N x N)
    cosTheta2 = np.tensordot(orients,orients,axes=((0),(0)))**2

    # get contact number for this frame
    contactNumber = np.sum( np.exp( -1*( min_dist[:,:]**2)), axis=1)

    # dist gaussian factor
    g_factor = np.exp( -1*min_dist**2 )

    for idx in np.arange( orients.shape[1]):

        # orientation of reference filament
        o1 = orients[:,idx].reshape(-1,1)

        # indices of other filaments
        idx_rest = np.delete( np.arange(orients.shape[1]), idx)

        # local polar order
        Porder[idx] = np.sum( np.sum( o1*orients[:,idx_rest], axis=0)*g_factor[idx,idx_rest]) / contactNumber[idx]
        
        # Local Nematic Order
        Sorder[idx] = 0.5*np.sum( (3*cosTheta2[idx,idx_rest] - 1)*g_factor[idx,idx_rest] ) / contactNumber[idx]

    return Porder, Sorder
# }}}

# calc_local_order_frame_com {{{
@njit 
def calc_dist_com(com, box_size):
    # calculate com distances

    nfil = com.shape[1]
    ndim = com.shape[0]
    cdist= np.zeros((ndim, nfil, nfil))
    for jdim in np.arange(ndim):
        bs = box_size[jdim]
        for jfil in np.arange(nfil):
            for kfil in np.arange(nfil):
                dd = com[jdim,kfil]-com[jdim,jfil]
                if dd <= -0.5*bs:
                    dd += bs
                elif dd > 0.5*bs:
                    dd -= bs
                cdist[jdim,jfil,kfil] = dd
            cdist[jdim,jfil,jfil] = 40000

    # pdb.set_trace()
    MD = np.sqrt(np.sum(cdist**2, axis=0))
    return MD

@src.decorators.timer
def calc_local_order_frame_com( orients, com, box_size, diameter, scale=1):
    """
    Calculate the local polar order via com 
    Inputs: 
        orient_array : 3 x N 
        min_dist : N x N
    """

    # distance com
    min_dist = calc_dist_com(com, box_size)/diameter

    Porder = np.zeros( orients.shape[1])
    Sorder = np.zeros( orients.shape[1])

    # Cos theta^2 of all filaments (N x N)
    cosTheta2 = np.tensordot(orients,orients,axes=((0),(0)))**2

    # dist gaussian factor
    g_factor = np.exp( -1*(min_dist/scale)**2 )

    # get contact number for this frame
    contactNumber = np.sum( g_factor, axis=1)

    for idx in np.arange( orients.shape[1]):

        # orientation of reference filament
        o1 = orients[:,idx].reshape(-1,1)

        # indices of other filaments
        idx_rest = np.delete( np.arange(orients.shape[1]), idx)

        # local polar order
        Porder[idx] = np.sum( np.sum( o1*orients[:,idx_rest], axis=0)*g_factor[idx,idx_rest]) / contactNumber[idx]
        
        # Local Nematic Order
        Sorder[idx] = 0.5*np.sum( (3*cosTheta2[idx,idx_rest] - 1)*g_factor[idx,idx_rest] ) / contactNumber[idx]

    return Porder, Sorder
# }}}


if __name__ == '__main__':
    simpath = Path(sys.argv[1])
    params = {
            'sim_name': simpath.name,
            'sim_path': simpath,
            'plot_path': simpath / 'plots',
            }
    main(params)
