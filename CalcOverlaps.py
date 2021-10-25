from numba import njit
import numpy as np
import decorators
from DataHandler import *

# Plotting
def PlotEffectiveDiameter(FData, savepath):
    """ Plot the effective diameter probablity distribution"""

    print('Plotting effective diameter')
    # Overlaps
    overlaps = calc_effective_diameter( FData.pos_minus_, FData.pos_plus_, FData.diameter_)

    # Bins
    bins = np.linspace(0,1,100)
    cen = (bins[:-1]+bins[1:])/2

    # Plots
    fig,ax = plt.subplots()
    pdb.set_trace()
    counts = ax.hist(overlaps, bins, density=True)[0]
    pdb.set_trace()
    
    # Effective diameter
    Deff = np.sum( counts*cen) / np.sum(counts)

    # Labels
    ax.set(xlabel='Collision diameter / D', 
            ylabel='Probability Density', 
            title='Effective Diameter = {:.3f}D'.format(Deff))
    ax.set_xlim(left=-0.001)
    ax.set_ylim(bottom=-0.01)

    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")

def calc_effective_diameter(pos_minus, pos_plus, diameter):
    # Calculate the effective diameter by computing overlaps 
    
    # Number of frames 
    n_frames = pos_minus.shape[2]

    # Overlap distance list
    overlapList = []

    for jframe in range(0,n_frames,10):
        print('Frame = {0}/{1}'.format(jframe,n_frames) )
    
        # overlap matrix (normalized by filament diameter)
        bmat = minDistBetweenAllFilaments( pos_minus[:,:,jframe], 
                pos_plus[:,:,jframe], 
                pos_minus[:,:,jframe], 
                pos_plus[:,:,jframe]) / diameter

        # unique distances as upper right triangle 
        dists = bmat[np.triu_indices(bmat.shape[0], 1)]

        for dd in dists:
            if dd <= 1:
                overlapList.append(dd)
    
    return np.array( overlapList)

@njit
def minDistBetweenTwoFil(p1, p2, p3, p4):
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
    outV = dP;

    # outV = outV      # vector connecting the closest points
    # cp_1 = p2+sc*u  # Closest point on object 1
    # cp_2 = p4+tc*v  # Closest point on object 2
    return distance

# @njit
# def minDistToRefFilament(ref0,ref1,pos0,pos1):

    # n_fil = pos.shape[1]
    # dvec = np.zeros(n_fil)
    # for idx in np.arange( n_fil):
        # dvec[idx] = minDistBetweenTwoFil(ref0,ref1, pos0[:,idx], pos1[:,idx])
        # if dvec[idx] == 0:
            # dvec[idx] = 1000
    # return dvec

@njit
def minDistBetweenAllFilaments(a0,a1, b0,b1):

    n_fil = a0.shape[1]
    dmat = np.zeros((n_fil,n_fil))
    for idx1 in np.arange( n_fil):
        for idx2 in np.arange( n_fil):
            if idx1 == idx2:
                dmat[idx1,idx2] = 1000
            elif idx1 > idx2:
                continue
            else:
                dmat[idx1,idx2] = minDistBetweenTwoFil( a0[:,idx1],a1[:,idx1], b0[:,idx2], b1[:,idx2])
                dmat[idx2,idx1] = dmat[idx1,idx2]
                
    return dmat
