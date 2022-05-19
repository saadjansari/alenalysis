import numpy as np
import matplotlib.pyplot as plt
import src.decorators

def PlotXlinkOrientations(FData, XData, savepath, N=100):
    """ Plot the doubly bound xlinker head orientations w.r.t. to the filaments"""

    orts = calc_crosslinker_orientations(FData.orientation_, XData)

    fig,ax = plt.subplots(figsize=(4,3))
    ax.hist(orts[:,-1*N:].flatten(), bins=50, density=True)[-1]
    ax.set(xlabel=r'$\cos \theta$', ylabel='Probablity density', title='Filament-crosslinker orientation distribution')
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()

@src.decorators.timer
def calc_crosslinker_orientations(fil_orient, XData):
    """ Calculate the doubly bound xlinker head orientations w.r.t. to the filaments"""

    orts = np.zeros( (2*XData.nxlink_,XData.nframe_))
    orts[:] = np.NaN
    for jframe in np.arange(XData.nframe_):
    
        # xlink orientations
        xi = XData.pos_plus_[:,:,jframe] - XData.pos_minus_[:,:,jframe]
        
        # doubly-bound links
        xl_db = (XData.link0_[:,jframe] > -0.5) & (XData.link1_[:,jframe] > -0.5)
        xi[:,xl_db] = xi[:,xl_db] / np.linalg.norm(xi[:,xl_db], axis=0)
        
        # compute angle for each head separately, since their geometry is different.
        # For head 1, a supplementary angle is required.
        # Head 2, nothing special required.
        orts[np.where(xl_db)[0],jframe] = np.einsum('ij,ij->j', -1*xi[:,xl_db], fil_orient[:,XData.link0_[xl_db,jframe].astype(int),jframe])
        orts[XData.nxlink_+np.where(xl_db)[0],jframe] = np.einsum('ij,ij->j', xi[:,xl_db], fil_orient[:,XData.link1_[xl_db,jframe].astype(int),jframe])

    return orts
