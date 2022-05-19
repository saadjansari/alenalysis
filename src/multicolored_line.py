import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_multicolored_line(x,y,color_by,ax, label='None', alpha=0.7,lw=1):

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(color_by.min(), color_by.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(color_by)
    lc.set_linewidth(lw)
    lc.set_alpha(alpha)
    line = ax.add_collection(lc)
    plt.colorbar(line, ax=ax, label=label)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    return ax

def plot_multicolored_lines(xarr,yarr,color_by,ax, label='None', alpha=0.7, lw=1):

    # xarr : N x T
    # yarr : N x T
    for jfil in np.arange(xarr.shape[0]):
        x = xarr[jfil,:]
        y = yarr[jfil,:]
        
        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(color_by.min(), color_by.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(color_by)
        lc.set_linewidth(lw)
        lc.set_alpha(alpha)
        line = ax.add_collection(lc)

    plt.colorbar(line, ax=ax, label=label)
    ax.set_xlim(xarr.min(), xarr.max())
    ax.set_ylim(yarr.min(), yarr.max())
    return ax

def test_multicolored():
    x = np.linspace(0, 3 * np.pi, 500)
    y = np.sin(x)
    dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative
    fig,ax = plt.subplots()
    plot_multicolored_line(x,y,dydx,ax)
    plt.show()

if __name__ == '__main__':
    test_multicolored()
