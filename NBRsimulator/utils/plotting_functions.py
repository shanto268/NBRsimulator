import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./JTF.mplstyle')
from matplotlib.colors import LogNorm
from matplotlib.gridspec import SubplotSpec

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

def plotTauDist(h,dist,bins=80,color='grey',alpha=0.3,label=None):
    '''Creates new figure with given distribution as a histogram and returns nonzero bins with centers.
    
    returns pyplot subplot, nonzero bin counts, nonzero bin centers
    ---------------------------
    dist:   data to histogram
    bins:   passed to pyplot.hist
    color:  passed to pyplot.hist
    alpha:  passed to pyplot.hist
    figsize:    passed to pyplot.figure
    '''
    if type(label) is str:
        hi = h.hist(dist,bins=bins,color=color,alpha=alpha,density=False,histtype='step',label=label)
    else:
        hi = h.hist(dist,bins=bins,color=color,alpha=alpha,density=False,histtype='step')
    # h.set_xlim(hi[1][1],hi[1][-1])
    # h.set_ylim(0,1.5*np.max(hi[0][3:]))
    binmask = np.array(hi[0] > 0,dtype=bool)
    BinCenters = (hi[1][1:] + hi[1][:-1])/2
    return hi[0][binmask], BinCenters[binmask]


def plotTauDistScaled(h,dist,bins=80,color='grey',alpha=0.3,label=None):
    '''Creates new figure with given distribution as a histogram and returns nonzero bins with centers.
    
    returns pyplot subplot, nonzero bin counts, nonzero bin centers
    ---------------------------
    dist:   data to histogram
    bins:   passed to pyplot.hist
    color:  passed to pyplot.hist
    alpha:  passed to pyplot.hist
    figsize:    passed to pyplot.figure
    '''
    hi = np.histogram(dist,bins=bins,density=True)
    taumean = np.mean(dist)
    poisson_scaled = lambda x: (x/taumean)*np.exp(-x/taumean)
    # h.set_xlim(hi[1][1],hi[1][-1])
    # h.set_ylim(0,1.5*np.max(hi[0][3:]))
    binmask = np.array(hi[0] > 0,dtype=bool)
    BinCenters = (hi[1][1:] + hi[1][:-1])/2
    logspace = np.logspace(np.log10(BinCenters[0]),np.log10(BinCenters[-1]),1000)
    poisson_model = poisson_scaled(BinCenters)
    Fidelity = np.sum(np.sqrt(BinCenters*hi[0]*poisson_model))/np.sum(BinCenters*hi[0])
    if type(label) is str:
        l = h.plot(BinCenters,BinCenters*hi[0],
                   c = color,ls='',marker='o',alpha=alpha,label=label)
    else:
        l = h.plot(BinCenters,BinCenters*hi[0],
                   c = color,ls='',marker='o',alpha=alpha)
    h.plot(logspace,poisson_scaled(logspace),ls='dashed',c=l[-1].get_color(),alpha=alpha)
    return hi[0][binmask], BinCenters[binmask]


def plotTimeSeries(ax,data,nEst,Q,time,start,stop,zeroTime=False):
    '''
    generates a new figure with 2 panel subplots comparing IQ data and 
    occupation for subset of time between start and stop.

    Parameters
    ----------
    data : array with shape (2, # samples)
        IQ data.
    nEst : array with shape (# samples)
        the occupation.
    time : array with shape (# samples)
        times corresponding to data.
    start : float
        start time for plotting (same units as parameter time)
    stop : flaot
        stop time for plotting (same units as parameter time).

    Returns
    -------
    fig : pyplot figure handle
    ax : pyplot axes for subplots
        shape is (2,) with ax[0] addressing the top panel.

    '''
    mask = np.logical_and(time >= start,time <=stop)
    plttime = time[mask] - time[mask][0] if zeroTime else time[mask]
    ax[0].plot(plttime,data[0][mask],label='I')
    ax[0].plot(plttime,data[1][mask],label='Q')
    ax[0].legend()
    ax[1].plot(plttime,nEst[mask],label='QP occ.')
    ax[1].plot(plttime,Q[mask],label='HMM est.')
    ax[1].legend()

def plotComplexHist(h,I,Q,bins=(80,80),returnHistData=False):
    '''From I and Q data, plot a greyscale log histogram in complex plane.
    
    returns a pyplot subplot for further modification.
    ----------------------------------------------------
    I:          1 dim data array.
    Q:          data array with same size aas I
    bins:       (a,b) where a is # bins along I and b is along Q
    figsize:    passed to pyplot.figure to set figure size
    ----------------------------------------------------
    
    Example:
    h = plotComplexHist(DATA[:,0],DATA[:,1])
    h.set_xlabel('I [mV]')
    h.set_ylabel('Q [mV]')
    h.set_title('test')
    plt.savefig(r'path/to/save.png')
    '''
    plt.figure(h.get_figure())
    hs = h.hist2d(I,Q,bins=bins,norm=LogNorm(),cmap=plt.get_cmap('Greys'))
    hb = plt.colorbar(hs[-1], ax=h, shrink=0.9, extend='both')
    h.grid()
    h.set_aspect('equal')