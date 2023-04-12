import matplotlib
matplotlib.use('TkAgg')
import sys
sys.path.append(r"/Users/jamesfarmer/ResPy/")
sys.path.append(r"/Users/jamesfarmer/lflPython/")
from Resonator import Resonator
import numpy as np
from TrappingSimulator import QPtrapper
from ResSimulator import NBResonator
from time import perf_counter
import matplotlib.pyplot as plt
plt.style.use('./JTF.mplstyle')
from hmmlearn import hmm
import fitTools.quasiparticleFunctions as qp
from plotting_functions import *

######################
# parameters
#####################
duration = 1 # seconds to record data
sampleRate = 1e6
N = int(duration*sampleRate)
xne = 8e-7
tauRelease = 1/3e4
tauCommon = 4e-4
tauRare = 1e-2
tauRecomb = 2e-3
phi = 0.45
Lj = 21.2e-12
L = 1.89e-9
C = 0.2776e-12
Qi = 25000
Qe = 5000
photonRO = 25
delKappa = -0.5
Delta = 2.72370016e-23
T = 0.025
memory = 200e-6
pairrateratio = 0.5

args = {'N':N,'Lj':Lj,'xne':xne,'tauRelease':tauRelease,'tauCommon':tauCommon,'tauRare':tauRare,
        'tauRecomb':tauRecomb, 'sampleRate':sampleRate, 'phi':phi, 'Delta':Delta, 'T':T, 
        'pairrateratio':pairrateratio}
resArgs = {'L':L, 'C':C, 'photonRO':photonRO, 'Qi':Qi, 
           'Qe':Qe, 'sampleRate':sampleRate, 'delKappa':delKappa}

##########################
# start the loop
#########################
lMethods = ['Poisson','non-Markovian','pairwise','burst']
# lMethods = ['poisson','pairs']
nMethods = len(lMethods)

mainfig,mainax = plt.subplots(nMethods,3,num=1,figsize=[6.6,8.8])
grid = plt.GridSpec(nMethods, 3)

for i,method in enumerate(lMethods):
    create_subtitle(mainfig, grid[i, ::], f'{method}')
    ##########################
    # run the QP trapper
    ##########################
    
    now = perf_counter()
    sim = QPtrapper(method=method,**args)
    timer = perf_counter()-now
    print(f'\n{method.upper()}\n')
    print('qptrapper runtime: {} seconds'.format(timer))
    print('The average number of trapped QPs is {:.4}'.format(np.mean(sim.nTrapped)))

    ###########################
    # run the resonator response
    ###########################

    now2 = perf_counter()
    res = NBResonator(sim,**resArgs)
    duration2 = perf_counter() - now2
    print('Resonator runtime: {}'.format(duration2))

    #################################
    # Make complex histogram plot and get the means - poisson
    ################################
    plotComplexHist(mainax[i,0],res.signal.real,res.signal.imag)
    mainax[i,0].set_xlabel('I')
    mainax[i,0].set_ylabel('Q')
    means = plt.ginput(-1,timeout=0)

    ###################
    # fit HMM
    ###################
    
    ncomp = len(means)

    M = hmm.GaussianHMM(
        n_components=ncomp,
        covariance_type='full',
        min_covar=0.0001,
        means_weight=1,
        means_prior=np.array(means),
        algorithm='viterbi',
        random_state=None,
        n_iter=20,
        tol=0.005,
        verbose=False,
        params='stmc',
        init_params='s',
        implementation='log',
    )
    covars = np.zeros((ncomp,2,2))
    for j in range(ncomp):
        covars[j] = np.array([[1e-2,1e-6],[1e-6,1e-2]])
    # covars = np.array([[1e-2,1e-6],[1e-6,1e-2]])
    M.covars_ = covars
    M.means_ = np.array(means)
    transmat = np.ones((ncomp,ncomp)) * 0.001*(i+1)
    np.fill_diagonal(transmat,1-(ncomp-1)*0.001*(i+1))
    M.transmat_ = transmat

    data = np.vstack((res.signal.real,res.signal.imag))
    M.fit(data.T[:N//4])
    logprob, Q = M.decode(data.T)

    colors = plt.cm.rainbow(np.linspace(0,1,ncomp))
    qp.make_ellipsesHMM(M,mainax[i,0],colors)

#     ###########################
#     # plot time series
#     ##########################
    
    time = np.arange(N)/sampleRate
#     plotTimeSeries(mainax[1,i],data_p,np.array(sim_poisson.nTrapped),Qp,time*1000,10,12,zeroTime=False)
#     plt.suptitle(f'Poisson simulation | xne = {xne}')
#     ax2[1].set_ylabel('n trapped')
#     ax2[0].set_ylabel('Resonator response')
#     ax2[1].set_xlabel('Time [ms]')
#     plt.show(block=False);

    #####################
    # print metrics
    ###################
    
    print(f'\n{method}\n')
    print(f'there are {np.mean(sim.nTrapped):.4f} trapped QPs on average')
    print(f'HMM estimates {np.mean(Q):.4f} average trapped QPs')
    print(M.means_)
    rates = qp.getTransRatesFromProb(sampleRate,M.transmat_)
    print(f'the actual trap rate provided is {1/(1e6/(8.73e15*(xne))):.1f} Hz, while our estimate is {rates[0,1]:.1f} Hz')
    trueocc = np.array(sim.nTrapped)
    tmask = Q == trueocc
    print(np.sum(tmask))
    F1 = []
    lPrec = []
    lRec = []
    for k in range(ncomp):
        print('\nk\n')
        tp = np.sum(Q[tmask] == k)
        print(f'tp = {tp}')
        fp = np.sum(Q == k) - tp
        print(f'fp = {fp}')
        fn = np.sum(trueocc == k) - tp
        print(f'fn = {fn}')
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        lPrec.append(prec)
        lRec.append(rec)
        F1.append(2*prec*rec/(prec+rec))
        print(f'prec = {prec:.3f} | rec = {rec:.3f}')
    print(f'\nF1s are {F1}')
    print(f'The composite F1 score for this test is {np.mean(F1):.3f}')
    print(f'composite prec = {np.mean(lPrec):.3f} | composite rec = {np.mean(lRec):.3f}')


    #####################
    # plot distribution
    #####################
    
    hmmdist = qp.extractTimeBetweenTrapEvents(Q,time*1e6)
    dist = qp.extractTimeBetweenTrapEvents(sim.nTrapped,time*1e6)
    
    bins = np.linspace(0.95*min(np.min(hmmdist),np.min(dist)),
                       1.05*max(np.max(hmmdist),np.max(dist)),
                       80)
    countshmm, centershmm = plotTauDist(mainax[i,1],hmmdist,bins=bins,color='purple',alpha=0.5,label='HMM')
    counts, centers = plotTauDist(mainax[i,1],dist,bins=bins,color='orange',alpha=0.5,label='Simulated')
    if i == 0:
        mainax[i,1].legend(loc = 'upper right')
    mainax[i,1].set_xlabel('$\\tau$ [$\mu$s]')
    mainax[i,1].set_ylabel('Counts')
    mainax[i,1].set_yscale('log')
    # mainax[1,i].set_title('Poisson')
    
    ##################
    # plot scaled dist
    ##################

    logbins = np.logspace(np.log10(0.95*min(np.min(hmmdist),np.min(dist))),
                          np.log10(1.05*max(np.max(hmmdist),np.max(dist))),
                          40)
    countshmm2, centershmm2 = plotTauDistScaled(mainax[i,2],hmmdist,bins=logbins,
                                                color='purple',alpha=0.5,label='HMM')
    counts2, centers2 = plotTauDistScaled(mainax[i,2],dist,bins=logbins,
                                          color='orange',alpha=0.5,label='Simulated')
    mainax[i,2].set_xscale('log')
    if i == 0:
        mainax[i,2].legend(loc = 'upper left')
    mainax[i,2].set_xlabel('$\\tau$ [$\mu$s]')
    mainax[i,2].set_ylabel('$\\tau P(\\tau)$')
    # mainax[2,i].set_title('Poisson')
    
###########################################
# simulation finished, save and show the final product
###########################################
plt.tight_layout()
plt.savefig('comparison.pdf',dpi=300,transparent=True)
plt.show()