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
from scipy.constants import hbar
import h5py
import os
from scipy.stats import mode

######################
# parameters
#####################
duration = 0.5 # seconds to record data
sampleRate = 5e6
sampleRateMHz = sampleRate * 1e-6
N = int(duration*sampleRate)
xne = 8e-7
tauRelease = 1/3e4
tauCommon = 4e-4
tauRare = 1e-2
tauRecomb = 2e-3
phi = 0.47
# Lj = 21.2e-12
# L = 1.89e-9
# C = 0.2776e-12
L = 1.82897e-9
Lj = 20.8475e-12
C = 0.739929e-12
Qi = 250000
Qe = 18500
# photonRO = 25
delKappa = -0.5
Delta = 2.72370016e-23
T = 0.025
memory = 200e-6
pairrateratio = 0.5
method = 'Poisson'
photonNoise = 1.5

args = {'N':N,'Lj':Lj,'xne':xne,'tauRelease':tauRelease,'tauCommon':tauCommon,'tauRare':tauRare,
        'tauRecomb':tauRecomb, 'sampleRate':sampleRate, 'phi':phi, 'Delta':Delta, 'T':T, 
        'pairrateratio':pairrateratio,'method':method}
resArgs = {'L':L, 'C':C, 'Qi':Qi, 
           'Qe':Qe, 'sampleRate':sampleRate, 'delKappa':delKappa,'photonNoise':photonNoise}

##########################
# start the loop
#########################
# lMethods = ['Poisson','non-Markovian','pairwise','burst']
# lMethods = ['poisson','pairs']
# nMethods = len(lMethods)
powersdB = np.arange(-127,-155,-1)

Ljphi = Lj/(1-np.sin(np.pi*phi/2)*np.arctanh(np.sin(np.pi*phi/2)))
omega0 = 1/(np.sqrt((L + Ljphi)*C))
Q0 = 1/(1/Qi + 1/Qe)
print(omega0/2/np.pi)
print(1/(1/Qi + 1/Qe))
print(omega0*(1/Qi + 1/Qe)/2/np.pi)

Gain = 120

def Pin_to_n(Pdb,k,kc):
    P = 10**(Pdb/10)*1e-3
    return P*(1 + np.sqrt(kc/k))**2 / (hbar*omega0*k)

def dBm2mV(x):
    return np.sqrt(50 * 10**(x/10) / 1000)*1000*np.sqrt(2)

def BoxcarMode(data,avgTime,sampleRate,returnRate=False):
    nAvg = int(max(avgTime*sampleRate,1))
    if nAvg == 1 and returnRate:
        return data,sampleRate
    elif nAvg == 1:
        return data
    if len(data.shape) == 2:
        nSamples = data.shape[1]
        data2 = data[:,:(nSamples//nAvg)*nAvg].reshape((2,nSamples//nAvg,nAvg))
    else:
        nSamples = len(data)
        data2 = data[:(nSamples//nAvg)*nAvg].reshape((nSamples//nAvg,nAvg))
    if returnRate:
        return mode(data2,keepdims=True,axis=1)[0].squeeze(), sampleRate/nAvg
    return mode(data2,keepdims=True,axis=1)[0].squeeze()

# for p in powersdB:
#     print(f'{p} dB = {Pin_to_n(p,omega0/Q,omega0/Qe):.4f} photons')

# mainfig,mainax = plt.subplots(nMethods,3,num=1,figsize=[6.6,8.8])
# grid = plt.GridSpec(nMethods, 3)

########
# save file
#######
metainfo = {
    'f0': omega0/np.pi/2,
    'LOf': (omega0 + delKappa*omega0/Q0)/np.pi/2,
    'phi': phi,
    'sampleRateMHz': sampleRateMHz,
    'durationSeconds': duration,
    'Temperature': T}
SPATH = r'/Users/jamesfarmer/Documents/LFL/HMM_Bootstrap_1MHz'
savefile = os.path.join(SPATH,'dataset.hdf5')
if not os.path.exists(os.path.split(savefile)[0]):
    os.makedirs(os.path.split(savefile)[0])
with h5py.File(savefile,'a') as f:
    g = f.require_group(f'PHI{phi*1000:3.0f}')
    for key in metainfo:
        g.attrs.create(key,metainfo[key])

figpath = os.path.join(SPATH,'figures',f'phi{phi*1000:3.0f}')
import os
if not os.path.exists(figpath):
    os.makedirs(figpath)

for i,p in enumerate(powersdB):
    # create_subtitle(mainfig, grid[i, ::], f'{method}')
    ##########################
    # run the QP trapper
    ##########################
    
    now = perf_counter()
    sim = QPtrapper(**args)
    timer = perf_counter()-now
    # print(f'\n{method.upper()}\n')
    print('\nqptrapper runtime: {} seconds'.format(timer))
    print('The average number of trapped QPs is {:.4}'.format(np.mean(sim.nTrapped)))

    ###########################
    # run the resonator response
    ###########################

    now2 = perf_counter()
    res = NBResonator(sim,photonRO=Pin_to_n(p,omega0/Q0,omega0/Qe),**resArgs)
    duration2 = perf_counter() - now2
    print('Resonator runtime: {}'.format(duration2))

    #########################
    # scale up the system from gain
    #########################
    Vscale = dBm2mV(p + Gain)
    data = np.vstack((res.signal.real,res.signal.imag)) * Vscale


    #################################
    # Make complex histogram plot and get the means - poisson
    ################################
    figmeans,axmeans = plt.subplots(1,1)
    plotComplexHist(axmeans,data[0],data[1])
    axmeans.set_xlabel('I')
    axmeans.set_ylabel('Q')
    if i == 0:
        means = plt.ginput(-1,timeout=0)
        n_comp = len(means)
        covars = np.zeros((n_comp,2,2))
        for j in range(n_comp):
            covars[j] = np.array([[1,1e-4],[1e-4,1]])
    plt.close()

    ###################
    # fit HMM
    ###################
    
    

    intTime = 1
    # breakTime = 50
    SNRmin = 3

    # pull in data and downsample 
    
    data, sr = qp.BoxcarDownsample(data,intTime,sampleRateMHz,returnRate = True)
    # data = qp.uint16_to_mV(data)

    # print(i)
    # fit the HMM
    M = hmm.GaussianHMM(n_components=n_comp,covariance_type='full',init_params='s',n_iter=50,tol=0.005)
    M.means_ = means
    M.covars_ = covars
    transmat = np.ones((n_comp,n_comp)) * 0.001
    np.fill_diagonal(transmat,1-(n_comp-1)*0.001)
    M.transmat_ = transmat

    M.fit(data.T)




    with h5py.File(savefile,'r') as ff:
        f = ff[f'PHI{phi*1000:3.0f}']
        oldrates = f[f'{i-1}/transitionRatesMHz'][:] if i != 0 else 0.0001*np.ones((n_comp,n_comp))
        lifetimes = np.array([1/oldrates[j,j] for j in range(n_comp)])
        t01 = 1/oldrates[0,1]
        t10 = 1/oldrates[1,0]
        ttimes = np.array([t01,t10])

    # get SNR
    snr01 = qp.getSNRhmm(M,mode1=0,mode2=1)
    snr12 = qp.getSNRhmm(M,mode1=1,mode2=2)
    snr02 = qp.getSNRhmm(M,mode1=0,mode2=2)
    SNRs = np.array([snr01,snr12,snr02])

    logprob,Q = M.decode(data.T)
    Qmean = np.mean(Q)
    P0 = np.sum(Q == 0)/Q.size
    P1 = np.sum(Q == 1)/Q.size
    P2 = np.sum(Q == 2)/Q.size

    if np.min(SNRs) < SNRmin and intTime <= np.min(lifetimes)/2 and P2 < P1:
        success = False
        srold = np.copy(sr)
        while not success:
            intTime *= 1.2*SNRmin/np.min(SNRs)
            print(f'\n\n\nNew integration time = {intTime} because SNR was {np.min(SNRs):.6}\n\n\n')
            data = np.vstack((res.signal.real,res.signal.imag)) * Vscale
            srold = np.copy(sr)
            data, sr = qp.BoxcarDownsample(data,intTime,sampleRateMHz,returnRate = True)
            # data = qp.uint16_to_mV(data)
            M = hmm.GaussianHMM(n_components=n_comp,covariance_type='full',init_params='s',n_iter=50,tol=0.005)
            with h5py.File(savefile,'r') as ff:
                f = ff[f'PHI{phi*1000:3.0f}']
                M.means_ = f[f'{i-1}'].attrs.get('HMMmeans_') if i != 0 else means
                M.covars_ = f[f'{i-1}'].attrs.get('HMMcovars_') if i != 0 else covars
            transmat = np.ones((n_comp,n_comp)) * 0.001*(i+1)
            np.fill_diagonal(transmat,1-(n_comp-1)*0.001*(i+1))
            M.transmat_ = transmat
            M.fit(data.T)
            # get SNR
            snr01 = qp.getSNRhmm(M,mode1=0,mode2=1)
            snr12 = qp.getSNRhmm(M,mode1=1,mode2=2)
            snr02 = qp.getSNRhmm(M,mode1=0,mode2=2)
            SNRs = np.array([snr01,snr12,snr02])
            if np.min(SNRs) > SNRmin:
                success = True
            elif intTime > np.min(lifetimes)/2 or intTime > np.min(ttimes):
                print('Integration time has grown too large.')
                break
            if sr == srold:
                print('stuck in loop, exiting')
                break
            srold = np.copy(sr)
    if intTime > np.min(lifetimes)/2 and i > 3:
        print('Integration time has grown too large.')
        break
    if P2 > P1:
        print('P2 became larger than P1!')
        break

    # plot the fit
    # h = qp.plotComplexHist(data[0],data[1],figsize=[4,4])
    figmeans,axmeans = plt.subplots(1,1)
    plotComplexHist(axmeans,data[0],data[1])
    colors = plt.cm.rainbow(np.linspace(0,1,n_comp))
    qp.make_ellipsesHMM(M,axmeans,colors)
    plt.xlabel('I [mV]')
    plt.ylabel('Q [mV]')
    plt.title('HMM fit | {:.2} MHz | {} dBm'.format(sr,p))
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,'HMMfits_{}_{}dBm.png'.format(i,p)))
    plt.close()

    # extract occupation
    logprob,Q = M.decode(data.T)
    nT = BoxcarMode(np.array(sim.nTrapped),intTime,sampleRateMHz,returnRate=False)
    Qmean = np.mean(Q)
    P0 = np.sum(Q == 0)/Q.size
    P1 = np.sum(Q == 1)/Q.size
    P2 = np.sum(Q == 2)/Q.size

    # plot a section of time series
    timefig, timeax = plt.subplots(2,1)
    time = np.arange(Q.size)/sr
    plotTimeSeries(timeax,data,nT,Q,time,1500,2000,zeroTime=True)
    plt.title('{:.2} MHz | {} dBm'.format(sr,p))
    # plotTimeSeries(mainax[1,i],data_p,np.array(sim_poisson.nTrapped),Qp,time*1000,10,12,zeroTime=False)
    # plt.suptitle(f'Poisson simulation | xne = {xne}')
    timeax[1].set_ylabel('n trapped')
    timeax[0].set_ylabel('Resonator response')
    timeax[1].set_xlabel('Time [$\mu$s]')
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,'TimeSeries__{}_{}dBm.png'.format(i,p)))
    plt.close()

    #####################
    # print metrics
    ###################

    # get the transition rates
    rates = qp.getTransRatesFromProb(sr,M.transmat_)
    
    print(f'\n{i}\n')
    print(f'there are {np.mean(sim.nTrapped):.4f} trapped QPs on average')
    print(f'HMM estimates {np.mean(Q):.4f} average trapped QPs')
    # rates = qp.getTransRatesFromProb(sampleRate,M.transmat_)
    print(f'the actual trap rate provided is {1/(1e6/(8.73e15*(xne))):.1f} Hz, while our estimate is {rates[0,1]:.1f} Hz')
    tmask = Q == nT
    # print(np.sum(tmask))
    F1 = []
    lPrec = []
    lRec = []
    lTP = []
    lFP = []
    lFN = []
    for k in range(n_comp):
        # print('\nk\n')
        tp = np.sum(Q[tmask] == k)
        lTP.append(tp)
        # print(f'tp = {tp}')
        fp = np.sum(Q == k) - tp
        lFP.append(fp)
        # print(f'fp = {fp}')
        fn = np.sum(nT == k) - tp
        lFN.append(fn)
        # print(f'fn = {fn}')
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        lPrec.append(prec)
        lRec.append(rec)
        F1.append(2*prec*rec/(prec+rec))
        # print(f'prec = {prec:.3f} | rec = {rec:.3f}')
    # print(f'\nF1s are {F1}')
    print(f'The composite F1 score for this test is {np.mean(F1):.3f}')
    print(f'composite prec = {np.mean(lPrec):.3f} | composite rec = {np.mean(lRec):.3f}')

    #####################
    # plot distribution
    #####################
    distfig,distax = plt.subplots(1,2)
    plt.suptitle('{:.2} MHz | {} dBm'.format(sr,p))
    hmmdist = qp.extractTimeBetweenTrapEvents(Q,time)
    dist = qp.extractTimeBetweenTrapEvents(nT,time)
    
    bins = np.linspace(0.95*min(np.min(hmmdist),np.min(dist)),
                       1.05*max(np.max(hmmdist),np.max(dist)),
                       80)
    countshmm, centershmm = plotTauDist(distax[0],hmmdist,bins=bins,color='purple',alpha=0.5,label='HMM')
    counts, centers = plotTauDist(distax[0],dist,bins=bins,color='orange',alpha=0.5,label='Simulated')
    distax[0].legend(loc = 'upper right')
    distax[0].set_xlabel('$\\tau$ [$\mu$s]')
    distax[0].set_ylabel('Counts')
    distax[0].set_yscale('log')
    # mainax[1,i].set_title('Poisson')
    
    ##################
    # plot scaled dist
    ##################

    logbins = np.logspace(np.log10(0.95*min(np.min(hmmdist),np.min(dist))),
                          np.log10(1.05*max(np.max(hmmdist),np.max(dist))),
                          40)
    countshmm2, centershmm2 = plotTauDistScaled(distax[1],hmmdist,bins=logbins,
                                                color='purple',alpha=0.5,label='HMM')
    counts2, centers2 = plotTauDistScaled(distax[1],dist,bins=logbins,
                                          color='orange',alpha=0.5,label='Simulated')
    distax[1].set_xscale('log')
    distax[1].legend(loc = 'upper left')
    distax[1].set_xlabel('$\\tau$ [$\mu$s]')
    distax[1].set_ylabel('$\\tau P(\\tau)$')
    taumean = np.mean(countshmm2)
    poisson_scaled = lambda x: (x/taumean)*np.exp(-x/taumean)
    poisson_model = poisson_scaled(centershmm2)
    Fidelity = np.sum(np.sqrt(centershmm2*countshmm2*poisson_model))/np.sum(centershmm2*countshmm2)
    # mainax[2,i].set_title('Poisson')
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,'distributions__{}_{}dBm.png'.format(i,p)))
    plt.close()

    

    # save all the information to aggregate file
    with h5py.File(savefile,'a') as ff:
        f = ff[f'PHI{phi*1000:3.0f}']
        fp = f.require_group('{}'.format(i))
        fp.create_dataset('Q',data = Q)
        fp.create_dataset('TrueOcc',data = nT)
        fp.create_dataset('data',data = data)
        fp.create_dataset('transitionRatesMHz',data = rates)
        fp.attrs.create('logprobQ',logprob)
        fp.attrs.create('mean',Qmean)
        fp.attrs.create('Truemean',np.mean(sim.nTrapped))
        fp.attrs.create('F1',np.mean(F1))
        fp.attrs.create('F1s',F1)
        fp.attrs.create('prec',np.mean(lPrec))
        fp.attrs.create('precs',lPrec)
        fp.attrs.create('rec',np.mean(lRec))
        fp.attrs.create('recs',lRec)
        fp.attrs.create('TPs',lTP)
        fp.attrs.create('FPs',lFP)
        fp.attrs.create('FNs',lFN)
        fp.attrs.create('poissonFidelity',Fidelity)
        fp.attrs.create('P0',P0)
        fp.attrs.create('P1',P1)
        fp.attrs.create('P2',P2)
        fp.attrs.create('SNRs',SNRs)
        fp.attrs.create('downsampleRateMHz',sr)
        fp.attrs.create('HMMmeans_',M.means_)
        fp.attrs.create('HMMstartprob_',M.startprob_)
        fp.attrs.create('HMMcovars_',M.covars_)
        fp.attrs.create('HMMtransmat_',M.transmat_)
        fp.attrs.create('LOpower',p)
        for key in metainfo:
            fp.attrs.create(key,metainfo[key])

    # save a copy of current means to use as estimate for next power
    means = np.copy(M.means_)
    covars = np.copy(M.covars_)




    ######################
    # old code
    ######################

#     M = hmm.GaussianHMM(
#         n_components=ncomp,
#         covariance_type='full',
#         min_covar=0.0001,
#         means_weight=1,
#         means_prior=np.array(means),
#         algorithm='viterbi',
#         random_state=None,
#         n_iter=20,
#         tol=0.005,
#         verbose=False,
#         params='stmc',
#         init_params='s',
#         implementation='log',
#     )
#     covars = np.zeros((ncomp,2,2))
#     for j in range(ncomp):
#         covars[j] = np.array([[1e-2,1e-6],[1e-6,1e-2]])
#     # covars = np.array([[1e-2,1e-6],[1e-6,1e-2]])
#     M.covars_ = covars
#     M.means_ = np.array(means)
#     transmat = np.ones((ncomp,ncomp)) * 0.001*(i+1)
#     np.fill_diagonal(transmat,1-(ncomp-1)*0.001*(i+1))
#     M.transmat_ = transmat

#     data = np.vstack((res.signal.real,res.signal.imag))
#     M.fit(data.T[:N//4])
#     logprob, Q = M.decode(data.T)

#     colors = plt.cm.rainbow(np.linspace(0,1,ncomp))
#     qp.make_ellipsesHMM(M,mainax[i,0],colors)

# #     ###########################
# #     # plot time series
# #     ##########################
    
#     time = np.arange(N)/sampleRate
# #     plotTimeSeries(mainax[1,i],data_p,np.array(sim_poisson.nTrapped),Qp,time*1000,10,12,zeroTime=False)
# #     plt.suptitle(f'Poisson simulation | xne = {xne}')
# #     ax2[1].set_ylabel('n trapped')
# #     ax2[0].set_ylabel('Resonator response')
# #     ax2[1].set_xlabel('Time [ms]')
# #     plt.show(block=False);

    #####################
    # print metrics
    ###################
    
    # print(f'\n{method}\n')
    # print(f'there are {np.mean(sim.nTrapped):.4f} trapped QPs on average')
    # print(f'HMM estimates {np.mean(Q):.4f} average trapped QPs')
    # print(M.means_)
    # rates = qp.getTransRatesFromProb(sampleRate,M.transmat_)
    # print(f'the actual trap rate provided is {1/(1e6/(8.73e15*(xne))):.1f} Hz, while our estimate is {rates[0,1]:.1f} Hz')
    # tmask = Q == nT
    # print(np.sum(tmask))
    # F1 = []
    # lPrec = []
    # lRec = []
    # for k in range(ncomp):
    #     print('\nk\n')
    #     tp = np.sum(Q[tmask] == k)
    #     print(f'tp = {tp}')
    #     fp = np.sum(Q == k) - tp
    #     print(f'fp = {fp}')
    #     fn = np.sum(nT == k) - tp
    #     print(f'fn = {fn}')
    #     prec = tp/(tp+fp)
    #     rec = tp/(tp+fn)
    #     lPrec.append(prec)
    #     lRec.append(rec)
    #     F1.append(2*prec*rec/(prec+rec))
    #     print(f'prec = {prec:.3f} | rec = {rec:.3f}')
    # print(f'\nF1s are {F1}')
    # print(f'The composite F1 score for this test is {np.mean(F1):.3f}')
    # print(f'composite prec = {np.mean(lPrec):.3f} | composite rec = {np.mean(lRec):.3f}')


    # #####################
    # # plot distribution
    # #####################
    
    # hmmdist = qp.extractTimeBetweenTrapEvents(Q,time*1e6)
    # dist = qp.extractTimeBetweenTrapEvents(sim.nTrapped,time*1e6)
    
    # bins = np.linspace(0.95*min(np.min(hmmdist),np.min(dist)),
    #                    1.05*max(np.max(hmmdist),np.max(dist)),
    #                    80)
    # countshmm, centershmm = plotTauDist(mainax[i,1],hmmdist,bins=bins,color='purple',alpha=0.5,label='HMM')
    # counts, centers = plotTauDist(mainax[i,1],dist,bins=bins,color='orange',alpha=0.5,label='Simulated')
    # if i == 0:
    #     mainax[i,1].legend(loc = 'upper right')
    # mainax[i,1].set_xlabel('$\\tau$ [$\mu$s]')
    # mainax[i,1].set_ylabel('Counts')
    # mainax[i,1].set_yscale('log')
    # # mainax[1,i].set_title('Poisson')
    
    # ##################
    # # plot scaled dist
    # ##################

    # logbins = np.logspace(np.log10(0.95*min(np.min(hmmdist),np.min(dist))),
    #                       np.log10(1.05*max(np.max(hmmdist),np.max(dist))),
    #                       40)
    # countshmm2, centershmm2 = plotTauDistScaled(mainax[i,2],hmmdist,bins=logbins,
    #                                             color='purple',alpha=0.5,label='HMM')
    # counts2, centers2 = plotTauDistScaled(mainax[i,2],dist,bins=logbins,
    #                                       color='orange',alpha=0.5,label='Simulated')
    # mainax[i,2].set_xscale('log')
    # if i == 0:
    #     mainax[i,2].legend(loc = 'upper left')
    # mainax[i,2].set_xlabel('$\\tau$ [$\mu$s]')
    # mainax[i,2].set_ylabel('$\\tau P(\\tau)$')
    # # mainax[2,i].set_title('Poisson')
    
# ###########################################
# # simulation finished, save and show the final product
# ###########################################
# plt.tight_layout()
# plt.savefig('comparison.pdf',dpi=300,transparent=True)
# plt.show()