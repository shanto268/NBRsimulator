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
Qi = 20000
Qe = 4000
photonRO = 25
delKappa = -0.5
Delta = 2.72370016e-23
T = 0.025
memory = 200e-6


##########################
# run the QP trapper for Poisson
##########################
args = {'N':N,'Lj':Lj,'xne':xne,'tauRelease':tauRelease,'tauCommon':tauCommon,'tauRare':tauRare,
        'tauRecomb':tauRecomb,'sampleRate':sampleRate,
             'phi':phi,'Delta':Delta,'T':T,'method':'poisson'}

now = perf_counter()
sim_poisson = QPtrapper(**args)
timer = perf_counter()-now
print('\nPOISSON\n')
print('qptrapper runtime: {} seconds for poisson simulation'.format(timer))
print('The average number of trapped QPs is {:.4}'.format(np.mean(sim_poisson.nTrapped)))

###########################
# run the resonator response for poisson
###########################

now2 = perf_counter()
resArgs = {'L':L,'C':C,'photonRO':photonRO,'Qi':Qi,'Qe':Qe,'sampleRate':sampleRate,'delKappa':delKappa}
res_poisson = NBResonator(sim_poisson,**resArgs)
duration2 = perf_counter() - now2
print('Resonator runtime: {}'.format(duration2))

##########################
# run QP trapper for Non-Markovian
##########################
args = {'N':N,'Lj':Lj,'xne':xne,'tauRelease':tauRelease,'tauCommon':tauCommon,'tauRare':tauRare,
        'tauRecomb':tauRecomb,'sampleRate':sampleRate,
             'phi':phi,'Delta':Delta,'T':T,'method':'non-markovian','memory':memory}

now = perf_counter()
sim_nm = QPtrapper(**args)
timer = perf_counter()-now
print('\nNON-MARKOVIAN\n')
print('qptrapper runtime: {} seconds for non-markovian simulation'.format(timer))
print('The average number of trapped QPs is {:.4}'.format(np.mean(sim_nm.nTrapped)))

###########################
# run the resonator response for non-markovian
###########################

now2 = perf_counter()
resArgs = {'L':L,'C':C,'photonRO':photonRO,'Qi':Qi,'Qe':Qe,'sampleRate':sampleRate,'delKappa':delKappa}
res_nm = NBResonator(sim_nm,**resArgs)
duration2 = perf_counter() - now2
print('Resonator runtime: {}'.format(duration2))

#################################
# Make complex histogram plot and get the means - poisson
################################

# plt.switch_backend('TkAgg')
fig1,ax1 = plt.subplots(1,1,num=1)
plotComplexHist(ax1,res_poisson.signal.real,res_poisson.signal.imag)
plt.title('Poisson simulation')
ax1.set_xlabel('I')
ax1.set_ylabel('Q')
means_p = plt.ginput(-1)

#################################
# Make complex histogram plot and get the means - non-markovian
################################

# plt.switch_backend('TkAgg')
fig5,ax5 = plt.subplots(1,1,num=5)
plotComplexHist(ax5,res_nm.signal.real,res_nm.signal.imag)
plt.title('Non-Markovian simulation')
ax5.set_xlabel('I')
ax5.set_ylabel('Q')
means_nm = plt.ginput(-1)

#########################
#########################
#########################
# POISSON
#########################
#########################
#########################


###################
# fit HMM poisson
###################
ncomp_p = len(means_p)

Mp = hmm.GaussianHMM(
    n_components=ncomp_p,
    covariance_type='full',
    min_covar=0.001,
    startprob_prior=1.0,
    transmat_prior=1.0,
    means_prior=0,
    means_weight=0,
    covars_prior=0.01,
    covars_weight=1,
    algorithm='viterbi',
    random_state=None,
    n_iter=10,
    tol=0.01,
    verbose=False,
    params='stmc',
    init_params='st',
    implementation='log',
)
covars = np.zeros((ncomp_p,2,2))
for i in range(ncomp_p):
    covars[i] = np.array([[1e-3,1e-6],[1e-6,1e-3]])
# M.transmat_ = np.array([[0.95,0.025,0.025],
#                         [0.03,0.94,0.03],
#                         [0.035,0.035,0.93]])
Mp.covars_ = covars
Mp.means_ = means_p

data_p = np.vstack((res_poisson.signal.real,res_poisson.signal.imag))
Mp.fit(data_p.T[:N//4])
logprob, Qp = Mp.decode(data_p.T)

colors = plt.cm.rainbow(np.linspace(0,1,ncomp_p))
qp.make_ellipsesHMM(Mp,ax1,colors)

###########################
# plot time series
##########################
# plt.switch_backend(r'module://matplotlib_inline.backend_inline')
time = np.arange(N)/sampleRate
fig2,ax2 = plt.subplots(2,1,sharex=True,num=2)
plotTimeSeries(ax2,data_p,np.array(sim_poisson.nTrapped),Qp,time*1000,10,12,zeroTime=False)
plt.suptitle(f'Poisson simulation | xne = {xne}')
ax2[1].set_ylabel('n trapped')
ax2[0].set_ylabel('Resonator response')
ax2[1].set_xlabel('Time [ms]')
plt.show(block=False);


#####################
# print metrics
###################
print('\nPOISSON\n')
print(f'there are {np.mean(sim_poisson.nTrapped):.4f} trapped QPs on average')
print(f'HMM estimates {np.mean(Qp):.4f} average trapped QPs')
print(Mp.means_)
rates_p = qp.getTransRatesFromProb(sampleRate,Mp.transmat_)
print(f'the actual trap rate provided is {1/(1e6/(8.73e15*(xne))):.1f} Hz, while our estimate is {rates_p[0,1]:.1f} Hz')


###################
# plot distribution
######################
fig3,ax3 = plt.subplots(1,1,num=3)
hmmdist = qp.extractTimeBetweenTrapEvents(Qp,time*1e6)
countshmm, centershmm = plotTauDist(ax3,hmmdist,bins=80,color='blue',alpha=0.3,label='HMM')

dist = qp.extractTimeBetweenTrapEvents(sim_poisson.nTrapped,time*1e6)
counts, centers = plotTauDist(ax3,dist,bins=80,color='grey',alpha=0.3,label='Simulated')
ax3.legend()
ax3.set_xlabel('$\tau$ between trap events [$\mu$s]')
ax3.set_ylabel('Counts')
ax3.set_title('Poisson')
plt.show(block=False)


##################
# plot scaled dist
##################

fig4,ax4 = plt.subplots(1,1,num=4)
logbins = np.logspace(np.log10(0.95*min(np.min(hmmdist),np.min(dist))),
                      np.log10(1.05*max(np.max(hmmdist),np.max(dist))),
                      80)
countshmm2, centershmm2 = plotTauDistScaled(ax4,hmmdist,bins=logbins,color='blue',alpha=0.5,label='HMM')
counts2, centers2 = plotTauDistScaled(ax4,dist,bins=logbins,color='grey',alpha=0.5,label='Simulated')
ax4.set_xscale('log')
ax4.legend()
ax4.set_xlabel('$\tau$ between trap events [$\mu$s]')
ax4.set_ylabel('$\tau P(\tau)$')
ax4.set_title('Poisson')
plt.show(block=False)

#########################
#########################
#########################
# Non-Markovian
#########################
#########################
#########################

###################
# fit HMM poisson
###################
ncomp_nm = len(means_nm)

Mnm = hmm.GaussianHMM(
    n_components=ncomp_nm,
    covariance_type='full',
    min_covar=0.001,
    startprob_prior=1.0,
    transmat_prior=1.0,
    means_prior=0,
    means_weight=0,
    covars_prior=0.01,
    covars_weight=1,
    algorithm='viterbi',
    random_state=None,
    n_iter=10,
    tol=0.01,
    verbose=False,
    params='stmc',
    init_params='st',
    implementation='log',
)
covars = np.zeros((ncomp_nm,2,2))
for i in range(ncomp_nm):
    covars[i] = np.array([[1e-3,1e-6],[1e-6,1e-3]])
# M.transmat_ = np.array([[0.95,0.025,0.025],
#                         [0.03,0.94,0.03],
#                         [0.035,0.035,0.93]])
Mnm.covars_ = covars
Mnm.means_ = means_nm

data_nm = np.vstack((res_nm.signal.real,res_nm.signal.imag))
Mnm.fit(data_nm.T[:N//4])
logprob, Qnm = Mnm.decode(data_nm.T)

colors = plt.cm.rainbow(np.linspace(0,1,ncomp_nm))
qp.make_ellipsesHMM(Mnm,ax5,colors)

###########################
# plot time series
##########################
# plt.switch_backend(r'module://matplotlib_inline.backend_inline')
time = np.arange(N)/sampleRate
fig6,ax6 = plt.subplots(2,1,sharex=True,num=6)
plotTimeSeries(ax6,data_nm,np.array(sim_nm.nTrapped),Qnm,time*1000,10,12,zeroTime=False)
plt.suptitle(f'Non-Markovian simulation | memory = {memory*1e6} $\mu$s')
ax6[1].set_ylabel('n trapped')
ax6[0].set_ylabel('Resonator response')
ax6[1].set_xlabel('Time [ms]')
plt.show(block=False);


#####################
# print metrics
###################
print('\nNON_MARKOVIAN\n')
print(f'there are {np.mean(sim_nm.nTrapped):.4f} trapped QPs on average')
print(f'HMM estimates {np.mean(Qnm):.4f} average trapped QPs')
print(Mnm.means_)
rates_nm = qp.getTransRatesFromProb(sampleRate,Mnm.transmat_)
print(f'the actual trap rate provided is {1/(1e6/(8.73e15*(xne))):.1f} Hz, while our estimate is {rates_nm[0,1]:.1f} Hz')


###################
# plot distribution
######################
fig7,ax7 = plt.subplots(1,1,num=7)
hmmdist_nm = qp.extractTimeBetweenTrapEvents(Qnm,time*1e6)
countshmm, centershmm = plotTauDist(ax7,hmmdist_nm,bins=80,color='blue',alpha=0.3,label='HMM')

dist_nm = qp.extractTimeBetweenTrapEvents(sim_nm.nTrapped,time*1e6)
counts, centers = plotTauDist(ax7,dist_nm,bins=80,color='grey',alpha=0.3,label='Simulated')
ax7.legend()
ax7.set_xlabel('$\tau$ between trap events [$\mu$s]')
ax7.set_ylabel('Counts')
ax7.set_title('Non-Markovian')
plt.show(block=False)


##################
# plot scaled dist
##################

fig8,ax8 = plt.subplots(1,1,num=8)
logbins = np.logspace(np.log10(0.95*min(np.min(hmmdist_nm),np.min(dist_nm))),
                      np.log10(1.05*max(np.max(hmmdist_nm),np.max(dist_nm))),
                      80)
countshmm2, centershmm2 = plotTauDistScaled(ax8,hmmdist_nm,bins=logbins,color='blue',alpha=0.5,label='HMM')
counts2, centers2 = plotTauDistScaled(ax8,dist_nm,bins=logbins,color='grey',alpha=0.5,label='Simulated')
ax8.set_xscale('log')
ax8.legend()
ax8.set_xlabel('$\tau$ between trap events [$\mu$s]')
ax8.set_ylabel('$\tau P(\tau)$')
ax8.set_title('Non-Markovian')
plt.show()