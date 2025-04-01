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



#########################
#########################
#########################
# POISSON
#########################
#########################
#########################


##########################
# run the QP trapper for Poisson
##########################
args = {'N':N,'Lj':Lj,'xne':xne,'tauRelease':tauRelease,'tauCommon':tauCommon,'tauRare':tauRare,
        'tauRecomb':tauRecomb,'sampleRate':sampleRate,
             'phi':phi,'Delta':Delta,'T':T,'method':'poisson'}

now = perf_counter()
sim_poisson = QPtrapper(**args)
timer = perf_counter()-now
print('qptrapper runtime: {} seconds for poisson simulation'.format(timer))
print('The average number of trapped QPs is {:.4}'.format(np.mean(sim_poisson.nTrapped)))

# ##########################
# # run QP trapper for Non-Markovian
# ##########################
# args = {'N':N,'Lj':Lj,'xne':xne,'tauRelease':tauRelease,'tauCommon':tauCommon,'tauRare':tauRare,
#         'tauRecomb':tauRecomb,'sampleRate':sampleRate,
#              'phi':phi,'Delta':Delta,'T':T,'method':'non-markovian'}

# now = perf_counter()
# sim_nm = QPtrapper(**args)
# timer = perf_counter()-now
# print('qptrapper runtime: {} seconds for poisson simulation'.format(timer))
# print('The average number of trapped QPs is {:.4}'.format(np.mean(sim_nm.nTrapped)))

###########################
# run the resonator response
###########################

now2 = perf_counter()
resArgs = {'L':L,'C':C,'photonRO':photonRO,'Qi':Qi,'Qe':Qe,'sampleRate':sampleRate,'delKappa':delKappa}
res_poisson = NBResonator(sim_poisson,**resArgs)
duration2 = perf_counter() - now2
print('Resonator runtime: {}'.format(duration2))

#################################
# Make complex histogram plot and get the means
################################

# plt.switch_backend('TkAgg')
fig1,ax1 = plt.subplots(1,1,num=1)
plotComplexHist(ax1,res_poisson.signal.real,res_poisson.signal.imag)
plt.title('Poisson simulation')
ax1.set_xlabel('I')
ax1.set_ylabel('Q')
plt.tight_layout()
means = plt.ginput(-1)
# M.means_ = np.array([[-0.18,0.85],[0.12,0.79],[0.42,0.6]])


###################
# fit HMM
###################
ncomp = len(means)

M = hmm.GaussianHMM(
    n_components=ncomp,
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
covars = np.zeros((ncomp,2,2))
for i in range(ncomp):
    covars[i] = np.array([[1e-3,1e-6],[1e-6,1e-3]])
# M.transmat_ = np.array([[0.95,0.025,0.025],
#                         [0.03,0.94,0.03],
#                         [0.035,0.035,0.93]])
M.covars_ = covars
M.means_ = means

data = np.vstack((res_poisson.signal.real,res_poisson.signal.imag))
M.fit(data.T[:N//4])
logprob, Q = M.decode(data.T)

colors = plt.cm.rainbow(np.linspace(0,1,ncomp))
qp.make_ellipsesHMM(M,ax1,colors)
plt.tight_layout()

###########################
# plot time series
##########################
# plt.switch_backend(r'module://matplotlib_inline.backend_inline')
time = np.arange(N)/sampleRate
fig2,ax2 = plt.subplots(2,1,sharex=True,num=2)
plotTimeSeries(ax2,data,np.array(sim_poisson.nTrapped),Q,time*1000,10,12,zeroTime=False)
plt.suptitle(f'Poisson simulation | xne = {xne}')
ax2[1].set_ylabel('n trapped')
ax2[0].set_ylabel('Resonator response')
ax2[1].set_xlabel('Time [ms]')
plt.tight_layout()
plt.show(block=False);


#####################
# print metrics
###################

print(f'there are {np.mean(sim_poisson.nTrapped):.4f} trapped QPs on average')
print(f'HMM estimates {np.mean(Q):.4f} average trapped QPs')
print(M.means_)
rates = qp.getTransRatesFromProb(sampleRate,M.transmat_)
print(f'the actual trap rate provided is {1/(1e6/(8.73e15*(xne))):.1f} Hz, while our estimate is {rates[0,1]:.1f} Hz')


###################
# plot distribution
######################
fig3,ax3 = plt.subplots(1,1,num=3)
hmmdist = qp.extractTimeBetweenTrapEvents(Q,time*1e6)
countshmm, centershmm = plotTauDist(ax3,hmmdist,bins=80,color='blue',alpha=0.3,label='HMM')

dist = qp.extractTimeBetweenTrapEvents(sim_poisson.nTrapped,time*1e6)
counts, centers = plotTauDist(ax3,dist,bins=80,color='grey',alpha=0.3,label='Simulated')
ax3.legend()
ax3.set_xlabel('Delay between trap events [$\mu$s]')
ax3.set_ylabel('Counts')
plt.tight_layout()
plt.show(block=False)


##################
# plot scaled dist
##################

fig4,ax4 = plt.subplots(1,1,num=4)
countshmm2, centershmm2 = plotTauDistScaled(ax4,hmmdist,bins=80,color='blue',alpha=0.3,label='HMM')
counts2, centers2 = plotTauDistScaled(ax4,dist,bins=80,color='grey',alpha=0.3,label='Simulated')
ax4.legend()
ax3.set_xlabel('Delay between trap events [$\mu$s]')
ax3.set_ylabel('Counts')
plt.tight_layout()
plt.show()

#########################
#########################
#########################
# Non-Markovian
#########################
#########################
#########################