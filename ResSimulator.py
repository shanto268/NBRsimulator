#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:50:59 2020

@author: jamesfarmer
"""
import sys
sys.path.append(r"/Users/jamesfarmer/ResPy/")
sys.path.append(r"/Users/jamesfarmer/lflPython/")
from Resonator import Resonator
import numpy as np
from scipy.signal import butter,lfilter


class NBResonator():
    
    def __init__(self,trapper,L=1e-9,C=0.7e-12,photonRO=1,photonNoise=0.5,Qi=5e4,Qe=5e4,sampleRate=300e6,delKappa = -0.5,fd=None):
        self.port1 = Resonator('R')
        self.N = trapper.N
        self.Lj0 = trapper.Lj0
        self.Lj = trapper.Lj
        self.f0 = 1/(2*np.pi*np.sqrt((L + self.Lj)*C))
        self.q0 = self.Lj/(L+self.Lj)
        self.photonRO = photonRO
        self.Qi = Qi
        self.Qe = Qe
        self.Qt = Qi*Qe / (Qi + Qe)
        self.sampleRate = sampleRate
        self.kappa = 2*np.pi*self.f0/self.Qt
        self.kappa_e = 2*np.pi*self.f0/self.Qe
        self.fwhm = self.f0/self.Qt
        self.diameter = 2*self.Qt/self.Qe
        self.f = self.f0 + delKappa*self.kappa/(2*np.pi) # the resonator drive frequency for measurement
        if fd != None:
            self.f = fd
        self.f_form = self.f0 - (self.q0*self.f0*self.Lj*np.array(trapper.freqFactors)/2)
        self.f_shift = self.q0*self.f0*self.Lj*trapper.L1/2
#        self.SNR = photonRO*self.kappa/(4*sampleRate)*(1 - 4*self.Qt/self.Qe * (1-self.Qt/self.Qe))
        self.pSNR = photonRO*self.kappa/(((1 + np.sqrt(self.kappa/self.kappa_e))**2)*(0.5+photonNoise)*sampleRate)
        self.pSNRdB = 10*np.log10(self.pSNR)
        
#        self.sigma = self.diameter/(2*np.sqrt(2*self.SNR))
        self.sigma = 1/np.sqrt(2*self.pSNR)
        self.complex_noise = np.empty(self.N,dtype=complex)
        self.complex_noise.real = np.random.normal(scale = self.sigma,size=self.N)
        self.complex_noise.imag = np.random.normal(scale = self.sigma,size=self.N)
#        self.complex_noise = np.array(
#                [np.random.normal() * np.exp(1j*np.random.uniform(low=0,high=2*np.pi)) for i in range(self.N)
#                ])
        #get the response at given frequency
#        self._get_clean_response(self.w)
        kwargs = dict(fr = self.f_form,
                      Ql = self.Qt,
                      Qc = self.Qe,
                      a = 1.,
                      alpha = 0.,
                      delay = 0.)
        signal = self.port1._S11_directrefl(self.f,**kwargs)
        # self.signal = self.butter_lowpass_filter(signal - signal[0],self.kappa/10,self.sampleRate)+signal[0]
        self.signal = signal
        self.signal += self.complex_noise
#        self.signal = self.port1._S11_directrefl(self.f,**kwargs)
        self.dParams = {'fd': self.f,
                        'f0': self.f0,
                        'Qt': self.Qt,
                        'Qi': self.Qi,
                        'Qe': self.Qe,
                        'N': self.N,
                        'q': self.q0,
                        'photonRO': self.photonRO,
                        'sampleRate': self.sampleRate,
                        'kappa': self.kappa,
                        'fwhm': self.fwhm,
                        'diameter': self.diameter,
                        'freq_shift': self.f_shift,
                        'SNR': self.pSNR,
                        'SNRdB': self.pSNRdB,
                        'sigma': self.sigma}
        
    def butter_lowpass(self,cutoff, fs, order=1):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(self,data, cutoff, fs, order=1):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
        
        
if __name__ == '__main__':
    from TrappingSimulator import QPtrapper
    from time import perf_counter
    import matplotlib.pyplot as plt
    from hmmlearn import hmm
    import fitTools.quasiparticleFunctions as qp
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
    photonRO = 10
    delKappa = -0.5
    
    args = {'N':N,'Lj':Lj,'xne':xne,'tauRelease':tauRelease,'tauCommon':tauCommon,'tauRare':tauRare,
            'tauRecomb':tauRecomb,'sampleRate':sampleRate,
                 'phi':phi,'Delta':2.72370016e-23,'T':0.025}
    
    now = perf_counter()
    test = QPtrapper(**args)
    timer = perf_counter()-now
    print('qptrapper runtime: {} seconds'.format(timer))
    print('The average number of trapped QPs is {:.4}'.format(np.mean(test.nTrapped)))
    
    
    now2 = perf_counter()
    resArgs = {'L':L,'C':C,'photonRO':photonRO,'Qi':Qi,'Qe':Qe,'sampleRate':sampleRate,'delKappa':delKappa}
    res = NBResonator(test,**resArgs)
    duration2 = perf_counter() - now2
    print('Resonator runtime: {}'.format(duration2))
    
    # avgTime = 4*res.Qt*50/(photonRO*2*np.pi*res.f0)
    # nAvg = int(max(avgTime*sampleRate,1))
    # from scipy.signal import windows,convolve
    # window = windows.hann(nAvg)
    # rhann = convolve(res.signal.real,window,mode='same')/sum(window)
    # ihann = convolve(res.signal.imag,window,mode='same')/sum(window)
    
    # plt.hist2d(res.signal.real,res.signal.imag,bins=(50,50));plt.show()
    qp.plotComplexHist(res.signal.real,res.signal.imag);plt.show()
    
    time = np.arange(N)/sampleRate
    h,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(time*1e3,test.nTrapped,'-g')
    plt.suptitle(f'Simulation with x = {xne}')
    ax[0].set_ylabel('n trapped')
    ax[1].plot(time*1e3,res.signal.real,c='orange',label='I')
    ax[1].plot(time*1e3,res.signal.imag,c='purple',label='Q')
    ax[1].legend()
    ax[1].set_ylabel('Resonator response')
    ax[1].set_xlabel('Time [ms]')
    
    
    
    
#    kwargs = dict(fr = res.f0,
#                  Ql = res.Qt,
#                  Qc = Qe,
#                  a = 1.,
#                  alpha = 0.,
#                  delay = 0.)
#    freq = np.linspace(res.f0-2*res.gamma/np.pi,res.f0+2*res.gamma/np.pi,3000)
#    S11 = res.complex_noise + res.port1._S11_directrefl(freq,**kwargs)
#    res.port1.add_data(freq,S11)
#    plt.rcParams["figure.figsize"] = [10,5]
#    port1.plotrawdata()
#    f0 = res.f_form*1e-9
#    f = res.f * 1e-9
#    
#    signal = res.signal.real
#    
#    time = np.arange(N)/sampleRate
#    
#    fig, axs = plt.subplots(4, 1, constrained_layout=True,figsize=[12,12])
#    axs[0].plot(time, f0, '.r')
#    axs[0].set_title('resonant frequency')
#    axs[0].set_ylabel('f0 [GHz]')
#    axs[1].plot(time,signal,'.b')
#    axs[1].set_title('real resonse at {:.7} GHz'.format(f))
#    axs[1].set_ylabel('Re[S11]')
#    axs[2].plot(time,test.nTrapped,'-g')
#    axs[2].set_title('actual number of trapped QPs')
#    axs[2].set_ylabel('n trapped')
#    axs[2].set_xlabel('Time [s]')
#    hist = axs[3].hist(signal,bins=50,density=True)
#    axs[3].set_title('Histogram of Re[S11] at {:.7} GHz'.format(f))
#    axs[3].set_xlabel('Re[S11]')
#    axs[3].set_ylabel('p(Re[S11])')
#    fig.suptitle('QP trapping simulator',fontsize=16)
#    print('The average number of trapped QPs is {:.4}'.format(np.mean(test.nTrapped)))