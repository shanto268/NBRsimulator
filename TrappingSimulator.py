#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:04:41 2020

A QP trapping simulator that uses a simplified model:
    
### TODO ###
    - document the simplified model
    - implement some initial population, why does it always seem to increase at the end?
    

@author: jamesfarmer
"""
# Imports
#import numpy as np
from numpy import sin, cos, pi, exp, sqrt, zeros, ones, arange, mean, inf, cumsum, array,arctanh
from numpy import sum as npsum
from numpy.random import random, choice, uniform, normal
from math import factorial
from scipy.stats import poisson
from nonmarkovian_helper import poisson_mem, poisson_var




class QPtrapper:
    
    def __init__(self,N=1000,Lj=3.6e-11,xne=8e-7,tauRelease=1e-5,tauCommon=1e-4,
                 tauRare=1e-2,tauRecomb=1e-4,sampleRate=300e6,
                 phi=0.4,Delta=2.72370016e-23,T=0.025,memory=50e-6,method='poisson',pairrateratio=0.25):

        self.N = int(N)
        self.tau = 1e6/(8.73e15*(xne))
        self.xne = xne
        self.xne0 = xne
        self.taupair = self.tau/pairrateratio
        self.tauR = tauRelease
        self.sampleRate = sampleRate
        self.dt = 1/sampleRate
        self.phi = phi
        self.Delta = Delta
        self.T = T
        self.de = pi*phi
#        if ((phi // 0.5) % 2):
#            self.de += pi
        self.cosd = cos(self.de)
        self.sind2 = sin(self.de)**2
        self.sin4 = sin(self.de/2)**4
        self.sin2 = sin(self.de/2)**2
        self.phi0 = 2.06783383*1e-15 #h/2e
        self.kb = 1.38064852e-23
        self.rphi0 = self.phi0/(2*pi) #hbar/2e
        self.Ne = int(8*(self.rphi0**2)/(self.Delta*Lj))
        self.Lj0 = Lj
        self.Lj = Lj/(1-sin(self.de/2)*arctanh(sin(self.de/2)))
        # self.Lj = Lj/self.cosd
        # self.tauCommon = tauCommon
        self.tauRare = tauRare
        self.tauRecomb = tauRecomb
        # alpha = self.Delta/(4*(self.rphi0**2))
        alpha = self.Delta/(2*(self.rphi0**2))
        self.L1 = alpha*(self.cosd/sqrt(1-self.sin2) + self.sind2/(4*sqrt(1-self.sin2)**3))
        
        # define the channels
        self.channels = []
        scale = self._dorokhov(0.99999)
        for _ in range(self.Ne):
            while True:
                x = uniform(low=0.0,high=1)
                y = random()*scale
                try:
                    if y < self._dorokhov(x):
                        self.channels.append({'t':x})
                        break
                except (ZeroDivisionError,RuntimeWarning):
                    pass
        # fill in some channel info
        for di in self.channels:
            di.update({'E': self._Ea(di['t'],self.Delta,self.de),
                       'bf': self._boltz(di['t'],self.Delta,self.de,T)})
            di.update({'Da':self.Delta - di['E']})
        
        
        # add the memory, only used for method = 'non-markovian'
        self.memory = memory
        
        self._getSwitchingEvents(method=method)

    def _dorokhov(self,tau):
        return 1/(tau*sqrt(1-tau))
    
    def _boltz(self,tau,delta=2.72370016e-23,de=pi/2,T=0.025):
        return exp(-self._Ea(tau,delta,de)/(self.kb*T))

    def _dorokhov_boltz(self,tau,Ne=680,delta=2.72370016e-23,de=pi/2,T=0.025):
        return Ne/(tau*sqrt(1-tau)) * exp(-self._Ea(tau,delta,de)/(self.kb*T))
    
    def _Ea(self,tau,delta=2.72370016e-23,de=pi/2):
        return delta*sqrt(1-tau*sin(de/2)**2)
    
    def _MC_doro(self,Ne=680,delta=2.72370016e-23,de=pi/2,T=0.025):
        scale = self._dorokhov_boltz(0.999999999,Ne,delta,de,T)
        while True:
            x = uniform(low=0.0,high=1)
            y = random()*scale
            try:
                if y < self._dorokhov_boltz(x,Ne,delta,de,T):
                    return x
                    break
            except (ZeroDivisionError,RuntimeWarning):
                pass
            
    def _Poisson(self,tau):
        return (self.dt/tau)*exp(-self.dt/tau)
    
    def _getSwitchingEvents(self,method='poisson'):
#        phaseFactor = sin(abs(self.de))
#        pTrap = phaseFactor*self._Poisson(self.tau)
#        pTrap = self._Poisson(self.tau)
#        pRelease = self._Poisson(self.tauR)
#        pCommon = self._Poisson(self.tauCommon)
        # pRare = self._Poisson(self.tauRare)
        cs = array([1,2,3,4,5,6,7,8,9],dtype=int)
        factorials = array([factorial(k) for k in cs],dtype=int)
        # pR = ((self.dt/self.tauR)**cs)*exp(-self.dt/self.tauR)/factorials
        ppT = ((self.dt/self.tau)**cs)*exp(-self.dt/self.tau)/factorials
        pRare = poisson(self.dt/self.tauRare)
        pR = poisson(self.dt/self.tauR)
        pT = poisson(self.dt/self.tau)
        # pCommon = ((self.dt/self.tauCommon)**cs)*exp(-self.dt/self.tauCommon)/factorials
        # pRare = ((self.dt/self.tauRare)**cs)*exp(-self.dt/self.tauRare)/factorials
        # pRecomb = ((self.dt/self.tauRecomb)**cs)*exp(-self.dt/self.tauRecomb)/factorials
        
        trappedChannels = []
        self.nTrapped = []
        lFreqFactors = []
        phi0 = self.phi0
        rphi0 = phi0/(2*pi)
        alpha = self.Delta/(2*(rphi0**2))
        # alpha = self.Delta/(4*(rphi0**2))
        self.freqFactors = zeros(self.N)
        self.nBulk = list(ones(3))
        self.bulkPop = []
        self.burstIndices = []
        self.burstSizes = []
        
        if method.lower() == 'poisson':
            trapdist = poisson(self.dt/self.tau)
            trapevents = trapdist.rvs(size=self.N)

            for n in range(self.N):
               # Produce trapping events
                k = trapevents[n]
                for _ in range(k):
                    boltzfacts = array([c['bf'] for c in self.channels])
                    boltzprob = boltzfacts/npsum(boltzfacts)
                    ch = choice(self.channels,p=boltzprob)
                    trappedChannels.append(ch)
                    self.channels.remove(ch)

               # Produce release events
                for _ in range(len(trappedChannels)):
                    # relmask = random() < pR
                    # k = cs[relmask][-1] if relmask.any() else 0
                    k = pR.rvs()
                    k = min((k,len(trappedChannels)))
                    for _ in range(k):
                        ch = choice(trappedChannels)
                        trappedChannels.remove(ch)
                        self.channels.append(ch)

               # Track changes
                self.nTrapped.append(len(trappedChannels))
                self.bulkPop.append(len(self.nBulk))

               # Calculate frequency shift terms for each time point -- Sum_i 1/L_i
                lFreqFactors.append([alpha*c['t']*(self.cosd/sqrt(1-c['t']*self.sin2) 
                                                   + c['t']*self.sind2/(4*sqrt(1-c['t']*self.sin2)**3))for c in trappedChannels])  
                self.freqFactors[n] += sum(lFreqFactors[n])
                
                
        elif method.lower() == 'non-markovian':
            delay = 0
            pm = poisson_mem()
            for n in range(self.N):
               # Produce trapping events
                # trapmask = random() < ppT*(1 - exp(-delay/self.memory))
                # k = cs[trapmask][-1] if trapmask.any() else 0
                k = pm.rvs(self.dt/self.tau, (delay+self.dt/2)/self.memory) # delay = 0 cause error for scipy rv_discrete, so add half a timestep so delay is always positive without skewing results.
                for _ in range(k):
                    boltzfacts = array([c['bf'] for c in self.channels])
                    boltzprob = boltzfacts/npsum(boltzfacts)
                    ch = choice(self.channels,p=boltzprob)
                    trappedChannels.append(ch)
                    self.channels.remove(ch)
                    # reset the delay to count time since last trap event
                    delay = 0

               # Produce release events
                for _ in range(len(trappedChannels)):
                    # relmask = random() < pR
                    # k = cs[relmask][-1] if relmask.any() else 0
                    k = pR.rvs()
                    k = min((k,len(trappedChannels)))
                    for _ in range(k):
                        ch = choice(trappedChannels)
                        trappedChannels.remove(ch)
                        self.channels.append(ch)

               # Track changes
                self.nTrapped.append(len(trappedChannels))
                self.bulkPop.append(len(self.nBulk))

               # Calculate frequency shift terms for each time point -- Sum_i 1/L_i
                lFreqFactors.append([alpha*c['t']*(self.cosd/sqrt(1-c['t']*self.sin2) 
                                                   + c['t']*self.sind2/(4*sqrt(1-c['t']*self.sin2)**3))for c in trappedChannels])  
                self.freqFactors[n] += sum(lFreqFactors[n])  

                # increment delay
                delay += self.dt
                
        elif method.lower() == 'pairwise':
            trapdist = poisson(self.dt/self.tau)
            trapevents = trapdist.rvs(size=self.N)
            pairdist = poisson(self.dt/self.taupair)
            pairevents = pairdist.rvs(size=self.N)

            for n in range(self.N):
               # Produce trapping events
                k = trapevents[n]
                for _ in range(k):
                    boltzfacts = array([c['bf'] for c in self.channels])
                    boltzprob = boltzfacts/npsum(boltzfacts)
                    ch = choice(self.channels,p=boltzprob)
                    trappedChannels.append(ch)
                    self.channels.remove(ch)
                    
                # Produce paired events
                k = pairevents[n]
                for _ in range(k):
                    boltzfacts = array([c['bf'] for c in self.channels])
                    boltzprob = boltzfacts/npsum(boltzfacts)
                    lch = choice(self.channels,p=boltzprob,size=2,replace=False)
                    for ch in lch:
                        trappedChannels.append(ch)
                        self.channels.remove(ch)

               # Produce release events
                for _ in range(len(trappedChannels)):
                    # relmask = random() < pR
                    # k = cs[relmask][-1] if relmask.any() else 0
                    k = pR.rvs()
                    k = min((k,len(trappedChannels)))
                    for _ in range(k):
                        ch = choice(trappedChannels)
                        trappedChannels.remove(ch)
                        self.channels.append(ch)

               # Track changes
                self.nTrapped.append(len(trappedChannels))
                self.bulkPop.append(len(self.nBulk))

               # Calculate frequency shift terms for each time point -- Sum_i 1/L_i
                lFreqFactors.append([alpha*c['t']*(self.cosd/sqrt(1-c['t']*self.sin2) 
                                                   + c['t']*self.sind2/(4*sqrt(1-c['t']*self.sin2)**3))for c in trappedChannels])  
                self.freqFactors[n] += sum(lFreqFactors[n]) 
                
        
        elif method.lower() == 'burst':
            pB = poisson_var()

            for n in range(self.N):
                
               # Produce rare QP generation events -- such as cosmic ray bursts
                # raremask = random() < pRare
                # k = cs[raremask][-1] if raremask.any() else 0
                k = pRare.rvs()
                for _ in range(k):
                    self.burstIndices.append(n)
                    burstsize = abs(normal(loc=3e-6,scale=5e-7))
                    self.burstSizes.append(burstsize)
                self.xne = self.xne0 + npsum([X*exp(-(n-N)*self.dt/self.tauRecomb) 
                                         for X,N in zip(self.burstSizes,self.burstIndices)])
                
                        
               # Produce trapping events
                self.tau = 1e6/(8.73e15*(self.xne))
                k = pB.rvs(self.dt/self.tau)
                for _ in range(k):
                    boltzfacts = array([c['bf'] for c in self.channels])
                    boltzprob = boltzfacts/npsum(boltzfacts)
                    ch = choice(self.channels,p=boltzprob)
                    trappedChannels.append(ch)
                    self.channels.remove(ch)

               # Produce release events
                for _ in range(len(trappedChannels)):
                    # relmask = random() < pR
                    # k = cs[relmask][-1] if relmask.any() else 0
                    k = pR.rvs()
                    k = min((k,len(trappedChannels)))
                    for _ in range(k):
                        ch = choice(trappedChannels)
                        trappedChannels.remove(ch)
                        self.channels.append(ch)

               # Track changes
                self.nTrapped.append(len(trappedChannels))
                self.bulkPop.append(len(self.nBulk))

               # Calculate frequency shift terms for each time point -- Sum_i 1/L_i
                lFreqFactors.append([alpha*c['t']*(self.cosd/sqrt(1-c['t']*self.sin2) 
                                                   + c['t']*self.sind2/(4*sqrt(1-c['t']*self.sin2)**3))for c in trappedChannels])  
                self.freqFactors[n] += sum(lFreqFactors[n])  
        
        else:
            raise NotImplementedError('Please use method from {"poisson","non-markovian","pairwise","burst"}')

if __name__ == '__main__':
    from time import perf_counter
    import matplotlib.pyplot as plt
    duration = 0.05 # seconds to record data
    sampleRate = 5e6
    N = int(duration*sampleRate)
    xne = 8e-7
    tauRelease = 1e-5
    tauCommon = 4e-4
    tauRare = 1e-2
    tauRecomb = 2e-3
    phi = 0.45
    Lj = 21.2e-12
    
    args = {'N':N,'Lj':Lj,'xne':xne,'tauRelease':tauRelease,'tauCommon':tauCommon,'tauRare':tauRare,
            'tauRecomb':tauRecomb,'sampleRate':sampleRate,
                 'phi':phi,'Delta':2.72370016e-23,'T':0.025}
    
    now = perf_counter()
    test = QPtrapper(**args)
    timer = perf_counter()-now
    print('qptrapper runtime: {} seconds'.format(timer))
    print('The average number of trapped QPs is {:.4}'.format(mean(test.nTrapped)))
    
    time = arange(N)/sampleRate
    
    h,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(time*1e3,test.nTrapped,'-g')
    plt.suptitle('actual number of trapped QPs')
    ax[0].set_ylabel('n trapped')
    ax[1].plot(time*1e3,test.bulkPop,'-b')
    ax[1].set_ylabel('number in bulk')
    ax[1].set_xlabel('Time [ms]')