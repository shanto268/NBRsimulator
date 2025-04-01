import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./JTF.mplstyle')
from plotting_functions import *
import h5py
import os

phi = 0.47

#############
# files
############
SNRwithpurenoise = r'/Users/jamesfarmer/Documents/LFL/HMM_SNR_SIM_154purenoise/dataset.hdf5'
SNRmatchedBootstrap = r'/Users/jamesfarmer/Documents/LFL/HMM_Matched/dataset.hdf5'
Bootstrap = r'/Users/jamesfarmer/Documents/LFL/HMM_Bootstrap_sim/dataset.hdf5'
Bootstrap1MHz = r'/Users/jamesfarmer/Documents/LFL/HMM_Bootstrap_1MHz/dataset.hdf5'
SNRmatchedBootstrap1MHz = r'/Users/jamesfarmer/Documents/LFL/HMM_Matched_1MHz/dataset.hdf5'

files = [SNRmatchedBootstrap1MHz,Bootstrap1MHz]
names = ['matched','bootstrap']
# files = [SNRwithpurenoise,]
# names = ['SNR',]


figpath = r'/Users/jamesfarmer/Documents/LFL/HMM_SNR_figures'
if not os.path.exists(figpath):
    os.makedirs(figpath)
# plots to make:
# F1,prec,recall vs power (vs SNR)
# <n>, P0,P1,P2 vs power (vs SNR)
# rates vs power (vs SNR)

figmain,axmain = plt.subplots(1,1)
figmean,axmean = plt.subplots(1,1)
figrates,axrates = plt.subplots(1,1)

for savefile,name in zip(files,names):

    # make empty lists
    rates = []
    mean = []
    truemean = []
    F1 = []
    F1s = []
    prec = []
    precs = []
    rec = []
    recs = []
    TPs = []
    FPs = []
    FNs = []
    Fidelity = []
    P0 = []
    P1 = []
    P2 = []
    SNRs = []
    sr = []
    pow = []

    # pull data
    with h5py.File(savefile,'r') as ff:
        f = ff[f'PHI{phi*1000:3.0f}']
        for key in f:
            fp = f[key]
            # fp.create_dataset('Q')
            # fp.create_dataset('TrueOcc')
            # fp.create_dataset('data')
            rates.append(fp['transitionRatesMHz'][:])
            # fp.attrs.get('logprobQ')
            mean.append(fp.attrs.get('mean'))
            truemean.append(fp.attrs.get('Truemean'))
            F1.append(fp.attrs.get('F1'))
            F1s.append(fp.attrs.get('F1s'))
            prec.append(fp.attrs.get('prec'))
            precs.append(fp.attrs.get('precs'))
            rec.append(fp.attrs.get('rec'))
            recs.append(fp.attrs.get('recs'))
            TPs.append(fp.attrs.get('TPs'))
            FPs.append(fp.attrs.get('FPs'))
            FNs.append(fp.attrs.get('FNs'))
            Fidelity.append(fp.attrs.get('poissonFidelity'))
            P0.append(fp.attrs.get('P0'))
            P1.append(fp.attrs.get('P1'))
            P2.append(fp.attrs.get('P2'))
            SNRs.append(fp.attrs.get('SNRs'))
            sr.append(fp.attrs.get('downsampleRateMHz'))
            # fp.attrs.get('HMMmeans_')
            # fp.attrs.get('HMMstartprob_')
            # fp.attrs.get('HMMcovars_')
            # fp.attrs.get('HMMtransmat_')
            pow.append(fp.attrs.get('LOpower'))
    
    # make arrays
    rates = np.array(rates)
    mean = np.array(mean)
    truemean = np.array(truemean)
    F1 = np.array(F1)
    F1s = np.array(F1s)
    prec = np.array(prec)
    precs = np.array(precs)
    rec = np.array(rec)
    recs = np.array(recs)
    TPs = np.array(TPs)
    FPs = np.array(FPs)
    FNs = np.array(FNs)
    Fidelity = np.array(Fidelity)
    P0 = np.array(P0)
    P1 = np.array(P1)
    P2 = np.array(P2)
    SNRs = np.array(SNRs)
    sr = np.array(sr)
    pow = np.array(pow)

    # sorting by power
    sortind = np.argsort(pow)

    # Plot 1, F1 metrics
    fig,ax = plt.subplots(1,1)
    ax.plot(pow[sortind],F1[sortind],marker='o',label='F1')
    ax.plot(pow[sortind],prec[sortind],marker='o',label='Prec')
    ax.plot(pow[sortind],rec[sortind],marker='o',label='Rec')
    ax.legend()
    # ax.set_title(f'{name}')
    ax.set_xlabel('Power [dBm]')

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    c2 = 'tab:grey'
    ax2.set_ylabel('Sample Rate [MHz]', color=c2)  # we already handled the x-label with ax1
    ax2.plot(pow[sortind],sr[sortind],ls='dashed',c=c2)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,f'F1_pow_{name}.pdf'))

    ######
    # add to main plot
    axmain.plot(pow[sortind],F1[sortind],marker='o',label=f'{name}')

    # Plot 2, mean and prob
    fig,ax = plt.subplots(1,1)
    # ax.plot(pow[sortind],sr[sortind],ls='dashed',c='grey')
    ax.semilogy(pow[sortind],mean[sortind],marker='o',label='Mean')
    ax.semilogy(pow[sortind],P0[sortind],marker='o',label='P0')
    ax.semilogy(pow[sortind],P1[sortind],marker='o',label='P1')
    ax.semilogy(pow[sortind],P2[sortind],marker='o',label='P2')
    ax.legend()
    # ax.set_title(f'{name}')
    ax.set_xlabel('Power [dBm]')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Sample Rate [MHz]', color=c2)  # we already handled the x-label with ax1
    ax2.plot(pow[sortind],sr[sortind],ls='dashed',c=c2)
    ax2.tick_params(axis='y', labelcolor=c2)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,f'mean_pow_{name}.pdf'))

    #######
    # add to main
    lm = axmean.plot(pow[sortind],mean[sortind],ls = '',marker='o',label=f'{name}')
    axmean.plot(pow[sortind],truemean[sortind],c=lm[0].get_color())

    # Plot 3, rates
    g01 = rates[:,0,1]
    g12 = rates[:,1,2]
    # g02 = rates[:,0,2]
    # g20 = rates[:,2,0]
    g21 = rates[:,2,1]
    g10 = rates[:,1,0]
    fig,ax = plt.subplots(1,1)
    # ax.plot(pow[sortind],sr[sortind],ls='dashed',c='grey')
    ax.semilogy(pow[sortind],g01[sortind],marker='o',label='$\\Gamma_{01}$')
    ax.semilogy(pow[sortind],g12[sortind],marker='o',label='$\\Gamma_{12}$')
    # ax.semilogy(pow[sortind],g02[sortind],marker='o',label='$\\Gamma_{02}$')
    # ax.semilogy(pow[sortind],g20[sortind],marker='o',label='$\\Gamma_{20}$')
    ax.semilogy(pow[sortind],g21[sortind],marker='o',label='$\\Gamma_{21}$')
    ax.semilogy(pow[sortind],g10[sortind],marker='o',label='$\\Gamma_{10}$')
    ax.legend()
    # ax.set_title(f'{name}')
    ax.set_xlabel('Power [dBm]')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Sample Rate [MHz]', color=c2)  # we already handled the x-label with ax1
    ax2.plot(pow[sortind],sr[sortind],ls='dashed',c=c2)
    ax2.tick_params(axis='y', labelcolor=c2)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,f'rates_pow_{name}.pdf'))

    #####
    # add to main
    axrates.plot(pow[sortind],g01[sortind],marker='o',label=f'{name} trap')
    axrates.plot(pow[sortind],g10[sortind],marker='o',label=f'{name} release')



    # sort by SNR
    minSNR = np.min(SNRs,axis=1)
    sortsnr = np.argsort(minSNR)

    # Plot 4, F1 metrics vs SNR
    fig,ax = plt.subplots(1,1)
    ax.plot(minSNR[sortsnr],F1[sortsnr],marker='o',label='F1')
    ax.plot(minSNR[sortsnr],prec[sortsnr],marker='o',label='Prec')
    ax.plot(minSNR[sortsnr],rec[sortsnr],marker='o',label='Rec')
    ax.axvline(3,ls='dashdot',c='black',alpha=0.7)
    ax.legend()
    # ax.set_title(f'{name}')
    ax.set_xlabel('SNR')
    ax.set_ylabel('Score')
    # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Sample Rate [MHz]', color=c2)  # we already handled the x-label with ax1
    # ax2.plot(minSNR[sortsnr],sr[sortsnr],ls='dashed',c=c2)
    # ax2.tick_params(axis='y', labelcolor=c2)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,f'F1_snr_{name}.pdf'))

    # Plot 5, mean and prob vs snr
    fig,ax = plt.subplots(1,1)
    # ax.plot(minSNR[sortsnr],sr[sortsnr],ls='dashed',c='grey')
    ax.semilogy(minSNR[sortsnr],mean[sortsnr],marker='o',label='Mean')
    ax.semilogy(minSNR[sortsnr],P0[sortsnr],marker='o',label='P0')
    ax.semilogy(minSNR[sortsnr],P1[sortsnr],marker='o',label='P1')
    ax.semilogy(minSNR[sortsnr],P2[sortsnr],marker='o',label='P2')
    ax.axvline(3,ls='dashdot',c='black',alpha=0.7)
    ax.legend()
    # ax.set_title(f'{name}')
    ax.set_xlabel('SNR')
    # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Sample Rate [MHz]', color=c2)  # we already handled the x-label with ax1
    # ax2.plot(minSNR[sortsnr],sr[sortsnr],ls='dashed',c=c2)
    # ax2.tick_params(axis='y', labelcolor=c2)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,f'mean_snr_{name}.pdf'))

    # Plot 6, rates vs snr
    g01 = rates[:,0,1]
    g12 = rates[:,1,2]
    # g02 = rates[:,0,2]
    # g20 = rates[:,2,0]
    g21 = rates[:,2,1]
    g10 = rates[:,1,0]
    fig,ax = plt.subplots(1,1)
    # ax.plot(minSNR[sortsnr],sr[sortsnr],ls='dashed',c='grey')
    ax.semilogy(minSNR[sortsnr],g01[sortsnr],marker='o',label='$\\Gamma_{01}$')
    ax.semilogy(minSNR[sortsnr],g12[sortsnr],marker='o',label='$\\Gamma_{12}$')
    # ax.semilogy(minSNR[sortsnr],g02[sortsnr],marker='o',label='$\\Gamma_{02}$')
    # ax.semilogy(minSNR[sortsnr],g20[sortsnr],marker='o',label='$\\Gamma_{20}$')
    ax.semilogy(minSNR[sortsnr],g21[sortsnr],marker='o',label='$\\Gamma_{21}$')
    ax.semilogy(minSNR[sortsnr],g10[sortsnr],marker='o',label='$\\Gamma_{10}$')
    ax.axvline(3,ls='dashdot',c='black',alpha=0.7)
    ax.legend()
    # ax.set_title(f'{name}')
    ax.set_xlabel('SNR')
    # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Sample Rate [MHz]', color=c2)  # we already handled the x-label with ax1
    # ax2.plot(minSNR[sortsnr],sr[sortsnr],ls='dashed',c=c2)
    # ax2.tick_params(axis='y', labelcolor=c2)
    plt.tight_layout()
    plt.savefig(os.path.join(figpath,f'rates_snr_{name}.pdf'))


plt.figure(figmain)
axmain.legend()
# ax.set_title(f'{name}')
axmain.set_xlabel('Power [dBm]')
axmain.set_ylabel('F1 score')
plt.tight_layout()
plt.savefig(os.path.join(figpath,f'F1_main.pdf'))

plt.figure(figmean)
axmean.legend()
# ax.set_title(f'{name}')
axmean.set_xlabel('Power [dBm]')
axmean.set_ylabel('Mean')
plt.tight_layout()
plt.savefig(os.path.join(figpath,f'mean_main.pdf'))

plt.figure(figrates)
axrates.legend()
# ax.set_title(f'{name}')
axrates.set_xlabel('Power [dBm]')
axrates.set_ylabel('Rates [MHz]')
axrates.axhline(0.006984,ls='dashed',c='lightgrey')
axrates.axhline(0.03,ls='dashdot',c='darkgrey')
plt.tight_layout()
plt.savefig(os.path.join(figpath,f'rates_main.pdf'))
plt.show()