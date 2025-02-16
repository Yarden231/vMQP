#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from dataclasses import dataclass
import numpy as np
import numpy.typing
from tqdm import tqdm

from utils_vMQP import getM_gauss, getM_exp, Gibbs_vMP, bridge_log_factor


@dataclass
class MCMC_Samples:
    ''' containers for sampled angles and parameters '''
    l1s: np.array = None 
    l2s: np.array = None 
    kappas: np.array = None 
    nus: np.array = None 
    angles: np.array = None 



def sample(Nsamples, pars, xs, ind_obs, ys, i=None, samples = None, kernel = "exp"):    
    
    if kernel == "gauss":        
        gM = getM_gauss(xs) # Gaussian kernel
    else:    
        gM = getM_exp(xs) # Exponential kernel

    d = gM.d
    
    if d > len(ys):
        sample_latents = True
    else:
        sample_latents = False


    if samples is None:
        samples = MCMC_Samples()        
        samples.l2s = np.zeros(Nsamples)
        samples.l1s = np.zeros(Nsamples)
        samples.kappas = np.zeros(Nsamples)
        samples.nus = np.zeros(Nsamples)            
        if sample_latents: 
            samples.angles = np.zeros([d, Nsamples])
        
        # initial angles values
        if sample_latents: 
            theta = np.zeros([d,1])
            theta[ind_obs,0] = ys    
        else:
            theta = np.array(ys)
            
        i = 0
        # set initial parameter values
        kappa = 0
        nu = 0
        l1 = pars.l10
        l2 = pars.l20
        
        
    else:
        # initialize the Markov chain with the last values sampled
        kappa = samples.kappas[i-1]
        nu = samples.nus[i-1]
        l1 = samples.l1s[i-1]
        l2 = samples.l2s[i-1]
        if sample_latents: 
            theta = samples.angles[:,i-1:i] 
        else:
            theta = np.array(ys)

    
    M = gM.M(l1= l1, l2 = l2)     
        
    thetap = theta

    if pars.bridging:
        _, AM = Gibbs_vMP(M, kappa, nu, [], [], pars.N_thetap , pars.N_thetap, discard=pars.discard, initial_sample = theta) # We want to have the initial A
        K = pars.K
    
    AQ = None
    acc = 0
    all_acc = 0
    trials = 0
    try:        
        
        for i in tqdm(range(i,Nsamples)):
        
            # propose new parameters 
            l1p = np.abs(l1 + pars.sigma_prop_l1*np.random.randn())            
            l2p = np.abs(l2 + pars.sigma_prop_l2*np.random.randn())
            kappap = np.abs(kappa + pars.sigma_prop_kappa*np.random.randn())        
            nup = nu + pars.sigma_prop_nu*(np.random.rand()-0.5)  # add uniform noise symmetric around zero 
                
            # sample fictitious data point
            Mp = gM.M(l1 = l1p, l2 = l2p)         
            thetap, AMp = Gibbs_vMP(Mp, kappap, nup, [], [],pars.N_thetap , pars.N_thetap , tqdm_disable=pars.tqdm_disable, discard=pars.discard, initial_sample = thetap) # [d,1], [d,d]    
            
        
            
            #compute MH log-factors for priors
            mh_prior_kappa = 0.5*(kappa**2-kappap**2)/pars.sigma2_kappa    
            mh_priors = mh_prior_kappa + 0.5*(l1**2-l1p**2)/pars.sigma2_l1  + 0.5*(l2**2- l2p**2)/pars.sigma2_l2
        
            cos = np.cos(theta - theta.T)    
        
            # MH factor
            f1 = 0.5*(M*cos).sum() - kappa*np.cos(theta-nu).sum()
            f3 = 0.5*(Mp*cos).sum() - kappap*np.cos(theta-nup).sum()
                        
            mh = mh_priors -f3 + f1
        
        
            if pars.bridging:            
                bridge =  bridge_log_factor(thetap,Mp,AMp,kappap,nup,M,AM,kappa,nu,K)
                mh = mh + bridge                
                
            else:        
                cosp = np.cos(thetap - thetap.T)            
                f2 = 0.5*(M*cosp).sum() - kappa*np.cos(thetap-nu).sum()        
                f4 = 0.5*(Mp*cosp).sum() - kappap*np.cos(thetap-nup).sum()                        
                mh = mh  -f2 + f4 
                
            # Accept/Reject step     
            trials +=1
            if mh> 0 or np.random.random() < np.exp(mh):   # accept
                l1 = l1p
                l2 = l2p
                kappa = kappap
                nu = nup
                acc += 1
                all_acc += 1
                M = Mp
                AM = AMp
                AQ = None # Since the parameters were updated, AQ has to be recomputed
        
            if pars.Geweke:    
                # sample from the model assuming all the variables are unobserved 
                theta, _  = Gibbs_vMP(M, kappa, nu, [],[] ,pars.N_theta, pars.N_theta, tqdm_disable=pars.tqdm_disable,  discard=pars.discard, initial_sample = theta, A=AM) # [d,1]            
            else:
                # sample unobserved variables conditioned on the observed 
                if sample_latents:
                    theta, AQ  = Gibbs_vMP(M, kappa, nu, ys,ind_obs,pars.N_theta, pars.N_theta, discard=pars.discard, initial_sample = theta, A=AQ) # [d,1]    
            
            #print(i,mh,'\n',theta) 
            samples.l2s[i]= l2
            samples.l1s[i]= l1
            samples.kappas[i] = kappa
            samples.nus[i] = nu    
            if sample_latents:
                samples.angles[:,i] = theta[:,0]
            
            if i>0 and i % 500==0:
            
                print(' accept rate:', acc/500 )
                acc = 0 

    except KeyboardInterrupt:
         print('Returned')    
         pass
    
    samples.nus = np.mod(samples.nus,2*np.pi) 
    samples.nus[samples.nus>np.pi] = samples.nus[samples.nus>np.pi] -2*np.pi 
    mean_acc = all_acc/trials
    print('mean_acc', mean_acc) 
    return samples,i, mean_acc
    
    
# def sample2(Nsamples, pars, xs, ind_obs, ys, i=None, samples = None):    
    
#     gM = getM(xs)
#     d = gM.d
    
#     if d > len(ys):
#         sample_latents = True
#     else:
#         sample_latents = False


#     if samples is None:
#         samples = MCMC_Samples()        
#         samples.l2s = np.zeros(Nsamples)
#         samples.l1s = np.zeros(Nsamples)
#         samples.kappas = np.zeros(Nsamples)
#         samples.nus = np.zeros(Nsamples)            
#         if sample_latents: 
#             samples.angles = np.zeros([d, Nsamples])
        
#         # initial angles values
#         if sample_latents: 
#             theta = np.zeros([d,1])
#             theta[ind_obs,0] = ys    
#         else:
#             theta = np.array(ys)
            
#         i = 0
#         # set initial parameter values
#         kappa = 0
#         nu = 0
#         l1 = pars.l10
#         l2 = pars.l20
        
        
#     else:
#         # initialize the Markov chain with the last values sampled
#         kappa = samples.kappas[i-1]
#         nu = samples.nus[i-1]
#         l1 = samples.l1s[i-1]
#         l2 = samples.l2s[i-1]
#         if sample_latents: 
#             theta = samples.angles[:,i-1:i] 
#         else:
#             theta = np.array(ys)

    
#     M = gM.M(l1= l1, l2 = l2)     
        
#     thetap = theta

#     if pars.bridging:
#         _, AM = Gibbs_vMP(M, kappa, nu, [], [], pars.N_thetap , pars.N_thetap, discard=pars.discard, initial_sample = theta) # We want to have the initial A
#         K = pars.K
    
#     AQ = None
#     acc = 0
#     all_acc = 0
#     trials = 0
#     try:        
        
#         for i in tqdm(range(i,Nsamples)):
        
#             # propose new parameters 
#             l1p = np.abs(l1 + pars.sigma_prop_l1*np.random.randn())            
#             l2p = np.abs(l2 + pars.sigma_prop_l2*np.random.randn())
#             kappap = np.abs(kappa + pars.sigma_prop_kappa*np.random.randn())        
#             nup = nu + pars.sigma_prop_nu*(np.random.rand()-0.5)  # add uniform noise symmetric around zero 
                
#             # sample fictitious data point
#             Mp = gM.M(l1 = l1p, l2 = l2p)         
#             thetap, AMp = Gibbs_vMP(Mp, kappap, nup, [], [],pars.N_thetap , pars.N_thetap , tqdm_disable=pars.tqdm_disable, discard=pars.discard, initial_sample = thetap) # [d,1], [d,d]    
            
        
            
#             #compute MH log-factors for priors
#             mh_prior_kappa = 0.5*(kappa**2-kappap**2)/pars.sigma2_kappa    
#             mh_priors = mh_prior_kappa + 0.5*(l1**2-l1p**2)/pars.sigma2_l1  + 0.5*(l2**2- l2p**2)/pars.sigma2_l2
        
#             cos = np.cos(theta - theta.T)    
        
#             # MH factor
#             f1 = 0.5*(M*cos).sum() - kappa*np.cos(theta-nu).sum()
#             f3 = 0.5*(Mp*cos).sum() - kappap*np.cos(theta-nup).sum()
                        
#             mh = mh_priors -f3 + f1
        
        
#             if pars.bridging:            
#                 bridge =  bridge_log_factor(thetap,Mp,AMp,kappap,nup,M,AM,kappa,nu,K)
#                 mh = mh + bridge                
                
#             else:        
#                 cosp = np.cos(thetap - thetap.T)            
#                 f2 = 0.5*(M*cosp).sum() - kappa*np.cos(thetap-nu).sum()        
#                 f4 = 0.5*(Mp*cosp).sum() - kappap*np.cos(thetap-nup).sum()                        
#                 mh = mh  -f2 + f4 
                
#             # Accept/Reject step     
#             trials +=1
#             if mh> 0 or np.random.random() < np.exp(mh):   # accept
#                 l1 = l1p
#                 l2 = l2p
#                 kappa = kappap
#                 nu = nup
#                 acc += 1
#                 all_acc += 1
#                 M = Mp
#                 AM = AMp
#                 AQ = None # Since the parameters were updated, AQ has to be recomputed
        
#             if pars.Geweke:    
#                 # sample from the model assuming all the variables are unobserved 
#                 theta, _  = Gibbs_vMP(M, kappa, nu, [],[] ,pars.N_theta, pars.N_theta, tqdm_disable=pars.tqdm_disable,  discard=pars.discard, initial_sample = theta, A=AM) # [d,1]            
#             else:
#                 # sample unobserved variables conditioned on the observed 
#                 if sample_latents:
#                     theta, AQ  = Gibbs_vMP(M, kappa, nu, ys,ind_obs,pars.N_theta, pars.N_theta, discard=pars.discard, initial_sample = theta, A=AQ) # [d,1]    
            
#             #print(i,mh,'\n',theta) 
#             samples.l2s[i]= l2
#             samples.l1s[i]= l1
#             samples.kappas[i] = kappa
#             samples.nus[i] = nu    
#             if sample_latents:
#                 samples.angles[:,i] = theta[:,0]
            
#             if i>0 and i % 1000==0:
            
#                 print(' accept rate:', acc/1000 )
#                 acc = 0 

#     except KeyboardInterrupt:
#          print('Returned')    
#          pass
    
#     samples.nus = np.mod(samples.nus,2*np.pi) 
#     samples.nus[samples.nus>np.pi] = samples.nus[samples.nus>np.pi] -2*np.pi 
#     mean_acc = all_acc/trials
#     print('mean_acc', mean_acc) 
#     return samples,i, mean_acc