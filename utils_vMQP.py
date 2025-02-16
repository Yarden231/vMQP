#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
from numpy import linalg as LA
from tqdm import tqdm

from scipy.linalg import eigh

def get_M(xs, l1= .005, l2 = .01):
    
    jitter = 0.000001
    def corrFunc(xa,xb):
      return l1*np.exp(-((xa-xb)**2)/(2.0*l2)) 

    d = xs.shape[0]
    K = np.zeros([d,d])
    for i in range(d):
        for j in range(i+1):
            noise=(jitter if i==j else 0)
            k = corrFunc(xs[i],xs[j]) + noise

            K[i,j] = k
            K[j,i] = k
            
    M = np.linalg.inv(K)
    
    return M




class getM_exp():    
    def __init__(self, xs):        
        diff = (np.expand_dims(xs,1) - np.expand_dims(xs,0))**2
        self.xdiff = np.sum(diff, axis=2)  
        self.jitter = 0.000001
        self.d = xs.shape[0]
        
    def M(self, l1= 1, l2 = .01 ):                
        K = l1*np.exp(-0.5*self.xdiff/l2) + self.jitter*np.eye(self.d)

        return np.linalg.inv(K)
    
class getM_gauss():    
    def __init__(self, xs):        
        diff = np.abs(np.expand_dims(xs,1) - np.expand_dims(xs,0))
        self.xdiff = np.sum(diff, axis=2)  
        self.jitter = 0.000001
        self.d = xs.shape[0]
        
    def M(self, l1= 1, l2 = .01 ):                
        K = l1*np.exp(-0.5*self.xdiff/l2) + self.jitter*np.eye(self.d)

        return np.linalg.inv(K)
    
    
    

def Gibbs_vMP(M, kappa, nu,ys, ind_obs,N, every, ind_un= None, tqdm_disable=True, discard = 20, initial_sample = None, A = None, get_likelihoods = False, HMC= False, reflect = 0):

    d  = M.shape[0]
    du = d - len(ys)  # number of unobserved variables in the model
    if ind_un is None:
        ind_un = [i for i in range(d) if i not in ind_obs]    
    
    if len(list(ind_obs))== []:
        Q = M
        rhoc = np.ones([du,1])*kappa*np.cos(nu)
        rhos = np.ones([du,1])*kappa*np.sin(nu)
    else:
        Q = M[ind_un,:][:,ind_un]   # precision matrix restricted to the unobserved variables     
        Muo = M[ind_un,:][:,ind_obs]
        
        rhoc = -np.matmul(Muo, np.cos(ys)) + kappa*np.cos(nu)
        rhoc = np.reshape(rhoc, [du,1])         
        rhos = -np.matmul(Muo, np.sin(ys)) + kappa*np.sin(nu)
        rhos = np.reshape(rhos, [du,1]) 
    
    
    if A is None:
        lo = eigh(Q, eigvals_only=True, subset_by_index= [du-1,du-1])[0]
        if lo >0:        
            lamda = 1.03*lo + 1e-100
        else:
            lamda = 0.97*lo + 1e-100
        # eigvs, _ = LA.eig(Q)            
        
        # lamda = 1.03*np.real(np.max(eigvs)) + 1e-100
        # ee, _ = LA.eig( lamda*np.eye(d) - M)
        
        A = LA.cholesky(lamda*np.eye(du) - Q ).T
    
    # set initial values of the random variables 
    if initial_sample is None:
        thetas = np.random.rand(du) -0.5 
        thetas = np.reshape(thetas, [du,1]) 
    else:
        thetas = np.reshape(initial_sample[ind_un], [du,1]) 

    samples = np.zeros([d, int(N/every)])
    samples[ind_obs,:] = np.expand_dims(np.array(ys),1) # fix the observations for all the samples 

    if get_likelihoods:
        nllikes = [] 
        def log_likelihood(varphi):    
            cos = np.cos(varphi - varphi.T)
            return 0.5*(cos*M).sum()
    
    
    varphi =  np.zeros([d,1])
    varphi[ind_obs,0] = ys
    
    # Gibbs sampler
    for i in tqdm(range(-discard,N+1), disable=tqdm_disable):
        
        
        cs = np.cos(thetas)
        ss = np.sin(thetas)
    
        z1 = A.dot(cs) + np.random.normal(size=[du,1])
        z2 = A.dot(ss) + np.random.normal(size=[du,1])
    
        kc = rhoc+ (A.T).dot(z1)
        ks = rhos + (A.T).dot(z2)
        
        alphas = np.sqrt(kc**2 + ks**2)
        gammas = np.arctan2(ks,kc)          # gammas in [-pi, pi]
        
        thetas = np.random.vonmises(gammas, alphas)   # theta in [-pi, pi]
        varphi[ind_un] = thetas
        
        if i >0 and i%every==0:
            samples[ind_un,int(i/every)-1] = thetas[:,0]
            if get_likelihoods:
                nllikes.append(log_likelihood(varphi)) 
    
    
    if not get_likelihoods:
        return samples, A     
    
    else:
        return samples, np.array(nllikes), A

    

def  bridge_log_factor(thetap,Mp,Ap,kappap,nup,M,A,kappa,nu,K):
    
    
    xi = thetap
    
    d  = M.shape[0] 
    
    betas = np.arange(0,K+2)/(K+1)    
    sqb = np.sqrt(betas)
    osqb = np.sqrt(1-betas)
    
    
    k = 0
    Mk = betas[k]*M + (1-betas[k])*Mp
    Mkp = betas[k+1]*M + (1-betas[k+1])*Mp     
    
    alphc = betas[k]*kappa*np.cos(nu) + (1-betas[k])*kappap*np.cos(nup)
    alphs = betas[k]*kappa*np.sin(nu) + (1-betas[k])*kappap*np.sin(nup)

    alphcp = betas[k+1]*kappa*np.cos(nu) + (1-betas[k+1])*kappap*np.cos(nup)
    alphsp = betas[k+1]*kappa*np.sin(nu) + (1-betas[k+1])*kappap*np.sin(nup)
    
    cosp = np.cos(xi - xi.T)
    sumcosp = np.cos(xi).sum()
    sumsinp = np.sin(xi).sum()
    
    bridge = 0.5*(Mk*cosp).sum() -alphc*sumcosp -alphs*sumsinp    - 0.5*(Mkp*cosp).sum() + alphcp*sumcosp + alphsp*sumsinp
    
    for k in range(1,K+1):   # k samples
        cs = np.cos(xi)
        ss = np.sin(xi)
    
        
        y1 = sqb[k]*A.dot(cs) + np.random.normal(size=[d,1])
        y2 = sqb[k]*A.dot(ss) + np.random.normal(size=[d,1])
        y3 = osqb[k]*Ap.dot(cs) + np.random.normal(size=[d,1])
        y4= osqb[k]*Ap.dot(ss) + np.random.normal(size=[d,1])
    
        kc = sqb[k]*(A.T).dot(y1) + osqb[k]*(Ap.T).dot(y3) + alphcp*np.ones([d,1])
        ks = sqb[k]*(A.T).dot(y2) + osqb[k]*(Ap.T).dot(y4) + alphsp*np.ones([d,1])
        
        alphas = np.sqrt(kc**2 + ks**2)
        gammas = np.arctan2(ks,kc)        
        xi = np.random.vonmises(gammas, alphas)   # xi in [-pi, pi]
        
        Mk = Mkp 
        alphc = alphcp
        alphs = alphsp
        
        Mkp = betas[k+1]*M + (1-betas[k+1])*Mp 
        cosp = np.cos(xi - xi.T)
        alphcp = betas[k+1]*kappa*np.cos(nu) + (1-betas[k+1])*kappap*np.cos(nup)
        alphsp = betas[k+1]*kappa*np.sin(nu) + (1-betas[k+1])*kappap*np.sin(nup)

        sumcosp = np.cos(xi).sum()
        sumsinp = np.sin(xi).sum()

        bridge += 0.5*(Mk*cosp).sum() -alphc*sumcosp -alphs*sumsinp - 0.5*(Mkp*cosp).sum() + alphcp*sumcosp + alphsp*sumsinp
        
    
    return bridge 




    
    
    
    


