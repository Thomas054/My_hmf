from my_hmf import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os


cosmo_params = {
    "H0": 70,
    "Om0": 0.294,
    "Ob0": 0.022 / 0.7**2,
    "ns": 0.965,
    "As": 2e-9
}

N_z = 5
zmax = 1




def MCMC(N,stepfactor,thetai,Ncamb, add=True):
    """
    `add` indique si on ajoute les données à celles déjà existantes (quand elles existent) ou bien si on les écrase
    """
    
    # car = f"{N}__{stepfactor}__{thetai[0]}__{thetai[1]}__{Ncamb}"
    car = f"{N}__{stepfactor}__"
    for par in thetai:
        car += f"{par}__"
    car += f"{Ncamb}"
    
    if add and os.path.exists(f'data/{car}__pars.csv') and os.path.exists(f'data/{car}__chi2.csv'):
        L_pars_prec = np.loadtxt(f'data/{car}__pars.csv')
        L_chi2_prec = np.loadtxt(f'data/{car}__chi2.csv')
        new_N = len(L_chi2_prec-1) + N - 1      # Nombre total d'itérations. On enlève 1 pour prendre en compte la "fusion" des deux listes
    else:
        L_pars_prec = None
        L_chi2_prec = None
        new_N = N
        
        
    s = Study(N_z,zmax, ["Om0","As"], knownpars = cosmo_params, Ncamb=Ncamb)
    s.create_artificial_data(cosmo_params)
    
    step = stepfactor*thetai


    L_pars, L_chi2 = s.calc_params(thetai, N, step, plot=True, L_pars_prec=L_pars_prec, L_chi2_prec=L_chi2_prec)


    car = f"{new_N}__{stepfactor}__"        # On stocke dans un nouveau fichier vu qu'on a plus d'itérations
    for par in thetai:
        car += f"{par}__"
    car += f"{Ncamb}"
    
    np.savetxt(f'data/{car}__pars.csv', L_pars)
    np.savetxt(f'data/{car}__chi2.csv', L_chi2)


Ncamb = 1000
N = 100         # Nombre d'itérations de MCMC
thetai = np.array([0.2, 2e-9])
stepfactor = 0.1


MCMC(N,stepfactor,thetai,Ncamb, add=True)