from my_hmf import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
import re


cosmo_params = {
    "H0": 70,
    "Om0": 0.294,
    "Ob0": 0.022 / 0.7**2,
    "ns": 0.965,
    "As": 2e-9
}

N_z = 5
zmax = 1




def MCMC(computedpars,N,stepfactor,thetai,Ncamb,plot,add=True):
    """
    `add` indique si on ajoute les données à celles déjà existantes (quand elles existent) ou bien si on les écrase
    """
    
    # car = f"{N}__{stepfactor}__{thetai[0]}__{thetai[1]}__{Ncamb}"
    car = f"{stepfactor}__"
    for par in thetai:
        car += f"{par}__"
    if add:
        car += "add"
    
    # files_pars = glob.glob(f'data/{car}__pars.csv')
    # files_chi2 = glob.glob(f'data/{car}__chi2.csv')
    # if add and len(files_pars) != 0 and len(files_chi2) != 0:       # "add" dans le nom du fichier indique si on a le droit d'ajouter des choses ou pas
    #     print("file found")
    #     numeros = [int(re.findall(r'\d+', file)[0]) for file in files_pars]     # On récupère les premiers numéros dans les noms de fichiers existants
        
    #     L_pars_prec = np.loadtxt(files_pars[np.argmax(numeros)])
    #     L_chi2_prec = np.loadtxt(files_chi2[np.argmax(numeros)])
    #     # Bref, on récupère les données avec le plus d'itérations
    #     # Puis on supprime les autres
    #     for i in range(len(files_pars)):
    #         if i != np.argmax(numeros):
    #             os.remove(files_pars[i])
    #             os.remove(files_chi2[i])
    if add and os.path.exists(f'data/{car}__pars.csv') and os.path.exists(f'data/{car}__chi2.csv'):
        L_pars_prec = np.loadtxt(f'data/{car}__pars.csv')
        L_chi2_prec = np.loadtxt(f'data/{car}__chi2.csv')
    
    
    else:
        L_pars_prec = None
        L_chi2_prec = None
        
        
    s = Study(N_z,zmax, computedpars, knownpars = cosmo_params, Ncamb=Ncamb)
    s.create_artificial_data(cosmo_params)
    
    step = stepfactor*thetai


    L_pars, L_chi2 = s.calc_params(thetai, N, step, plot, L_pars_prec=L_pars_prec, L_chi2_prec=L_chi2_prec)


    car = f"{stepfactor}__"        # On stocke dans un nouveau fichier vu qu'on a plus d'itérations
    for par in thetai:
        car += f"{par}__"
    if add:
        car += "add"
    
    np.savetxt(f'data/{car}__pars.csv', L_pars)
    np.savetxt(f'data/{car}__chi2.csv', L_chi2)


computedpars = ["Om0","As"]
Ncamb = 1000
N = 10000         # Nombre d'itérations de MCMC
thetai = np.array([0.32, 2.5e-9])
# thetai = np.array([1.3e-9])
stepfactor = 0.05


MCMC(computedpars,N,stepfactor,thetai,Ncamb,plot=True,add=True)