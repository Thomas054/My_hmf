from my_hmf import *
import numpy as np
import matplotlib.pyplot as plt
import time


cosmo_params = {
    "H0": 70,
    "Om0": 0.294,
    "Ob0": 0.022 / 0.7**2,
    "ns": 0.965,
    "As": 2e-9
}

Ncamb = 1000
N_z = 5
zmax = 1



s = Study(N_z,zmax, ["Om0","As"], knownpars = cosmo_params, Ncamb=Ncamb)
s.create_artificial_data(cosmo_params)



N = 100
thetai = np.array([0.4, 5e-9])
stepfactor = 0.1
step = stepfactor*thetai

L_pars, L_chi2 = s.calc_params(thetai, N, step, plot=True)


car = f"{N}__{stepfactor}__{thetai[0]}__{thetai[1]}__{Ncamb}"

np.savetxt(f'data/{car}__pars.csv', L_pars)
np.savetxt(f'data/{car}__chi2.csv', L_chi2)