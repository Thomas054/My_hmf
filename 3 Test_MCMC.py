from my_hmf import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
import re


cosmo_params = {"H0": 70, "Om0": 0.294, "Ob0": 0.022 / 0.7**2, "ns": 0.965, "As": 2e-9}


# Fonction p(M,z)
def cutting_function(z):
    return 8 * np.log(z + 1) * 1e14


def p(m, z):
    """
    p(M,z) = 1 si M > cutting_function(z), 0 sinon
    """
    return (m > cutting_function(z)).astype(int)


N_z = 5
N_m = 10
zmax = 1


computedpars = ["Om0", "As"]
Ncamb = 1000
N = 10000  # Nombre d'itérations de MCMC
thetai = np.array([0.31, 2.1e-9])
# thetai = np.array([1.3e-9])
stepfactor = 0.051

s = Study(N_z, zmax, computedpars, knownpars=cosmo_params, Ncamb=Ncamb, N_m=N_m)
s.set_p(p)
s.create_artificial_data(cosmo_params)
s.MCMC(N, stepfactor, thetai, plot=False, add=True, newpos=False)

# MCMC(computedpars,N,stepfactor,thetai,Ncamb,plot=False,add=True)
