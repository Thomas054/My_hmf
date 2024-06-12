import matplotlib.pyplot as plt
import numpy as np
from hmf import MassFunction


# plt.loglog: échelles logarithmiques
# plt.semilogx: échelle logarithmique pour l'axe des x
# plt.semilogy: échelle logarithmique pour l'axe des y

"""
Noms possibles (fitting functions): SP, SMT, ST, Jenkins, Warren,
Reed03, Reed07, Peacock, Angulo, Angulobound, Watson_FOF, Watson, Crocce, Courtin,
Bhattacharya, Tinker08, Tinker10, Behroozi, Pillepich, Manera, Ishiyama, 
Bocquet200mDMOnly, Bocquet200mHydro, Bocquet200cDMOnly, Bocquet200cHydro,
Bocquet500cDMOnly, Bocquet500cHydro
"""

params = ["overdensity", "z", "Om0"]
overdensity = 500
Om0 = 0.307
nom1 = "Bocquet500cHydro"
nom2 = "Bocquet500cDMOnly"

zdebut = 0
zfin = 2
nb_z = 5

Om0debut = 0.08  # Valeur de Ob0. Quand on augmente Om0, on augmente Odm0
Om0fin = 1
nb_Om0 = 5


# Trace une courbe de la fonction de masse sans afficher le graphe
def comparer_plot(
    nom1=nom1,
    nom2=nom2,
    mdef_model="SOCritical",
    overdensity=overdensity,
    Om0=Om0,
    z=0,
    Mmin=13,
    varying_param="z",
):
    mf1 = MassFunction(
        hmf_model=nom1,
        mdef_model=mdef_model,
        mdef_params={"overdensity": overdensity},
        cosmo_params={"Om0": Om0},
        z=z,
        Mmin=Mmin,
    )
    mf2 = MassFunction(
        hmf_model=nom2,
        mdef_model=mdef_model,
        mdef_params={"overdensity": overdensity},
        cosmo_params={"Om0": Om0},
        z=z,
        Mmin=Mmin,
    )

    plt.plot(
        mf1.sigma,
        mf1.fsigma / mf2.fsigma,
        label=f"{varying_param} = {eval(varying_param)}",
    )


# Met en place le graphe et l'affiche
def comparer(
    nom1=nom1,
    nom2=nom2,
    mdef_model="SOCritical",
    overdensity=overdensity,
    Om0=Om0,
    z=0,
    Mmin=13,
):
    comparer_plot(nom1, nom2, mdef_model, overdensity, Om0, z, Mmin)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\sigma$", size=15)
    plt.ylabel(r"$f(\sigma)$", size=15)
    plt.legend()
    plt.title(
        rf"{nom1} / {nom2}, $\quad \Delta = {overdensity}$ , $\quad \Omega_m (0) = {Om0}$"
    )
    plt.show()


comparer()
