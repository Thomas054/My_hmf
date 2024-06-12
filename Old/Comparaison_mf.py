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

# Paramètres
overdensity = 200
Om0 = 0.307
nom1 = "Bocquet200cHydro"
nom2 = "Bocquet200cDMOnly"


zdebut = 0
zfin = 1.4
nb_z = 3

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
        mf1.m,
        mf1.dndm / mf2.dndm - 1,
        label=f"{varying_param} = {eval(varying_param)}",
    )
    plt.plot(mf1.m, np.zeros(len(mf1.m)), linestyle="--", color="grey", linewidth=0.5)


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
    plt.xlabel(r"$M$", size=12)
    plt.ylabel(r"$\dfrac{dn}{dM}$", size=18)
    plt.legend()
    plt.title(
        rf"{nom1} / {nom2} - 1, $\quad \Delta = {overdensity}$ , $\quad \Omega_m (0) = {Om0}$"
    )
    plt.show()


# Trace plusieurs courbes sur le même graphe en faisant varier le redshift
def comparer_z(
    nom1=nom1,
    nom2=nom2,
    zdebut=zdebut,
    zfin=zfin,
    nb_redshifts=nb_z,
    mdef_model="SOCritical",
    overdensity=overdensity,
    Om0=Om0,
    Mmin=13,
):
    for z in np.linspace(zdebut, zfin, nb_redshifts):
        comparer_plot(nom1, nom2, mdef_model, overdensity, Om0, z, Mmin, "z")
    plt.xscale("log")
    plt.xlabel(r"$M$", size=12)
    plt.ylabel(r"$\frac{dn}{dM}$", size=18)
    plt.legend()
    # params_fixes = [param for param in params if param != varying_param]
    # s = rf"{nom1} / {nom2},"
    # for param in params_fixes:
    #     s += f" {param} = {eval(param)},"
    # s = s[:-1]
    plt.title(
        rf"{nom1} / {nom2} - 1, $\quad \Delta = {overdensity}$ , $\quad \Omega_m (0) = {Om0}$"
    )
    plt.show()


# Trace plusieurs courbes sur le même graphe en faisant varier Om0
def comparer_Om0(
    nom1=nom1,
    nom2=nom2,
    Om0debut=Om0debut,
    Om0fin=Om0fin,
    nb_Om0=nb_Om0,
    mdef_model="SOCritical",
    overdensity=overdensity,
    z=0,
    Mmin=13,
):
    for Om0 in np.linspace(Om0debut, Om0fin, nb_Om0):
        comparer_plot(nom1, nom2, mdef_model, overdensity, Om0, z, Mmin, "Om0")
    plt.xscale("log")
    plt.xlabel(r"$M$", size=12)
    plt.ylabel(r"$\dfrac{dn}{dM}$", size=18)
    plt.legend()
    plt.title(rf"{nom1} / {nom2} - 1, $\quad \Delta = {overdensity}$ , $\quad z = {z}$")
    plt.show()
    
comparer_z()
