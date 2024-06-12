from my_hmf import My_Tinker08
import numpy as np
import matplotlib.pyplot as plt


Giri = {
    "effect": "Giri",
    "params": {
        "log10Mc": 13.32,
        "mu": 0.93,
        "thej": 4.235,
        "gamma": 2.25,
        "delta": 6.40,
        "eta": 0.15,
        "deta": 0.14,
    },
}

Girimin = {
    "effect": "Giri",
    "params": {
        "log10Mc": 11,
        "mu": 0,
        "thej": 2,
        "gamma": 1,
        "delta": 3,
        "eta": 0.05,
        "deta": 0.05,
    },
}
Girimax = {
    "effect": "Giri",
    "params": {
        "log10Mc": 15,
        "mu": 2,
        "thej": 8,
        "gamma": 4,
        "delta": 11,
        "eta": 0.4,
        "deta": 0.25,
    },
}


# mf = My_Tinker08()
# mfG = My_Tinker08(baryons_effect=Giri)


# # On vérifie qu'on retrouve bien la courbe de l'article de Giri
# plt.loglog(mf.k,mfG.Pk/mf.Pk)
# plt.show()

# ## Test dndm
# plt.plot(mfG.M, mfG.dndm, label="hmf de Giri")
# plt.plot(mf.M, mf.dndm, label="my_hmf")
# plt.xlabel(r"Masse ($h^{-1} M_\odot$)")
# plt.xscale("log")
# plt.yscale("log")
# plt.legend()
# plt.show()

# ## Rapport des fonctions de masse
# plt.plot(mf.M,mfG.dndm/mf.dndm - 1)
# plt.plot(mf.M,np.zeros(len(mf.M)), linestyle="--", color="grey", linewidth=0.5)
# plt.xscale("log")
# plt.xlabel("Masse")
# plt.ylabel("Giri / classique - 1")
# plt.show()


## On fait varier z
def vary_z(L_z):
    for z in L_z:
        mf = My_Tinker08(z=z, kmin=0.034, kmax=1)
        mfG = My_Tinker08(z=z, baryons_effect=Giri, kmin=0.034, kmax=1)
        plt.plot(mf.M, mfG.dndm / mf.dndm - 1, label=f"z={z}")
    plt.plot(mf.M, np.zeros(len(mf.M)), linestyle="--", color="grey", linewidth=0.5)
    plt.legend()
    plt.xscale("log")
    plt.xlabel(r"Masse ($h^{-1} M_\odot$)")
    plt.title(
        "Rapport entre la dndm prenant en compte les effets de Giri et la dndm classique"
    )
    plt.ylabel(r"$\dfrac{Giri}{classique} - 1$")
    plt.show()


L_z = [0, 0.7, 1.4, 2]  # ,1.5,1.6,1.7,1.8,1.9


# vary_z(L_z)


## On fait varier z pour M fixée
def vary_z_Mfixe():
    L_z = np.linspace(0, 2, 10)
    dndm = []
    for z in L_z:
        mf = My_Tinker08(z=z, kmin=0.034, kmax=1)
        mfG = My_Tinker08(z=z, baryons_effect=Giri, kmin=0.034, kmax=1)
        i = np.where(mf.M > 1e14)[0][0]
        dndm.append(mfG.dndm[i] / mf.dndm[i] - 1)
    plt.plot(L_z, dndm)
    plt.plot(L_z, np.zeros(len(L_z)), linestyle="--", color="grey", linewidth=0.5)
    plt.xlabel("z")
    plt.title(
        r"Rapport entre la dndm prenant en compte les effets de Giri et la dndm classique, masse fixée à $10^14 h^{-1} M_\odot$"
    )
    plt.ylabel(r"$\dfrac{Giri}{classique} - 1$")
    plt.show()


# vary_z_Mfixe()


## On fait varier la cosmologie pour z=0
def vary_cosmo(L_params):
    for params in L_params:
        mf = My_Tinker08(cosmo_params=params, kmin=0.034, kmax=1)
        mfG = My_Tinker08(cosmo_params=params, baryons_effect=Giri, kmin=0.034, kmax=1)
        plt.plot(
            mf.M,
            mfG.dndm / mf.dndm - 1,
            label=rf"$H_0 =$ {params['H0']}, $\Omega_m =$ {params['Om0']}, $\Omega_b =$ {params['Ob0']}, $n =$ {params['n']}",
        )
    plt.plot(mf.M, np.zeros(len(mf.M)), linestyle="--", color="grey", linewidth=0.5)
    plt.plot([1e13], [-0.03], color="white")
    plt.legend()
    plt.xscale("log")
    plt.xlabel(r"Masse ($h^{-1} M_\odot$)")
    plt.title(
        r"Rapport entre la dndm prenant en compte les effets de Giri et la dndm classique à $z=0$"
    )
    plt.ylabel(r"$\dfrac{Giri}{classique} - 1$")
    plt.show()


cosmo_params1 = {
    "H0": 67.74,
    "Om0": 0.3075,
    "Ob0": 0.0486,
    "n": 0.9667,
}

cosmo_params2 = {"H0": 74, "Om0": 0.27, "Ob0": 0.052, "n": 1.0667}

# vary_cosmo([cosmo_params1,cosmo_params2])


## On teste l'influence des paramètres de Giri sur dndm, pour différents redshifts
def vary_Giri(L_z):
    Giribismin = {
        "effect": "Giri",
        "params": {c: Giri["params"][c] for c in Giri["params"]},
    }  # on copie le dictionnaire Giri
    Giribismax = {
        "effect": "Giri",
        "params": {c: Giri["params"][c] for c in Giri["params"]},
    }  # on copie le dictionnaire Giri à nouveau
    for param in Giri["params"]:
        Giribismin["params"][param] = Girimin["params"][param]
        Giribismax["params"][param] = Girimax["params"][param]
        for z in L_z:
            mfGmin = My_Tinker08(z=z, baryons_effect=Giribismin, kmin=0.034, kmax=1)
            mfGmax = My_Tinker08(
                z=z, baryons_effect=Giribismax, kmin=0.034, kmax=1
            )  # dndm modifiée (1 paramètre de Giri changé)
            plt.plot(mfGmin.M, mfGmax.dndm / mfGmin.dndm - 1, label=f"z={z}")
        Giribismin["params"][param] = Giri["params"][param]
        Giribismax["params"][param] = Giri["params"][param]
        plt.plot(
            mfGmin.M,
            np.zeros(len(mfGmin.M)),
            linestyle="--",
            color="grey",
            linewidth=0.5,
        )
        plt.legend()
        plt.xscale("log")
        plt.xlabel(r"Masse ($h^{-1} M_\odot$)")
        plt.title(
            f"Rapport entre la dndm de Giri avec {param} au maximum et au minimum"
        )
        plt.ylabel(r"$\dfrac{GiriMax}{GiriMin} - 1$")
        plt.show()


L_z = [0, 0.7, 1.4, 2]  # ,1.5,1.6,1.7,1.8,1.9

# vary_Giri(L_z)
