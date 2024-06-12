import numpy as np
import matplotlib.pyplot as plt
from hmf import MassFunction

import camb
from camb import model

"""
Infos:
- Les paramètres cosmologiques sont donnés dans les unités conventionnelles, 
puis converties en unités cosmologiques  l'intérieur du code.
- Au départ, on initialise les objets qu'on va calculer à None.
    Les fonnctions d'accès à ces objets (les @property) vont les calculer ils sont à None,
    puis enregistrer le résultat dans l'attribut correspondant, pour les prochaines fois.
"""

# 1 parsec = 3.0857e16 m

Mparsec_to_m = 3.0857e16 * 1e6
MSun = 1.989e30  # kg
GSI = 6.67430e-11  # m^3 kg^-1 s^-2
G = GSI * MSun / Mparsec_to_m**3  # Mpc^3 MSun^-1 s^-2
cSI = 3.0e8  # m s^-1
c = cSI / Mparsec_to_m  # Mpc s^-1
N = 1024  # Nombre de points pour les courbes


class My_MassFunction:
    def __init__(
        self,
        z=0,
        cosmo_params={
            "H0": 70,
            "Omega_m": 0.3,
            "Omega_L": 0.7,
            "Omega_b": 0.022,
            "Omega_dm": 0.122,
            "ns": 0.965,
        },
        baryons_effect="Aucun",
    ):
        """On définit tous les attributs"""
        self.z = z
        self.Omega_m = cosmo_params["Omega_m"]
        self.Omega_L = cosmo_params["Omega_L"]
        self.Omega_b = cosmo_params["Omega_b"]
        self.Omega_dm = cosmo_params["Omega_dm"]
        self.ns = cosmo_params["ns"]
        self.H0classique = cosmo_params["H0"]
        self.H0 = (
            self.H0classique * 1000 / Mparsec_to_m
        )  # On convertit des km s^-1 Mpc^-1 en s^-1
        self.h = cosmo_params["H0"] / 100
        self.rho_c = (
            3 * c**2 * self.H0**2 / (8 * np.pi * G)
        )  # MSun Mpc^-1 s^-2. Densité critique à z=0
        # self.H = H0 * (self.Omega_m * (1 + self.z) ** 3 + self.Omega_L) ** 0.5
        self.rho_m = (
            self.Omega_m * self.rho_c / self.h**2
        )  # h^2 MSun Mpc^-1 s^-2 (densité d'ENERGIE)
        self.rho_m_Masse = self.rho_m / c**2  # h**2 MSun Mpc^-3 (densité de MASSE)
        self.delta = 200  # Surdensité critique à z=0
        self.Mmin = 13  # Puissance de 10 dans la masse minimale considérée, en h^-1 . MasseSolaire
        self.Mmax = 15  # Puissance de 10 dans la masse maximale considérée, en h^-1 . MasseSolaire

        self.baryons_effect = baryons_effect

        self._M = np.linspace(10**self.Mmin, 10**self.Mmax, N)
        self._k = None
        self._Pk = None
        self._sigma = None
        self._sigma_8 = None
        self._fsigma = None
        self._fsigma = None
        self._dndm = None

    ### Méthodes ###

    @property
    def M(self):
        return self._M

    def get_k_Pk(self):
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H0classique, ombh2=self.Omega_b, omch2=self.Omega_dm)
        pars.InitPower.set_params(ns=self.ns)
        pars.set_matter_power(redshifts=[self.z], kmax=2.0)

        # Linear spectra
        pars.NonLinear = model.NonLinear_none
        results = camb.get_results(pars)
        k, _, Pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=N)
        return k, Pk[0]

    @property
    def k(self):
        if self._k is None:
            k, Pk = self.get_k_Pk()
            self._k = k
            self._Pk = Pk
        return self._k

    @property
    def Pk(self):
        if self._Pk is None:
            k, Pk = self.get_k_Pk()
            self._k = k
            self._Pk = Pk
        if self.baryons_effect == "BCemu":
            pass
        elif self.baryons_effect != "Aucun":
            print(f"Erreur: Effet des baryons {self.baryons_effect} inconnu.")
        return self._Pk

    def W(self, y):
        return 3 * (np.sin(y) - y * np.cos(y)) / y**3
        # return np.sin(y / 2) / (y / 2)    # Autre formule calculée à la main: transformée de Fourier d'un porte

    @property
    def sigma(self):
        pass

    # Calcul de sigma pour R = 8 Mpc/h.
    @property
    def sigma_8(self):
        if self._sigma_8 is None:
            R = 8  # 8 Mpc/h
            m = 4 / 3 * np.pi * R**3 * self.rho_m_Masse
            i = np.where(m <= self.M)[0][
                0
            ]  # L'indice i tel que M[i] est le premier élément de M supérieur à m
            self._sigma_8 = self.sigma[i]
        return self._sigma_8

    @property
    def fsigma(self):
        pass

    @property
    def dndm(self):
        pass


class My_Tinker08(My_MassFunction):
    def __init__(
        self,
        z=0,
        cosmo_params={
            "H0": 70,
            "Omega_m": 0.3,
            "Omega_L": 0.7,
            "Omega_b": 0.022,
            "Omega_dm": 0.122,
            "ns": 0.965,
        },
    ):
        super().__init__(z, cosmo_params)

        ## Paramètres de la fonction f ##
        self.alpha = 10 ** (-((0.75 / np.log10(self.delta / 75)) ** 1.2))
        self.A = 0.186 * (1 + self.z) ** (-0.14)
        self.a = 1.47 * (1 + self.z) ** (-0.06)
        self.b = 2.57 * (1 + self.z) ** (-self.alpha)
        self.c = 1.19

    @property
    def sigma(self):
        if self._sigma is None:
            R = (3 * self.M / (4 * np.pi * self.rho_m_Masse)) ** (1 / 3)
            sigma2 = 0
            for i in range(len(self.k)):
                dk = (
                    self.k[i + 1] - self.k[i] if i < len(self.k) - 1 else dk
                )  # Le if est pour éviter un "out of range quand on arrive au bout de la liste"
                sigma2 += self.k[i] ** 2 * self.Pk[i] * self.W(self.k[i] * R) ** 2 * dk
            sigma2 = sigma2 / (2 * np.pi**2)
            self._sigma = np.sqrt(sigma2)
        return self._sigma

    @property
    def fsigma(self):
        if self._fsigma is None:
            self._fsigma = (
                self.A
                * ((self.sigma / self.b) ** (-self.a) + 1)
                * np.exp(-self.c / self.sigma**2)
            )
        return self._fsigma

    @property
    def dndm(self):
        if self._dndm is None:
            ln_sigma_inv = np.log(1 / self.sigma)
            derivee = (ln_sigma_inv[1:] - ln_sigma_inv[:-1]) / (
                self.M[1:] - self.M[:-1]
            )  # Dérivée de ln(sigma^-1) par rapport à M
            derivee = np.append(
                derivee, derivee[-1]
            )  # Pour que la liste ait la même taille que M
            self._dndm = self.fsigma * self.rho_m_Masse / self.M * derivee
        return self._dndm


mf = MassFunction(
    hmf_model="Tinker08",
    Mmin=13,
    lnk_min=np.log(1e-4),
    lnk_max=np.log(1),
    z=1,
)
my_mf = My_Tinker08(z=1)


plt.plot(mf.k, mf.power, label="hmf")
plt.plot(my_mf.k, my_mf.Pk, label="my_hmf")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

## Test sigma
print(my_mf.sigma_8)
plt.plot(mf.m, mf.sigma, label="hmf")
plt.plot(my_mf.M, my_mf.sigma, label="my_hmf")
plt.xscale("log")
plt.legend()
plt.show()


## Test fsigma
plt.plot(mf.sigma, mf.fsigma, label="hmf")
plt.plot(my_mf.sigma, my_mf.fsigma, label="my_hmf")
plt.legend()
plt.show()


## Test dndm
plt.xscale("log")
plt.yscale("log")
plt.plot(mf.m, mf.dndm, label="hmf")
plt.plot(my_mf.M, my_mf.dndm, label="my_hmf")
plt.legend()
plt.show()

# print(mf.mean_density0)
# print(my_mf.rho_m_Masse)
