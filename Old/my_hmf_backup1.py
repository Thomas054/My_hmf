import numpy as np
import matplotlib.pyplot as plt
from hmf import MassFunction

import scipy.integrate as intg

import camb

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
N = 1000  # Nombre de points pour les courbes


class My_MassFunction:
    def __init__(
        self, z=0, cosmo_params={"Omega_m": 0.3075, "Omega_L": 1 - 0.3075, "H0": 67.74}
    ):
        """On définit tous les attributs"""
        self.z = z
        self.Omega_m = cosmo_params["Omega_m"]
        self.Omega_L = cosmo_params["Omega_L"]
        self.H0 = (
            cosmo_params["H0"] * 1000 / Mparsec_to_m
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

        self._M = 10 ** np.linspace(self.Mmin, self.Mmax, N, endpoint=True)
        self._sigma = None
        self._sigma_8 = None
        self._fsigma = None
        self._fsigma = None
        self._dndm = None

    ### Méthodes ###

    def W(self, y):
        return 3 * (np.sin(y) - y * np.cos(y)) / y**3
        # return np.sin(y / 2) / (y / 2)    # Autre formule calculée à la main: transformée de Fourier d'un porte

    @property
    def M(self):
        return self._M

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
        self, z=0, cosmo_params={"Omega_m": 0.3075, "Omega_L": 1 - 0.3075, "H0": 67.74}
    ):
        super().__init__(z, cosmo_params)
        self.hmf = MassFunction(
            hmf_model="Tinker08", z=self.z, Mmin=self.Mmin, Mmax=self.Mmax
        )
        # self.k = np.exp(
        #     np.arange(-18.42068074, 9.92931926, 0.05)
        # )  # valeurs de k (h/Mpc)
        self.k = self.hmf.k  # h Mpc^-1
        self.Pk = self.hmf.power  # h^-3 Mpc^3

        ## Paramètres de la fonction f ##
        self.alpha = np.exp(-((0.75 / np.log(self.delta / 75)) ** 1.2))
        self.A = 0.1858659 * (1 + self.z) ** (-0.14)
        self.a = 1.466904 * (1 + self.z) ** (-0.06)
        self.b = 2.571104 * (1 + self.z) ** (-self.alpha)
        self.c = 1.193958

    # @property
    # def sigma(self):
    #     if self._sigma is None:
    #         R = (3 * self.M / (4 * np.pi * self.rho_m_Masse)) ** (1 / 3)
    #         sigma2 = 0
    #         for i in range(len(self.k)):
    #             dk = (
    #                 self.k[i + 1] - self.k[i] if i < len(self.k) - 1 else dk
    #             )  # Le if est pour éviter un "out of range quand on arrive au bout de la liste"
    #             sigma2 += self.k[i] ** 2 * self.Pk[i] * self.W(self.k[i] * R) ** 2 * dk
    #         sigma2 = sigma2 / (2 * np.pi**2)
    #         self._sigma = np.sqrt(sigma2)
    #     return self._sigma

    # Calcul de sigma avec la même méthode d'intégration que hmf. Revient au même que la méthode précédente.
    @property
    def sigma(self):
        if self._sigma is None:
            R = (3 * self.M / (4 * np.pi * self.rho_m_Masse)) ** (1 / 3)
            y = (
                self.k**2 * self.Pk * self.W(np.outer(R, self.k)) ** 2
            )  # Fonction de k ET M à intégrer suivant k
            sigma2 = intg.simps(y, self.k, axis=-1) / (2 * np.pi**2)
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


mf = MassFunction(hmf_model="Tinker08", Mmin=13)
my_mf = My_Tinker08()


# # Test Pk
# plt.xscale("log")
# plt.yscale("log")
# plt.plot(mf.k, mf.power, label="hmf")
# plt.plot(my_mf.k, my_mf.Pk, label="my_hmf")
# plt.legend()
# plt.show()

# ## Test sigma
# print(mf.sigma_8)
# print(my_mf.sigma_8)


# plt.plot(mf.m, mf.radii / ((3 * my_mf.M / (4 * np.pi * my_mf.rho_m_Masse)) ** (1 / 3)))       # Test R
# plt.xscale("log")
# plt.show()

# plt.plot(mf.m, my_mf.sigma / mf.sigma, label="hmf/my_hmf")
# plt.xscale("log")
# plt.show()

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

print(mf.mean_density0)
print(my_mf.rho_m_Masse)
