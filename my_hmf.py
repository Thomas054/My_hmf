import numpy as np
import matplotlib.pyplot as plt
from hmf import MassFunction
import scipy.integrate as intg

import camb
from camb import model

try:
    import BCemu

    BCemuOK = True

except ImportError:
    BCemuOK = False

"""
Améliorations possibles:
- Au lieu de calculer dlnsigma/dlnm directement, on peut faire comme dans hmf:
    avec papier-crayon, rentrer la dérivée dans l'intégrale et pousser les calculs au maximum,
    puis utiliser la formule obtenue.
"""

"""
Infos:
- Les paramètres cosmologiques sont donnés dans les unités conventionnelles, 
puis converties en unités cosmologiques  l'intérieur du code.
- Au départ, on initialise les objets qu'on va calculer à None.
    Les fonnctions d'accès à ces objets (les @property) vont les calculer ils sont à None,
    puis enregistrer le résultat dans l'attribut correspondant, pour les prochaines fois.
- Le type d'effet des baryons considéré est entièrement compris dans le dictionnaire baryons_effect.
    La clé type donne accès au type d'effet considéré, les autres sont les paramètres de l'effet.
"""

# 1 parsec = 3.0857e16 m

Mparsec_to_m = 3.0857e16 * 1e6
MSun = 1.989e30  # kg
GSI = 6.67430e-11  # m^3 kg^-1 s^-2
G = GSI * MSun / Mparsec_to_m**3  # Mpc^3 MSun^-1 s^-2
cSI = 3.0e8  # m s^-1
c = cSI / Mparsec_to_m  # Mpc s^-1
N = 2000  # Nombre de points pour les courbes


Neff = 3.046
Tcmb0 = 2.7255  # K

# Effets baryoniques
if BCemuOK:
    Giri = {
            "log10Mc": 13.32,
            "mu": 0.93,
            "thej": 4.235,
            "gamma": 2.25,
            "delta": 6.40,
            "eta": 0.15,
            "deta": 0.14,
    }

cosmo_params = {
    "H0": 70,
    "Om0": 0.294,
    "Ob0": 0.022 / 0.7**2,
}

n = 0.965


class My_MassFunction:
    def __init__(
        self,
        z,
        cosmo_params,
        n,
        baryons_effect,
        baryons_params,
        Mmin,
        Mmax,
        kmin,
        kmax,
    ):
        """On définit tous les attributs"""
        self.z = z
        self.Om0 = cosmo_params["Om0"]
        self.Ode0 = 1 - self.Om0
        self.Ob0 = cosmo_params["Ob0"]
        self.Odm0 = self.Om0 - self.Ob0
        # self.Odm0 = 0.122 / 0.7**2
        self.n = n
        self.H0classique = cosmo_params["H0"]
        self.H0 = (
            self.H0classique * 1000 / Mparsec_to_m
        )  # On convertit des km s^-1 Mpc^-1 en s^-1
        self.h = cosmo_params["H0"] / 100
        self.rho_c = (
            3 * c**2 * self.H0**2 / (8 * np.pi * G)
        )  # MSun Mpc^-1 s^-2. Densité critique à z=0
        # self.H = H0 * (self.Om0 * (1 + self.z) ** 3 + self.Ode0) ** 0.5
        self.rho_m = (
            self.Om0 * self.rho_c / self.h**2
        )  # h^2 MSun Mpc^-1 s^-2 (densité d'ENERGIE)
        self.rho_m_Masse = self.rho_m / c**2  # h**2 MSun Mpc^-3 (densité de MASSE)
        self.Mmin = Mmin  # Puissance de 10 dans la masse minimale considérée, en h^-1 . MasseSolaire
        self.Mmax = Mmax  # Puissance de 10 dans la masse maximale considérée, en h^-1 . MasseSolaire
        self.kmin = kmin  # h/Mpc
        self.kmax = kmax  # h/Mpc

        self.baryons_effect = baryons_effect
        self.baryons_params = baryons_params

        self._m = np.linspace(10**self.Mmin, 10**self.Mmax, N)
        self._k = None
        self._Pk_camb = None
        self._Pk = None
        self._sigma = None
        self._sigma_8 = None
        self._fsigma = None
        self._fsigma = None
        self._dndm = None

    @property
    def m(self):
        return self._m

    def get_k_Pk_camb(self):
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.H0classique, ombh2=self.Ob0 * self.h**2, omch2=self.Odm0 * self.h**2
        )
        pars.InitPower.set_params(ns=self.n)
        pars.set_matter_power(redshifts=[self.z], kmax=2.0)

        # Linear spectra
        pars.NonLinear = model.NonLinear_none
        results = camb.get_results(pars)
        k, _, Pk = results.get_matter_power_spectrum(
            minkh=self.kmin, maxkh=self.kmax, npoints=N
        )
        return k, Pk[0]

    @property
    def k(self):
        if self._k is None:
            k, Pk = self.get_k_Pk_camb()
            self._k = k
            self._Pk_camb = Pk
        return self._k

    @property
    def Pk_camb(self):
        if self._Pk_camb is None:
            k, Pk = self.get_k_Pk_camb()
            self._k = k
            self._Pk_camb = Pk
        return self._Pk_camb

    @property
    def Pk(self):
        if self._Pk is None:
            if self.baryons_effect == "Giri":
                if not BCemuOK:
                    print(
                        "Erreur: BCemu n'est pas installé. Impossible de calculer l'effet des baryons."
                    )
                    self._Pk = self.Pk_camb
                else:
                    bfcemu = BCemu.BCM_7param(Ob=self.Ob0, Om=self.Om0)
                    bcmdict = self.baryons_params
                    P_quotient = bfcemu.get_boost(self.z, bcmdict, self.k)
                    self._Pk = P_quotient * self.Pk_camb
            elif self.baryons_effect == "Aucun":
                self._Pk = self.Pk_camb
            else:
                effect = self.baryons_effect
                print(f"Erreur: Effet des baryons {effect} inconnu.")
                self._Pk = self.Pk_camb

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
            i = np.where(m <= self.m)[0][
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
        cosmo_params=cosmo_params,
        n=n,
        baryons_effect="Aucun",
        baryons_params = {},
        delta=200,
        Mmin=13,
        Mmax=15,
        kmin=1e-4,
        kmax=1,
    ):
        super().__init__(
            z=z,
            cosmo_params=cosmo_params,
            n=n,
            baryons_effect=baryons_effect,
            baryons_params=baryons_params,
            Mmin=Mmin,
            Mmax=Mmax,
            kmin=kmin,
            kmax=kmax,
        )
        
        self.delta = delta
        ## Paramètres de la fonction f ##
        self.alpha = 10 ** (-((0.75 / np.log10(self.delta / 75)) ** 1.2))
        self.A = 0.1858659 * (1 + self.z) ** (-0.14)
        self.a = 1.466904 * (1 + self.z) ** (-0.06)
        self.b = 2.571104 * (1 + self.z) ** (-self.alpha)
        self.c = 1.193958

    @property
    def sigma(self):
        if self._sigma is None:
            R = (3 * self.m / (4 * np.pi * self.rho_m_Masse)) ** (1 / 3)
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
            ln_sigma = np.log(self.sigma)
            # derivee = (ln_sigma[1:] - ln_sigma[:-1]) / (
            #     self.m[1:] - self.m[:-1]
            # )  # Dérivée de ln(sigma) par rapport à M
            # derivee = np.append(
            #     derivee, derivee[-1]
            # )  # Pour que la liste ait la même taille que M
            derivee = np.gradient(ln_sigma, self.m)
            self._dndm = self.fsigma * self.rho_m_Masse / self.m * np.abs(derivee)
        return self._dndm
