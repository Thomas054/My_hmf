import numpy as np
import matplotlib.pyplot as plt
from hmf import MassFunction
import scipy.integrate as intg
import os

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
Ncamb = 2000  # Nombre de points pour les courbes
resolution_z = 100


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
    "ns": 0.965,
    "As": 2e-9
}




class My_MassFunction:
    def __init__(
        self,
        z,
        cosmo_params,
        baryons_effect,
        baryons_params,
        Mmin,
        Mmax,
        kmin,
        kmax,
        Ncamb,
    ):
        """On définit tous les attributs"""
        self.z = z
        self.Om0 = cosmo_params["Om0"]
        self.Ode0 = 1 - self.Om0
        self.Ob0 = cosmo_params["Ob0"]
        self.Odm0 = self.Om0 - self.Ob0
        # self.Odm0 = 0.122 / 0.7**2
        self.ns = cosmo_params["ns"]
        self.As = cosmo_params["As"]
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
        self.Ncamb = Ncamb

        self.baryons_effect = baryons_effect
        self.baryons_params = baryons_params

        self._m = np.linspace(10**self.Mmin, 10**self.Mmax, Ncamb)
        self._k = None
        self._Pk_camb = None
        self._Pk = None
        self._sigma = None
        self._sigma_8 = None
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
        pars.InitPower.set_params(ns=self.ns, As=self.As)
        pars.set_matter_power(redshifts=[self.z], kmax=2.0)

        # Linear spectra
        pars.NonLinear = model.NonLinear_none
        results = camb.get_results(pars)
        k, _, Pk = results.get_matter_power_spectrum(
            minkh=self.kmin, maxkh=self.kmax, npoints=self.Ncamb
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
        baryons_effect="Aucun",
        baryons_params = {},
        delta=200,
        Mmin=13,
        Mmax=15,
        kmin=1e-4,
        kmax=1,
        Ncamb=Ncamb,
    ):
        super().__init__(
            z=z,
            cosmo_params=cosmo_params,
            baryons_effect=baryons_effect,
            baryons_params=baryons_params,
            Mmin=Mmin,
            Mmax=Mmax,
            kmin=kmin,
            kmax=kmax,
            Ncamb=Ncamb,
        )
        
        self.delta = delta
        ## Paramètres de la fonction f ##
        self._alpha = None
        self._A = None
        self._a = None
        self._b = None
        self._c = None

    @property
    def alpha(self):
        if self._alpha is None:
            self._alpha = 10 ** (-((0.75 / np.log10(self.delta / 75)) ** 1.2))
        return self._alpha
    
    @property
    def A(self):
        if self._A is None:
            self._A = 0.1858659 * (1 + self.z) ** (-0.14)
        return self._A
    
    @property
    def a(self):
        if self._a is None:
            self._a = 1.466904 * (1 + self.z) ** (-0.06)
        return self._a
    
    @property
    def b(self):
        if self._b is None:
            self._b = 2.571104 * (1 + self.z) ** (-self.alpha)
        return self._b
    
    @property
    def c(self):
        if self._c is None:
            self._c = 1.193958
        return self._c
    
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
    
    def set_z(self, z):
        self.z = z
        self._Pk = None
        self._Pk_camb = None
        self._sigma = None
        self._fsigma = None
        self._dndm = None
        self._sigma_8 = None
        self._alpha = None
        self._A = None
        self._a = None
        self._b = None
        self._c = None
    
    
    # @property
    # def number_count(self, N, zmax):
    #     # On intègre dndm sur M puis sur z
    #     res = np.zeros(N)
    #     for i in range(N):
    #         z = zmax * i / N
    #         L_z = np.linspace(z, z+zmax/N, N)   # Prendre N pour le nombre de points est totalement arbitraire ici
    #         L_intgs = np.zeros(N)   # Liste des intégrales sur M en fonction de z
    #         for j in range(N):
    #             self.set_z(L_z[j])
    #             L_intgs[j] = intg.trapz(self.dndm, self.m)
    #         res[i] = intg.trapz(L_intgs, L_z)
    #     return res

def get_number_count(cosmo_params, N, zmax, Ncamb, resolution_z):
    """
    Number count as a function of z.

    Args:
        cosmo_params (dict): _description_
        N (int): number of points for the redshifts
        zmax (float): maximum redshift
        Ncamb (int): number of points for the mass
        resolution_z (int): number of points for the redshifts in each bin, for the integration

    Returns:
        (float list, float list): redshifts and number count
    """
    mf = My_Tinker08(z=0, cosmo_params=cosmo_params, Ncamb=Ncamb)
    # On intègre dndm sur M puis sur z
    res = np.zeros(N)
    intgs_sur_m = np.zeros(N+1) # On inclut les 2 bornes de l'intervalle considéré, d'où le +1
    for i in range(N+1):    # A chaque point, on le relie au point précédent par un segment et on calcule l'intégrale de la courbe ainsi créée
        z = zmax * i / N
        mf.set_z(z)
        intgs_sur_m[i] = intg.trapz(mf.dndm, mf.m)
        if i >= 1:
            fonction_interpolee = np.interp(np.linspace(z-zmax/N, z, resolution_z, endpoint = False), [z-zmax/N, z], intgs_sur_m[i-1:i+1])      # On relie les 2 points par une droite
            res[i-1] = intg.trapz(fonction_interpolee, np.linspace(z-zmax/N, z, resolution_z, endpoint = False))
    # intgs_sur_m = np.interp(np.linspace(0, zmax, resolution_z, endpoint = False), np.linspace(0, zmax, N, endpoint = False), intgs_sur_m)
    # for i in range(N):
    #     z = zmax * i / N
    return np.linspace(0, zmax, N, endpoint = False), res


def calc_chi2(data, model, std):
    return np.sum((data - model)**2 / std**2)




class Study():
    def __init__(self, N_z, zmax, computedpars, knownpars = cosmo_params, Ncamb=Ncamb, resolution_z=resolution_z): 
        """
        Initializes the study.
        Args:
            N_z (int): number of points for the redshifts (used for number count)
            zmax (float): maximum redshift
            computedpars (list[str]): list of the names of the parameters to compute
            knownpars (dict, optional): Permet d'avoir une valeurs pour les paramètres cosmologiques fixés. Peut contenir aussi des paramètres non fixés. Defaults to cosmo_params.
        """
        self.z = np.linspace(0, zmax, N_z, endpoint = False)
        self.N_z = N_z
        self.Ncamb=Ncamb
        self.zmax = zmax
        self.resolution_z = resolution_z
        self.computedpars = computedpars
        self.knownpars = knownpars
        self.cosmo_params = {"Om0": None, "Ob0": None, "H0": None, "ns": None, "As": None}
        for c in self.knownpars:
            self.cosmo_params[c] = self.knownpars[c]
        self._data = None
        self._std = None
        
        self.h = knownpars["H0"] / 100        
        self.Asmin = 1e-9
        self.Asmax = 3e-9
        # 0.1 = omch2min = Odm0min * h**2 = (Omm0min - Ob0) * h**2
        self.Om0min = 0.1/(self.h**2) + knownpars["Ob0"]
        self.Om0max = 0.15/(self.h**2) + knownpars["Ob0"]
        
        self._thetai = None
        self._stepfactor = None
    
    
    # def calc_number_count(self, cosmo_params):
    #     mf = My_Tinker08(z=0, cosmo_params=cosmo_params)
    #     # On intègre dndm sur M puis sur z
    #     res = np.zeros(self.N)
    #     for i in range(self.N):
    #         z = self.zmax * i / N
    #         L_z = np.linspace(z, z+self.zmax/self.N, self.N)   # Prendre N pour le nombre de points est totalement arbitraire ici
    #         L_intgs = np.zeros(self.N)   # Liste des intégrales sur M en fonction de z
    #         for j in range(self.N):
    #             mf.set_z(L_z[j])
    #             L_intgs[j] = intg.trapz(mf.dndm, mf.m)
    #         res[i] = intg.trapz(L_intgs, L_z)
    #     self._number_count = res
    
    
    def create_artificial_data(self, cosmo_params):
        _, self._data = get_number_count(cosmo_params, self.N_z, self.zmax, self.Ncamb, self.resolution_z)
        self._std = 0.1 * self._data
    
    @property
    def data(self):
        if self._data is None:
            raise ValueError("No data found. Please run create_artificial_data or provide real data.")
        return self._data
    
    @property
    def std(self):
        if self._std is None:
            raise ValueError("No data found. Please run create_artificial_data or provide real data.")
        return self._std

    @property
    def thetai(self):
        if self._thetai is None:
            raise ValueError("No initial guess for the parameters. Please run MCMC or set_tetai first.")
        return self._thetai
    
    def set_thetai(self, thetai):
        self._thetai = thetai
    
    @property
    def stepfactor(self):
        if self._stepfactor is None:
            raise ValueError("No step factor for the parameters. Please run MCMC or set_stepfactor first.")
        return self._stepfactor
    
    def set_stepfactor(self, stepfactor):
        self._stepfactor = stepfactor
        
    def update_params(self, theta):
        for i in range(len(theta)):
            self.cosmo_params[self.computedpars[i]] = theta[i]
            
    def theta_is_valid(self, theta):
        for i in range(len(theta)):
            if self.computedpars[i] == "Om0":
                if theta[i] < self.Om0min or theta[i] > self.Om0max:
                    return False
            elif self.computedpars[i] == "As":
                if theta[i] < self.Asmin or theta[i] > self.Asmax:
                    return False
        return True
    
    def calc_params(self, theta_i, N, step, plot=False, L_pars_prec=None, L_chi2_prec=None):
        """
        Computes the parameters.
        Args:
            theta_i (list[str]): Initial guess for the parameters to compute
            N (int): Number of iterations
            step (list[float]): Step for the parameters
        
        Returns:
            L_pars, L_chi2 (list[list[float]], list[float]): Respectively the parameters and the chi_2 associated for each iteration.\\
            `L_pars[:,j]` contains the values of the parameter j for each iteration. (so the tab is tall but not very large)
        """
        if len(theta_i) == 2:
            assert self.computedpars == ["Om0", "As"] and self.Om0min < theta_i[0] and theta_i[0] < self.Om0max and self.Asmin < theta_i[1] and theta_i[1] < self.Asmax
        if len(theta_i) == 1 and self.computedpars == ["Om0"]:
            assert self.Om0min < theta_i[0] and theta_i[0] < self.Om0max
        if len(theta_i) == 1 and self.computedpars == ["As"]:
            assert self.Asmin < theta_i[0] and theta_i[0] < self.Asmax
        i = 0       # Pour la boucle while
        if self._data is None:
            raise ValueError("No data to fit. Please run create_artificial_data or provide real data.")
        L_pars = np.zeros((N,len(theta_i)))
        L_chi2 = np.zeros(N)
        if L_pars_prec is not None:     # Si on veut continuer une chaîne
            theta_i = L_pars_prec[-1]
            L_pars = np.concatenate((L_pars_prec[:-1], L_pars))  # On enlève le dernier élément de L_pars_prec pour ne pas le rajouter deux fois
            L_chi2 = np.concatenate((L_chi2_prec[:-1], L_chi2))  # Idem
            i = len(L_pars_prec)-1        # On commence à l'indice suivant
            N = len(L_pars)
        self.update_params(theta_i)
        theta_prec = np.copy(theta_i)
        print(theta_prec)
        _, model = get_number_count(self.cosmo_params, self.N_z, self.zmax, self.Ncamb, self.resolution_z)
        chi2_prec = calc_chi2(self.data, model, self.std)
        print(chi2_prec)
        L_pars[i] = theta_prec
        L_chi2[i] = chi2_prec
        
        if plot:
            fig, axs = plt.subplots(2,2)
            axs[0,1].axis('off')
            axOm = axs[0,0]
            axOm.set_ylabel(r"$\chi_2$")
            axtot = axs[1,0]
            axtot.set_xlabel(r"$\Omega_m$")
            axtot.set_ylabel(r"$A_s$")
            axAs = axs[1,1]
            axAs.set_xlabel(r"$\chi_2$")
            lineOm, = axOm.plot(L_pars[:1,0], L_chi2[:1], 'o-')
            linetot, = axtot.plot(L_pars[:1,0], L_pars[:1,1], 'o-')
            lineAs, = axAs.plot(L_chi2[:1], L_pars[:1,1], 'o-')
            
            axOm.plot(L_pars[0,0], L_chi2[0], 'ro')
            axtot.plot(L_pars[0,0], L_pars[0,1], 'ro')
            axAs.plot(L_chi2[0], L_pars[0,1], 'ro')
            
            plt.ion()
            plt.show()
            
        
        burning = True
        i += 1
        try:
            while burning:
                print(i)
                theta_new = np.random.normal(theta_prec, step)
                print(theta_new)
                self.update_params(theta_new)
                _, model = get_number_count(self.cosmo_params, self.N_z, self.zmax, self.Ncamb, self.resolution_z)
                chi2_new = calc_chi2(self.data, model, self.std)
                print(chi2_new)
                x = np.random.uniform()
                if x < chi2_prec/chi2_new and self.theta_is_valid(theta_new):
                    theta_prec = np.copy(theta_new)
                    chi2_prec = chi2_new
                    print("Kept !")
                    L_pars[i] = theta_prec
                    L_chi2[i] = chi2_prec
                    if plot:
                        lineOm.set_data(L_pars[:i+1,0], L_chi2[:i+1])
                        linetot.set_data(L_pars[:i+1,0], L_pars[:i+1,1])
                        lineAs.set_data(L_chi2[:i+1], L_pars[:i+1,1])
                        
                        axOm.relim()
                        axOm.autoscale_view()
                        axtot.relim()
                        axtot.autoscale_view()
                        axAs.relim()
                        axAs.autoscale_view()
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    i += 1
                # Si les nouveaux paramètres ne sont pas retenus, on ne fait strictement rien (on n'incrémente même pas i)
                    
                if i >= N:
                    burning = False
        finally:
            return L_pars[:i], L_chi2[:i]
    
    
    def MCMC(self,N,stepfactor,thetai,plot,add=True,newpos=False):
        """
        `add` indique si on ajoute les données à celles déjà existantes (quand elles existent) ou bien si on les écrase
        """
        # Enregistrement de thetai et stepfactor
        self._thetai = thetai
        self._stepfactor = stepfactor
        
        # Création du nom du fichier
        car = f"{stepfactor}"
        if add:
            car += "__add"
         
        # On regarde si on peut reprendre une chaîne
        if add and os.path.exists(f'data/{car}__pars.csv') and os.path.exists(f'data/{car}__chi2.csv'):
            print("File found !")
            L_pars_prec = np.loadtxt(f'data/{car}__pars.csv')
            L_chi2_prec = np.loadtxt(f'data/{car}__chi2.csv')
        # Sinon on initialise les listes
        else:
            L_pars_prec = None
            L_chi2_prec = None
            
            
        # s = Study(N_z,zmax, computedpars, knownpars = cosmo_params, Ncamb=Ncamb)
        
        step = stepfactor*thetai

        if newpos:
            print("New starting position")
            # On rajoute thetai à L_pars_prec et le chi2 associé dans L_chi2_prec
            L_pars_prec = np.append(L_pars_prec, [thetai], axis = 0)
            self.update_params(thetai)
            L_chi2_prec = np.append(L_chi2_prec, calc_chi2(self.data, get_number_count(self.cosmo_params, self.N_z, self.zmax, self.Ncamb, self.resolution_z)[1], self.std))  # On calcule le chi2 pour thetai
        L_pars, L_chi2 = self.calc_params(thetai, N, step, plot, L_pars_prec=L_pars_prec, L_chi2_prec=L_chi2_prec)


        car = f"{stepfactor}"        # On stocke dans un nouveau fichier vu qu'on a plus d'itérations
        if add:
            car += "__add"
        
        np.savetxt(f'data/{car}__pars.csv', L_pars)
        np.savetxt(f'data/{car}__chi2.csv', L_chi2)
        
    
    def find_best_values(self):
        car = f"{self.stepfactor}__add"
        
        try:
            L_pars = np.loadtxt(f'data/{car}__pars.csv')
            L_chi2 = np.loadtxt(f'data/{car}__chi2.csv')
        except FileNotFoundError:
            print("File not found. Please run MCMC first.")
            return None
        i = np.argmin(L_chi2)
        return L_pars[i]
    
    def get_approximate_number_count(self):
        """Computes the number count with the best values of the parameters given by MCMC.

        Returns:
            list: Number count
        """
        theta = self.find_best_values()
        self.update_params(theta)
        return get_number_count(self.cosmo_params, self.N_z, self.zmax, self.Ncamb, self.resolution_z)[1]
        
            


    




# N_z = 5
# zmax = 1




# s = Study(N_z,zmax, ["Om0","As"], knownpars = cosmo_params)
# s.create_artificial_data(cosmo_params)

# thetai = thetai = [cosmo_params["Om0"],cosmo_params["As"]]
# res = s.calc_params(thetai)

# print("\n")
# print(f"Le vrai : {cosmo_params['Om0']}")
# print(f"Résultat : {res}")