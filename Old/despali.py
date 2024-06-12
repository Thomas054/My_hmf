import matplotlib.pyplot as plt
import numpy as np
from hmf import MassFunction
from hmf.mass_function.fitting_functions import FittingFunction
from hmf.halos import mass_definitions as md
import hmf.halos.mass_definitions as m
from astropy.cosmology import Planck15


class Despali(FittingFunction):
    # a = 0.7665
    # p = 0.2488
    # A = 0.3292

    # a = 0.794
    # p = 0.247
    # A = 0.333

    delta_vir = m.SOVirial().halo_density()
    delta_c = m.SOCritical().halo_density()
    x = np.log(delta_c / delta_vir)

    a = 0.4332 * x * x + 0.2263 * x + 0.7665
    p = -0.1151 * x * x + 0.2554 * x + 0.2488
    A = -0.1362 * x + 0.3292

    @property
    def fsigma(self):
        return (
            self.A
            * (1 + (1 / (self.a * self.nu**2)) ** self.p)
            * np.sqrt(self.a * self.nu**2 / (2 * np.pi))
            * np.exp(-self.a * self.nu**2 / 2)
        )


mf_D = MassFunction(hmf_model="Despali", z=0, Mmin=13)
mf_T = MassFunction(hmf_model="Bocquet200mDMOnly", z=0, Mmin=13)

plt.plot(mf_D.m, mf_D.dndm, label="Despali")
plt.plot(mf_T.m, mf_T.dndm, label="Bocquet200mDMOnly")

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$M$")
plt.ylabel(r"$\frac{dn}{dM}$")
plt.legend()
plt.show()


# delta_vir = m.SOVirial()
# print(delta_vir.halo_density() / delta_vir.mean_density(z=0) * Planck15.Om(z=0))
