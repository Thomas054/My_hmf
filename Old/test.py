import matplotlib.pyplot as plt
import numpy as np
import hmf
from hmf import MassFunction
from hmf.halos import mass_definitions as md
from hmf import Transfer


"""
Valeurs de hmf_model possibles (fitting functions): SP, SMT, ST, Jenkins, Warren,
Reed03, Reed07, Peacock, Angulo, Angulobound, Watson_FOF, Watson, Crocce, Courtin,
Bhattacharya, Tinker08, Tinker10, Behroozi, Pillepich, Manera, Ishiyama, 
Bocquet200mDMOnly, Bocquet200mHydro, Bocquet200cDMOnly, Bocquet200cHydro,
Bocquet500cDMOnly, Bocquet500cHydro
"""

mf = MassFunction(hmf_model="Tinker08", Mmin=13)
# Mmin: puissance de 10 de la masse minimum considérée

# print(mf.quantities_available())
print(MassFunction.get_all_parameter_defaults(recursive=False))


# print([e for e in mf.power])


# plt.plot(mf.m, mf.sigma)

# plt.plot(mf.sigma, mf.fsigma)

# plt.plot(mf.m, mf.dndm)
# plt.plot(mf.k, mf.power)
# plt.xscale("log")
# plt.yscale("log")
# plt.show()


# tr = Transfer()

# print(Transfer.get_all_parameter_defaults())
print(mf.cosmo_params)


mf = MassFunction(
    Mmin=13
)  # Note how to set all cosmological parameters (except sigma_8 and n) to a given common cosmology

for z in [0, 0.7, 1.4, 2]:
    mf.update(z=z)
    plt.plot(mf.m, mf.dndm, label=f"z = {z}")

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"Masse, $[h^{-1}M_\odot]$", size=11)
plt.ylabel(r"$dn/dM$, $[h^{4}{\rm Mpc}^{-3}M_\odot^{-1}]$", size=11)
plt.legend()
plt.show()
