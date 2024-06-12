import camb
import numpy as np
from camb import model
import matplotlib.pyplot as plt


z = 0
H0 = 70
Omega_m = 0.3
Omega_b = 0.022
Omega_dm = 0.122
ns = 0.965

# Now get matter power spectra and sigma8 at redshift 0 and 0.8
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=Omega_b, omch2=Omega_dm)
pars.InitPower.set_params(ns=ns)
# Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(redshifts=[z], kmax=2.0)

# Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh, _, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)
s8 = np.array(results.get_sigma8())

# Non-Linear spectra (Halofit)
pars.NonLinear = model.NonLinear_both
results.calc_power_spectra(pars)
kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(
    minkh=1e-4, maxkh=1, npoints=200
)

print(results.get_sigma8())

plt.loglog(kh, pk[0])
plt.xlabel(r"k $[h \: \mathrm{Mpc}^{-1}]$", size=11)
plt.ylabel(r"P(k) $[h^{-3} \: \mathrm{Mpc}^{3}]$", size=11)
plt.show()


# for i, (redshift, line) in enumerate(zip(z, ["-", "--"])):
#     plt.loglog(kh, pk[i, :], color="k", ls=line)
#     plt.loglog(kh_nonlin, pk_nonlin[i, :], color="r", ls=line)
# plt.xlabel("k/h Mpc")
# plt.legend(["linear", "non-linear"], loc="lower left")
# plt.title("Matter power at z=%s" % z)
# plt.show()
