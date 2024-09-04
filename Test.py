import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x):
    """La distribution cible (à échantillonner). Par exemple, une distribution normale centrée en 0."""
    return np.exp(-0.5 * x**2)

def mcmc_sampler(initial_value, iterations, proposal_std):
    samples = [initial_value]
    current_value = initial_value

    for i in range(iterations):
        # Proposer un nouvel état (par exemple, une perturbation normale)
        proposed_value = current_value + np.random.normal(0, proposal_std)

        # Calculer le ratio d'acceptation
        acceptance_ratio = target_distribution(proposed_value) / target_distribution(current_value)

        # Accepter ou rejeter la proposition
        if np.random.rand() < acceptance_ratio:
            current_value = proposed_value

        samples.append(current_value)

    return np.array(samples)

# Paramètres
initial_value = 0
iterations = 10000
proposal_std = 1.0

# Exécuter l'algorithme MCMC
samples = mcmc_sampler(initial_value, iterations, proposal_std)

# Tracer les résultats
plt.hist(samples, bins=50, density=True, label='Distribution échantillonnée')
x = np.linspace(-4, 4, 100)
plt.plot(x, target_distribution(x), label='Distribution cible', linewidth=2)
plt.legend()
plt.show()
