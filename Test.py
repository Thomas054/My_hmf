import matplotlib.pyplot as plt
import numpy as np
import time

# Créer une figure et un axe
fig, axs = plt.subplots(2,2)

# Initialiser les données
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Liste pour stocker les points affichés
x_points = []
y_points = []

# Configuration initiale du graphique
ax1 = axs[0][0]
ax2 = axs[1][0]
ax3 = axs[1][1]
axs[0][1].axis('off')
line1, = ax1.plot([], [], 'o-')  # 'bo' pour des points bleus
line2, = ax2.plot([], [], 'o-')  # 'ro' pour des points rouges
line3, = ax3.plot([], [], 'o-')  # 'go' pour des points verts
# ax.set_xlim(0, 10)
# ax.set_ylim(-1.5, 1.5)

# Afficher le graphique initial
plt.ion()  # Activer le mode interactif
plt.show()

# Boucle pour ajouter les points progressivement
for i in range(len(x)):
    x_points.append(x[i])
    y_points.append(y[i])
    
    # Mettre à jour les données des points
    line1.set_data(x_points, y_points)
    line2.set_data(x_points, y_points)
    line3.set_data(x_points, y_points)
    
    # Redessiner la figure
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # Pause pour voir l'animation
    plt.pause(0.1)  # Pause de 100 ms

plt.ioff()  # Désactiver le mode interactif
plt.show()
