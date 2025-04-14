import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Configuración estética
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})

# Definición de restricciones


def r1(x): return (200 - 0.5*x)/0.01  # Tierra: 0.5x₁ + 0.01x₂ ≤ 200
def r2(x): return 500*np.ones_like(x)  # Agua: x₃ ≤ 500 (no se grafica)
def r3(x): return (3000 - 10*x)/0.5   # Presupuesto: 10x₁ + 0.5x₂ + 2x₃ ≤ 3000


# Rango de valores para x₁ (área cultivada)
x = np.linspace(0, 400, 500)

# Puntos de intersección (cálculo analítico)
A = np.array([0, r1(0)])
B = np.array([400, r1(400)])
C = np.array([400, r3(400)])
D = np.array([0, r3(0)])

# Región factible (polígono)
vertices = np.vstack([A, B, C, D])
region = Polygon(vertices, alpha=0.3, color='#2ca02c', label='Región factible')

# Creación de figura
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar restricciones
ax.plot(x, r1(x), 'r-', lw=2, label='Límite de tierra: $0.5x_1 + 0.01x_2 \leq 200$')
ax.plot(x, r3(x), 'b-', lw=2,
        label='Límite presupuestal: $10x_1 + 0.5x_2 \leq 3000$')
ax.add_patch(region)

# Punto óptimo (solución del modelo)
optimo = np.array([150, 1250])
ax.plot(optimo[0], optimo[1], 'ko', markersize=8,
        label='Solución óptima (150, 1250)')

# Configuración visual
ax.set_xlim(0, 400)
ax.set_ylim(0, 3000)
ax.set_xlabel('Área cultivada ($x_1$) [hectáreas]', fontweight='bold')
ax.set_ylabel('Fertilizante orgánico ($x_2$) [kg]', fontweight='bold')
ax.set_title('Región Factible - Producción de Quinua en Puno',
             pad=20, fontweight='bold')
ax.legend(loc='upper right', framealpha=1)

# Líneas de cuadrícula
ax.grid(True, linestyle='--', alpha=0.7)

# Guardar en alta resolución (ideal para papers)
plt.savefig('region_factible_quinua.png',
            dpi=300,
            bbox_inches='tight',
            format='png')

plt.show()
