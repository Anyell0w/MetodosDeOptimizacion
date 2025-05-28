import numpy as np
import matplotlib.pyplot as plt

def regresion_lineal_simple(x, y, titulo):
    
    # Cálculos básicos
    n = len(x)
    x_media = np.mean(x)
    y_media = np.mean(y)
    
    # Fórmulas de regresión lineal
    numerador = np.sum((x - x_media) * (y - y_media))
    denominador = np.sum((x - x_media) ** 2)
    
    # Coeficientes
    m = numerador / denominador  # pendiente
    b = y_media - m * x_media    # intercepto
    
    # Predicciones y error
    y_pred = m * x + b
    mse = np.mean((y - y_pred) ** 2)
    r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_media) ** 2)
    
    # Mostrar resultados
    print(f"\n=== {titulo} ===")
    print(f"Ecuación: y = {m:.2f}x + {b:.2f}")
    print(f"Error cuadrático medio: {mse:.2f}")
    print(f"R² (bondad de ajuste): {r_squared:.3f}")
    
    # Gráfica
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', s=60, alpha=0.7, label='Datos reales')
    plt.plot(x, y_pred, color='red', linewidth=2, label=f'y = {m:.2f}x + {b:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return m, b, r_squared

# ========== EJEMPLO 1: HORAS DE ESTUDIO VS CALIFICACIÓN ==========
horas = np.array([1, 2, 3, 4, 5, 6, 7, 8])
notas = np.array([50, 55, 65, 70, 75, 85, 88, 92])

m1, b1, r2_1 = regresion_lineal_simple(horas, notas, "Horas de Estudio vs Calificación")

print("INTERPRETACIÓN:")
print(f"• Cada hora adicional de estudio aumenta la nota en {m1:.1f} puntos")
print(f"• Sin estudiar, la nota base sería {b1:.1f} puntos")
print(f"• El modelo explica el {r2_1*100:.1f}% de la variación en las notas")

# ========== EJEMPLO 2: TEMPERATURA VS VENTAS DE HELADO ==========
temperatura = np.array([20, 22, 25, 28, 30, 32, 35, 37])
ventas = np.array([10, 15, 25, 35, 45, 55, 70, 80])

m2, b2, r2_2 = regresion_lineal_simple(temperatura, ventas, "Temperatura vs Ventas de Helado")

print("INTERPRETACIÓN:")
print(f"• Cada grado adicional aumenta las ventas en {m2:.1f} unidades")
print(f"• A 0°C, las ventas serían {b2:.1f} unidades (extrapolación)")
print(f"• El modelo explica el {r2_2*100:.1f}% de la variación en ventas")

# ========== EJEMPLO 3: AÑOS DE EXPERIENCIA VS SALARIO ==========
experiencia = np.array([0, 1, 2, 3, 5, 7, 10, 12])
salario = np.array([25000, 28000, 32000, 36000, 42000, 48000, 58000, 65000])

m3, b3, r2_3 = regresion_lineal_simple(experiencia, salario, "Años de Experiencia vs Salario (USD)")

print("INTERPRETACIÓN:")
print(f"• Cada año adicional de experiencia aumenta el salario en ${m3:.0f}")
print(f"• El salario base (sin experiencia) sería ${b3:.0f}")
print(f"• El modelo explica el {r2_3*100:.1f}% de la variación salarial")

print("\n=== PREDICCIONES ===")
print("Ejemplo 1 - ¿Qué nota con 5.5 horas de estudio?")
prediccion1 = m1 * 5.5 + b1
print(f"Predicción: {prediccion1:.1f} puntos")

print("\nEjemplo 2 - ¿Ventas a 33°C?")
prediccion2 = m2 * 33 + b2
print(f"Predicción: {prediccion2:.1f} unidades")

print("\nEjemplo 3 - ¿Salario con 6 años de experiencia?")
prediccion3 = m3 * 6 + b3
print(f"Predicción: ${prediccion3:.0f}")

print("\n=== RESUMEN DE LOS 3 MODELOS ===")
print(f"1. Estudio → Nota:     R² = {r2_1:.3f} (ajuste: {'Excelente' if r2_1 > 0.9 else 'Bueno' if r2_1 > 0.7 else 'Regular'})")
print(f"2. Temperatura → Ventas: R² = {r2_2:.3f} (ajuste: {'Excelente' if r2_2 > 0.9 else 'Bueno' if r2_2 > 0.7 else 'Regular'})")
print(f"3. Experiencia → Salario: R² = {r2_3:.3f} (ajuste: {'Excelente' if r2_3 > 0.9 else 'Bueno' if r2_3 > 0.7 else 'Regular'})")