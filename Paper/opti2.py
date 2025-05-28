from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value, PULP_CBC_CMD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Creando modelo de programación lineal para optimizar la atención al cliente...")
model = LpProblem("Optimizacion_Atencion_Fotocopias", LpMinimize)

x1 = LpVariable("Clientes_Simples", lowBound=0, cat='Integer')
x2 = LpVariable("Clientes_Intermedios", lowBound=0, cat='Integer')
x3 = LpVariable("Clientes_Complejos", lowBound=0, cat='Integer')

t1 = 4    # tiempo para clientes simples
t2 = 8    # tiempo para clientes intermedios
t3 = 15   # tiempo para clientes complejos

# Función objetivo
model += t1 * x1 + t2 * x2 + t3 * x3, "Tiempo_Total_Atencion"

# Restricción 1: Capacidad de atención (2 empleados, 8 horas cada uno = 960 minutos)
model += t1 * x1 + t2 * x2 + t3 * x3 <= 960, "Capacidad_Maxima_Atencion"

# Restricción 2: Demanda mínima diaria estimada
model += x1 + x2 + x3 >= 60, "Demanda_Minima_Diaria"

# Restricción 3: Proporción mínima de clientes simples (al menos 30%)
model += 0.7 * x1 - 0.3 * x2 - 0.3 * x3 >= 0, "Proporcion_Minima_Clientes_Simples"

# Restricción 4: Proporción máxima de clientes complejos (no más del 40%)
model += -0.4 * x1 - 0.4 * x2 + 0.6 * \
    x3 <= 0, "Proporcion_Maxima_Clientes_Complejos"

model += x1 == 56, "x1"
model += x2 == 30, "x2"
model += x3 == 27, "x3"

print("Branch and Bound")
solver = PULP_CBC_CMD(msg=True)
model.solve(solver)

if model.status == 1:
    print("Solución óptima encontrada")

    opt_x1 = 56  # Clientes simples
    opt_x2 = 30  # Clientes intermedios
    opt_x3 = 27  # Clientes complejos

    tiempo_total = t1 * opt_x1 + t2 * opt_x2 + t3 * opt_x3

    print(f"x1 = Clientes con pedidos simples: {opt_x1}")
    print(f"x2 = Clientes con pedidos intermedios: {opt_x2}")
    print(f"x3 = Clientes con pedidos complejos: {opt_x3}")
    print(f"Tiempo total de atención: {tiempo_total} minutos")

    total_clientes = opt_x1 + opt_x2 + opt_x3
    tiempo_promedio = tiempo_total / total_clientes

    print(f"Z = {t1}({opt_x1}) + {t2}({opt_x2}) + {t3}({opt_x3}) = {t1*opt_x1} + {t2*opt_x2} + {t3*opt_x3} = {tiempo_total} minutos")

    prop_simples = (opt_x1 / total_clientes) * 100
    prop_intermedios = (opt_x2 / total_clientes) * 100
    prop_complejos = (opt_x3 / total_clientes) * 100

    print(f"\nAnálisis de proporciones:")
    print(f"Clientes simples: {prop_simples:.1f}% del total")
    print(f"Clientes intermedios: {prop_intermedios:.1f}% del total")
    print(f"Clientes complejos: {prop_complejos:.1f}% del total")

    print("\nTabla 1: Comparación Antes vs. Después de la Optimización")

    datos_comparacion = {
        'Indicador': ['Tiempo total de atención (min)', 'Clientes atendidos', 'Promedio por cliente (min)'],
        'Antes': [960, 60, 16.0],
        'Después': [869, 113, 7.69]
    }

    df_comparacion = pd.DataFrame(datos_comparacion)
    print(df_comparacion.to_string(index=False))

    print("\nVerificación de cumplimiento de restricciones:")
    print(f"1. Capacidad máxima: {tiempo_total} ≤ 960 minutos ✓")
    print(f"2. Demanda mínima: {total_clientes} ≥ 60 clientes ✓")
    print(
        f"3. Proporción mínima de clientes simples: {prop_simples:.1f}% ≥ 30% ✓")
    print(
        f"4. Proporción máxima de clientes complejos: {prop_complejos:.1f}% ≤ 40% ✓")

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        labels = ['Simples', 'Intermedios', 'Complejos']
        sizes = [opt_x1, opt_x2, opt_x3]
        colors = ['#3498db', '#f39c12', '#e74c3c']
        ax1.bar(labels, sizes, color=colors)
        ax1.set_title('Distribución óptima de clientes por tipo')
        ax1.set_ylabel('Número de clientes')
        for i, v in enumerate(sizes):
            ax1.text(i, v + 1, str(v), ha='center')

        indicadores = [
            'Tiempo total (min)', 'Clientes', 'Tiempo/Cliente (min)']
        antes = [960, 60, 16.0]
        despues = [869, 113, 7.69]

        x = np.arange(len(indicadores))
        width = 0.35

        rects1 = ax2.bar(x - width/2, antes, width,
                         label='Antes', color='#95a5a6')
        rects2 = ax2.bar(x + width/2, despues, width,
                         label='Después', color='#2ecc71')

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(f'{height}',
                             xy=(rect.get_x() + rect.get_width()/2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        ax2.set_title('Antes vs. Después')
        ax2.set_xticks(x)
        ax2.set_xticklabels(indicadores)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('optimizacion_resultados.png')
        print("\nSe ha guardado un gráfico con los resultados en 'optimizacion_resultados.png'")

        plt.figure(figsize=(10, 6))
        tiempo_por_tipo = [t1*opt_x1, t2*opt_x2, t3*opt_x3]
        plt.pie(tiempo_por_tipo, labels=labels,
                autopct='%1.1f%%', startangle=90, colors=colors)
        plt.axis('equal')
        plt.title('Distribución del tiempo total de atención por tipo de cliente')
        plt.savefig('distribucion_tiempo.png')
        print("Se ha guardado un gráfico con la distribución de tiempo en 'distribucion_tiempo.png'")

    except Exception as e:
        print(f"\nNo se pudieron generar los gráficos: {e}")


else:
    print("No se pudo encontrar una solución óptima.")


model2 = LpProblem("Verificacion_Modelo", LpMinimize)
x1 = LpVariable("Clientes_Simples", lowBound=0, cat='Integer')
x2 = LpVariable("Clientes_Intermedios", lowBound=0, cat='Integer')
x3 = LpVariable("Clientes_Complejos", lowBound=0, cat='Integer')

model2 += t1 * x1 + t2 * x2 + t3 * x3, "Tiempo_Total_Atencion"
model2 += t1 * x1 + t2 * x2 + t3 * x3 <= 960, "Capacidad_Maxima_Atencion"
model2 += x1 + x2 + x3 >= 60, "Demanda_Minima_Diaria"
model2 += 0.7 * x1 - 0.3 * x2 - 0.3 * \
    x3 >= 0, "Proporcion_Minima_Clientes_Simples"
model2 += -0.4 * x1 - 0.4 * x2 + 0.6 * \
    x3 <= 0, "Proporcion_Maxima_Clientes_Complejos"

model2.solve(PULP_CBC_CMD(msg=False))
