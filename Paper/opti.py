from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value

model = LpProblem("Optimizacion_Atencion_Fotocopias", LpMinimize)

x1 = LpVariable("Clientes_Simples", lowBound=0, cat='Integer')
x2 = LpVariable("Clientes_Intermedios", lowBound=0, cat='Integer')
x3 = LpVariable("Clientes_Complejos", lowBound=0, cat='Integer')

# Función objetivo
model += 4 * x1 + 8 * x2 + 15 * x3, "Tiempo_Total"

# Restricción de capacidad máxima diaria
model += 4 * x1 + 8 * x2 + 15 * x3 <= 960, "Tiempo_Maximo"

# Restricción de demanda diaria mínima
model += x1 + x2 + x3 >= 60, "Demanda_Minima"

# Restricciones de proporción
model += 0.7 * x1 - 0.3 * x2 - 0.3 * x3 >= 0, "Proporcion_Min_Simples"
model += -0.4 * x1 - 0.4 * x2 + 0.6 * x3 <= 0, "Proporcion_Max_Complejos"

model.solve()

print("Resultados óptimos:")
print(f"Clientes con pedidos simples: {x1.varValue}")
print(f"Clientes con pedidos intermedios: {x2.varValue}")
print(f"Clientes con pedidos complejos: {x3.varValue}")
print(f"Tiempo total de atención: {value(model.objective)} minutos")
