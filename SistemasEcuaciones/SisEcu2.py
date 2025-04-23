import numpy as np
from scipy.optimize import linprog


def resolver_sistema():
    print("\nSistema de Ecuaciones Lineales con NumPy")
    print("1. Resolver sistema de ecuaciones")
    print("2. Resolver problema de programación lineal")
    opcion = int(input("Seleccione una opción (1 o 2): "))

    if opcion == 1:
        resolver_ecuaciones()
    elif opcion == 2:
        resolver_programacion_lineal()
    else:
        print("Opción no válida")


def resolver_ecuaciones():
    n = int(input("\nIngrese el número de incógnitas: "))

    A = []
    b = []

    print("\nIngrese los coeficientes de cada ecuación:")
    for i in range(n):
        fila = list(map(float, input(
            f"Ingresa los coeficientes de la ecuación {i+1} separados por espacio: ").split()))
        A.append(fila)

    print("\nIngrese los términos independientes:")
    b = list(map(float, input("Separados por espacio: ").split()))

    A = np.array(A)
    b = np.array(b)

    try:
        solucion = np.linalg.solve(A, b)
        print("\nSolución del sistema:")
        for i in range(n):
            print(f"x{i+1} = {solucion[i]}")
    except np.linalg.LinAlgError as e:
        print("\nEl sistema no tiene solución única:")
        print(str(e))


def resolver_programacion_lineal():
    print("\nProgramación Lineal")
    print("Formato: Minimizar c^T x sujeto a A_ub x <= b_ub, A_eq x = b_eq")

    n = int(input("\nIngrese el número de variables: "))
    c = list(map(float, input(
        "Ingrese los coeficientes de la función objetivo (separados por espacio): ").split()))

    print("\nRestricciones de desigualdad (<=):")
    m_ub = int(input("Ingrese el número de restricciones de desigualdad: "))
    A_ub = []
    b_ub = []
    if m_ub > 0:
        print("Ingrese los coeficientes para cada restricción (lado izquierdo):")
        for i in range(m_ub):
            fila = list(map(float, input(
                f"Restricción {i+1} (separados por espacio): ").split()))
            A_ub.append(fila)
        print("Ingrese los valores del lado derecho (b_ub):")
        b_ub = list(map(float, input("Separados por espacio: ").split()))

    print("\nRestricciones de igualdad:")
    m_eq = int(input("Ingrese el número de restricciones de igualdad: "))
    A_eq = []
    b_eq = []
    if m_eq > 0:
        print("Ingrese los coeficientes para cada restricción (lado izquierdo):")
        for i in range(m_eq):
            fila = list(map(float, input(
                f"Restricción {i+1} (separados por espacio): ").split()))
            A_eq.append(fila)
        print("Ingrese los valores del lado derecho (b_eq):")
        b_eq = list(map(float, input("Separados por espacio: ").split()))

    print("\nLímites de las variables (dejar en blanco para 0 <= x_i <= inf):")
    bounds = []
    for i in range(n):
        bound_input = input(
            f"Límites para x{i+1} (ej. '0 10' para 0 <= x{i+1} <= 10): ")
        if bound_input:
            bounds.append(tuple(map(float, bound_input.split())))
        else:
            bounds.append((0, None))

    try:
        res = linprog(c, A_ub=A_ub if m_ub > 0 else None, b_ub=b_ub if m_ub > 0 else None,
                      A_eq=A_eq if m_eq > 0 else None, b_eq=b_eq if m_eq > 0 else None,
                      bounds=bounds, method='highs')

        print("\nResultado de la optimización:")
        if res.success:
            print("Solución óptima encontrada:")
            for i in range(n):
                print(f"x{i+1} = {res.x[i]}")
            print(f"Valor de la función objetivo: {res.fun}")
        else:
            print("No se encontró solución óptima:")
            print(res.message)
    except Exception as e:
        print("\nError al resolver el problema:")
        print(str(e))


if __name__ == "__main__":
    resolver_sistema()
