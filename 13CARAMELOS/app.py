from flask import Flask, render_template, request
import random
from collections import Counter
import time

app = Flask(__name__)

# Valores por defecto
DEFAULT_CARAMELOS = ["fresa", "huevo", "pera"]
DEFAULT_JUGADORES = 6
DEFAULT_CARAMELOS_POR_PERSONA = 2
DEFAULT_TIEMPO_LIMITE = 30
DEFAULT_MAX_INTERCAMBIOS = 10


def repartir_caramelos(cantidad_jugadores, tipos_caramelo, caramelos_por_persona):
    """Reparte caramelos aleatorios a cada jugador."""
    inventario = []
    for _ in range(cantidad_jugadores):
        bolsa = [random.choice(tipos_caramelo)
                 for _ in range(caramelos_por_persona)]
        inventario.append(bolsa)
    return inventario


def calcular_max_chupetines(inventario_global, tipos_caramelo):
    """Calcula cuántos chupetines se pueden hacer (1 por trio de sabores distintos)."""
    conteo = Counter(inventario_global)
    return min(conteo[tipo] for tipo in tipos_caramelo)


def simular_intercambios(jugadores, tipos_caramelo, max_intercambios, tiempo_limite):
    """Simula una ronda de intercambios con límites de tiempo e intentos."""
    if max_intercambios == 0 and tiempo_limite == 0:
        return 0  # Sin restricción, no simulamos

    inicio = time.time()
    intercambios_realizados = 0
    turno = 0

    while (max_intercambios == 0 or intercambios_realizados < max_intercambios):
        if tiempo_limite > 0 and time.time() - inicio > tiempo_limite:
            break

        idx1 = turno % len(jugadores)
        idx2 = (turno + 1) % len(jugadores)

        if jugadores[idx1] and jugadores[idx2]:
            caramelo1 = jugadores[idx1].pop(0)
            caramelo2 = jugadores[idx2].pop(0)
            jugadores[idx1].append(caramelo2)
            jugadores[idx2].append(caramelo1)
            intercambios_realizados += 1

        turno += 1

    return intercambios_realizados


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    form_data = {}

    if request.method == "POST":
        try:
            # Obtener datos del formulario
            jugadores = int(request.form.get("jugadores", DEFAULT_JUGADORES))
            tipos_caramelo = request.form.get(
                "tipos_caramelo", ",".join(DEFAULT_CARAMELOS)).split(",")
            tipos_caramelo = [t.strip() for t in tipos_caramelo if t.strip()]
            caramelos_por_persona = int(request.form.get(
                "caramelos_por_persona", DEFAULT_CARAMELOS_POR_PERSONA))
            tiempo_limite = int(request.form.get(
                "tiempo_limite", DEFAULT_TIEMPO_LIMITE))
            max_intercambios = int(request.form.get(
                "max_intercambios", DEFAULT_MAX_INTERCAMBIOS))

            # Guardar valores para rellenar el formulario
            form_data.update({
                "jugadores": jugadores,
                "tipos_caramelo": ",".join(tipos_caramelo),
                "caramelos_por_persona": caramelos_por_persona,
                "tiempo_limite": tiempo_limite,
                "max_intercambios": max_intercambios
            })

            # Validar que haya suficientes tipos de caramelos
            if len(tipos_caramelo) < 3:
                raise ValueError(
                    "Se necesitan al menos 3 tipos de caramelos diferentes.")

            # Repartir caramelos
            inventario = repartir_caramelos(
                jugadores, tipos_caramelo, caramelos_por_persona)

            # Calcular máximo inicial de chupetines
            inventario_global = [c for bolsa in inventario for c in bolsa]
            max_posible_inicial = calcular_max_chupetines(
                inventario_global, tipos_caramelo)

            # Simular intercambios
            intercambios_realizados = simular_intercambios(
                inventario, tipos_caramelo, max_intercambios, tiempo_limite)

            # Calcular máximos después de intercambios
            inventario_final = [c for bolsa in inventario for c in bolsa]
            max_posible_final = calcular_max_chupetines(
                inventario_final, tipos_caramelo)

            # Resultado final
            resultado = {
                "inventario_inicial": inventario,
                "max_posible_inicial": max_posible_inicial,
                "intercambios_realizados": intercambios_realizados,
                "max_posible_final": max_posible_final
            }

        except Exception as e:
            resultado = {"error": str(e)}

    return render_template("index.html", resultado=resultado, form=form_data)


def main():
    """Función principal que inicia la aplicación Flask."""
    print("🚀 Iniciando aplicación Flask...")
    app.run(debug=True)


if __name__ == "__main__":
    main()
