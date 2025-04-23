from flask import Flask, render_template, request, redirect, url_for
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            n = int(request.form['variables'])
            ecuaciones = []

            # Validar que n esté en un rango razonable
            if n < 1 or n > 10:
                raise ValueError(
                    "El número de incógnitas debe estar entre 1 y 10")

            # Recoger coeficientes
            for i in range(n):
                coeficientes = list(
                    map(float, request.form[f'ecuacion_{i}'].split()))
                if len(coeficientes) != n:
                    raise ValueError(
                        f"La ecuación {i+1} debe tener exactamente {n} coeficientes")
                ecuaciones.append(coeficientes)

            # Recoger términos independientes
            b_terms = list(
                map(float, request.form['terminos_independientes'].split()))
            if len(b_terms) != n:
                raise ValueError(
                    f"Debe ingresar exactamente {n} términos independientes")

            # Convertir a arrays numpy y verificar dimensiones
            A = np.array(ecuaciones)
            b = np.array(b_terms)

            if A.ndim != 2 or A.shape != (n, n):
                raise ValueError(
                    "La matriz de coeficientes no tiene las dimensiones correctas")

            if b.ndim != 1 or len(b) != n:
                raise ValueError(
                    "El vector de términos independientes no tiene la dimensión correcta")

            # Guardar en sesión para la página de resultados
            return redirect(url_for('resolver', n=n, ecuaciones=str(ecuaciones), b_terms=str(b_terms)))

        except ValueError as e:
            error = str(e)
            return render_template('index.html', error=error)

    return render_template('index.html')


@app.route('/resolver')
def resolver():
    try:
        n = int(request.args.get('n'))
        ecuaciones = eval(request.args.get('ecuaciones'))
        b_terms = eval(request.args.get('b_terms'))

        A = np.array(ecuaciones)
        b = np.array(b_terms)

        # Verificar dimensiones nuevamente por seguridad
        if A.shape != (n, n) or len(b) != n:
            raise ValueError("Dimensiones incorrectas de los datos")

        solucion = np.linalg.solve(A, b)
        resultados = [f"x{i+1} = {solucion[i]:.4f}" for i in range(n)]

        return render_template('resultado.html', resultados=resultados)

    except np.linalg.LinAlgError as e:
        error = f"El sistema no tiene solución única: {str(e)}"
        return render_template('resultado.html', error=error)

    except Exception as e:
        error = f"Error al resolver el sistema: {str(e)}"
        return render_template('resultado.html', error=error)


if __name__ == '__main__':
    app.run(debug=True)
