import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

# Ignorar advertencias, útil para evitar mensajes de optimización o pandas que no son críticos.
warnings.filterwarnings('ignore')

class OptimaBattleQuickSolver:
    """
    Clase para resolver rápidamente el problema de optimización de portafolios en OptimaBattle.
    Permite cargar datos, pre-filtrar activos y optimizar la asignación de capital
    considerando retorno, riesgo, liquidez y restricciones sectoriales/beta.
    """
    def __init__(self):
        """
        Inicializa el OptimaBattleQuickSolver con los parámetros del concurso.
        """
        self.presupuesto = 1_000_000  # Presupuesto total disponible para invertir (S/.).
        self.lambda_riesgo = 0.5    # Coeficiente de aversión al riesgo. Un valor más alto
                                    # prioriza portafolios con menor riesgo.
        self.max_sector = 0.30      # Máxima ponderación permitida para cualquier sector (30%).
        self.min_activos = 5        # Número mínimo de activos requeridos en el portafolio final.
        self.max_beta = 1.2         # Beta máximo permitido para el portafolio total.
        self.sectores = {1: 'Tech', 2: 'Salud', 3: 'Energía', 4: 'Financiero', 5: 'Consumo'}
                                    # Diccionario para mapear IDs de sector a nombres legibles.
        self.datos = None           # DataFrame que almacenará los datos de los activos.

    def cargar_datos(self, archivo='Ronda1.xlsx'):
        """
        Carga los datos de los activos desde un archivo Excel.
        Realiza la conversión de retornos y volatilidad a decimal si están en porcentaje.

        Args:
            archivo (str): Nombre del archivo Excel a cargar.

        Returns:
            pd.DataFrame: DataFrame con los datos de los activos o None si hay un error.
        """
        try:
            self.datos = pd.read_excel(archivo)
            print(f"✓ Cargados {len(self.datos)} activos desde {archivo}")
        except FileNotFoundError:
            print(f"❌ Error: El archivo '{archivo}' no se encontró.")
            return None
        except Exception as e:
            print(f"❌ Error cargando {archivo}: {str(e)}")
            return None

        # Convertir a porcentaje si los valores son > 2 (asumiendo que 1% o 2% son raros,
        # pero 10% o 20% son comunes y deben dividirse por 100).
        if self.datos['retorno_esperado'].max() > 2:
            self.datos['retorno_esperado'] = self.datos['retorno_esperado'] / 100
        if self.datos['volatilidad'].max() > 2:
            self.datos['volatilidad'] = self.datos['volatilidad'] / 100

        return self.datos

    def analizar_y_filtrar(self):
        """
        Realiza un análisis inicial y pre-selección de activos basándose en criterios
        de eficiencia, utilidad individual, beta y liquidez. Selecciona los mejores
        activos por sector y como respaldo, los mejores en general.

        Returns:
            pd.DataFrame: DataFrame con los activos pre-seleccionados.
        """
        print("\n🔍 ANÁLISIS RÁPIDO DE ACTIVOS")
        print("="*50)

        # Calcular métricas de eficiencia para cada activo.
        # Eficiencia: Retorno Esperado / Varianza (Retorno esperado / Volatilidad^2). Mayor es mejor.
        self.datos['eficiencia'] = self.datos['retorno_esperado'] / (self.datos['volatilidad'] ** 2)
        # Sharpe Ratio: Retorno Esperado / Volatilidad. Mayor es mejor.
        self.datos['sharpe_ratio'] = self.datos['retorno_esperado'] / self.datos['volatilidad']
        # Utilidad individual: Una medida personalizada que pondera el retorno frente al riesgo,
        # utilizando el coeficiente de aversión al riesgo. Mayor es mejor.
        self.datos['utilidad_individual'] = self.datos['retorno_esperado'] - self.lambda_riesgo * (self.datos['volatilidad'] ** 2)
        # Ranking de eficiencia: Asigna un rango a los activos basado en su eficiencia (el más eficiente es 1).
        self.datos['ranking_eficiencia'] = self.datos['eficiencia'].rank(ascending=False)

        # Definir criterios de filtrado iniciales para identificar candidatos.
        criterios = (
            (self.datos['utilidad_individual'] > 0) &      # La utilidad individual debe ser positiva.
            (self.datos['beta'] <= 1.5) &                  # El beta individual del activo no debe superar 1.5.
            (self.datos['liquidez_score'] >= 6)            # El score de liquidez debe ser al menos 6.
        )
        candidatos = self.datos[criterios].copy() # Se crea un subconjunto de datos con los activos que cumplen los criterios.

        mejores_por_sector = []
        # Iterar por cada sector definido para seleccionar los 3 mejores activos por eficiencia.
        for sector_id in range(1, 6): # Asume 5 sectores (1 a 5).
            sector_data = candidatos[candidatos['sector'] == sector_id]
            if not sector_data.empty: # Si hay datos en ese sector
                # Selecciona los 3 activos más eficientes dentro de ese sector.
                mejores_sector = sector_data.nlargest(3, 'eficiencia')
                mejores_por_sector.append(mejores_sector)

        # Concatenar los activos seleccionados por sector.
        if mejores_por_sector:
            activos_seleccionados = pd.concat(mejores_por_sector).drop_duplicates()
        else:
            # Si por alguna razón no se seleccionó nada por sector (ej. no hay datos que pasen los filtros),
            # se toman los 10 mejores activos en general por eficiencia.
            activos_seleccionados = candidatos.nlargest(10, 'eficiencia')

        print(f"✓ Pre-seleccionados {len(activos_seleccionados)} activos prometedores")
        return activos_seleccionados

    def optimizar_rapido(self, activos_filtrados):
        """
        Realiza la optimización del portafolio utilizando los activos filtrados.
        Define la función objetivo (maximizar utilidad del portafolio) y las restricciones
        (suma de pesos = 1, beta del portafolio, límite por sector).

        Args:
            activos_filtrados (pd.DataFrame): DataFrame con los activos pre-seleccionados.

        Returns:
            tuple: (pesos_optimos, activos_usados) si la optimización es exitosa,
                   None en caso contrario.
        """
        print(f"\n⚡ OPTIMIZACIÓN RÁPIDA")
        print("="*30)

        datos_opt = activos_filtrados.copy()
        n = len(datos_opt) # Número de activos a optimizar.

        if n < self.min_activos:
            print(f"❌ Necesito al menos {self.min_activos} activos. Solo hay {n} candidatos.")
            return None

        def objetivo(pesos):
            """
            Función objetivo para la optimización. Se busca maximizar la utilidad del portafolio
            (retorno esperado - lambda_riesgo * riesgo). Como `minimize` busca el mínimo,
            se retorna el negativo de la utilidad.
            """
            retorno_portafolio = np.sum(pesos * datos_opt['retorno_esperado'])
            # Asunción simplificada de riesgo (volatilidad ponderada por pesos al cuadrado).
            # Para un cálculo de riesgo más preciso, se necesitaría la matriz de covarianza.
            riesgo_portafolio = np.sum(pesos**2 * datos_opt['volatilidad']**2)
            utilidad = retorno_portafolio - self.lambda_riesgo * riesgo_portafolio
            return -utilidad # Se minimiza el negativo de la utilidad.

        # Definir restricciones para la optimización.
        restricciones = [
            # Restricción de igualdad: La suma de los pesos debe ser igual a 1 (100% de la inversión).
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]

        # Restricción de desigualdad: El beta total del portafolio no debe exceder self.max_beta.
        # beta_portafolio = sum(peso_i * beta_i para todos los activos)
        restricciones.append({'type': 'ineq', 'fun': lambda x: self.max_beta - np.sum(x * datos_opt['beta'])})

        # Restricciones de desigualdad para la asignación máxima por sector.
        for sector_id in range(1, 6):
            mask = datos_opt['sector'] == sector_id # Máscara para activos de un sector específico.
            if mask.any(): # Solo si hay activos de este sector en la selección actual.
                # La suma de pesos para los activos de este sector no debe exceder self.max_sector.
                restricciones.append({'type': 'ineq', 'fun': lambda x, m=mask: self.max_sector - np.sum(x[m])})

        # Pesos iniciales para el algoritmo de optimización (distribución equitativa).
        pesos_ini = np.ones(n) / n
        # Límites para los pesos: cada peso debe estar entre 0 (no invertir) y 1 (invertir todo en un activo).
        limites = [(0, 1) for _ in range(n)]

        # Ejecutar el algoritmo de optimización.
        # method='SLSQP' es adecuado para problemas con igualdad y desigualdad.
        resultado = minimize(objetivo, pesos_ini, method='SLSQP', bounds=limites, constraints=restricciones)

        if resultado.success:
            pesos_optimos = resultado.x
            # Filtrar activos con pesos significativos (ej. > 0.1% del portafolio)
            mask_significativo = pesos_optimos > 0.001
            activos_usados = datos_opt[mask_significativo].copy()
            pesos_usados = pesos_optimos[mask_significativo]

            print("\n📏 DETALLES DE RESTRICCIONES CUMPLIDAS:")
            print("-" * 60)
            beta_total = np.sum(pesos_usados * activos_usados['beta'])
            print(f"🔄 Beta total del portafolio: {beta_total:.4f} (Objetivo <= {self.max_beta})")

            # Mostrar la asignación por sector para verificar las restricciones.
            for sector_id in sorted(activos_usados['sector'].unique()):
                mask_sector = activos_usados['sector'] == sector_id
                peso_sector = np.sum(pesos_usados[mask_sector])
                sector_nombre = self.sectores.get(sector_id, f"Sector {sector_id}")
                print(f"🏭 Sector {sector_id} ({sector_nombre}): {peso_sector*100:.2f}% del portafolio (Objetivo <= {self.max_sector*100:.0f}%)")

            print("-" * 60)

            return pesos_usados, activos_usados
        else:
            print(f"❌ Error en la optimización: {resultado.message}")
            return None

    def generar_recomendaciones(self, pesos, activos):
        """
        Genera un informe detallado de las recomendaciones de inversión,
        calculando el monto a invertir y el número de acciones para cada activo.
        También resume las características clave del portafolio final.

        Args:
            pesos (np.array): Pesos óptimos de los activos en el portafolio.
            activos (pd.DataFrame): DataFrame de los activos correspondientes a los pesos.

        Returns:
            list: Lista de diccionarios, cada uno con los detalles de la inversión por activo.
        """
        print(f"\n🎯 RECOMENDACIONES DE INVERSIÓN")
        print("="*60)

        recomendaciones = []
        for i, (_, activo) in enumerate(activos.iterrows()):
            peso = pesos[i]
            monto_ideal_inversion = peso * self.presupuesto
            precio_accion = activo['precio_accion']

            # Calcular el número de acciones enteras y el monto real a invertir.
            num_acciones = int(monto_ideal_inversion / precio_accion)
            monto_real = num_acciones * precio_accion

            # Añadir detalles a la lista de recomendaciones.
            recomendaciones.append({
                'activo_id': activo['activo_id'],
                'sector': f"{activo['sector']} - {self.sectores.get(activo['sector'], 'Desconocido')}",
                'peso_optimo': peso,
                'peso_porcentaje': peso * 100,
                'monto_inversion': monto_real,
                'num_acciones': num_acciones,
                'precio_accion': precio_accion,
                'retorno_esperado': activo['retorno_esperado'] * 100, # Vuelve a porcentaje para mostrar
                'volatilidad': activo['volatilidad'] * 100,         # Vuelve a porcentaje para mostrar
                'beta': activo['beta'],
                'inversion_minima': activo['min_inversion'],
                'cumple_minimo': monto_real >= activo['min_inversion'],
                'liquidez_score': activo['liquidez_score']
            })

        # Recalcular métricas del portafolio con los montos reales invertidos
        # para un resumen más preciso.
        # Se necesita normalizar los pesos reales basados en el monto_real.
        total_invertido = sum(r['monto_inversion'] for r in recomendaciones)
        if total_invertido == 0:
            print("No se realizaron inversiones significativas.")
            return []

        # Calcular los pesos reales basados en el monto_real invertido
        pesos_reales = np.array([r['monto_inversion'] for r in recomendaciones]) / total_invertido
        retornos_activos = np.array([r['retorno_esperado'] / 100 for r in recomendaciones])
        volatilidades_activos = np.array([r['volatilidad'] / 100 for r in recomendaciones])
        betas_activos = np.array([r['beta'] for r in recomendaciones])

        retorno_portafolio = np.sum(pesos_reales * retornos_activos)
        volatilidad_portafolio = np.sqrt(np.sum(pesos_reales**2 * volatilidades_activos**2))
        beta_portafolio = np.sum(pesos_reales * betas_activos)


        print(f"\n--- Resumen del Portafolio ---")
        print(f"💹 Retorno esperado del portafolio: {retorno_portafolio*100:.2f}%")
        print(f"📈 Volatilidad del portafolio: {volatilidad_portafolio*100:.2f}%")
        print(f"🔄 Beta promedio del portafolio: {beta_portafolio:.3f}")
        print(f"💰 Total invertido: S/.{total_invertido:,.0f}")
        print(f"💸 Dinero sobrante: S/.{self.presupuesto - total_invertido:,.0f}")
        print(f"🏢 Número de activos en el portafolio: {len(recomendaciones)}")
        print("------------------------------")

        return recomendaciones

    def ejecutar_solver_completo(self, archivo='Ronda1.xlsx'):
        """
        Ejecuta el proceso completo de optimización del portafolio:
        carga de datos -> análisis y filtrado -> optimización -> generación de recomendaciones.

        Args:
            archivo (str): Nombre del archivo Excel con los datos de los activos.

        Returns:
            list: Lista de recomendaciones de inversión si el proceso es exitoso,
                  None en caso contrario.
        """
        print("🚀 OPTIMABATTLE QUICK SOLVER")
        print("="*50)

        datos = self.cargar_datos(archivo)
        if datos is None:
            return None

        activos_filtrados = self.analizar_y_filtrar()
        # Si no hay suficientes activos filtrados, se detiene.
        if activos_filtrados is None or len(activos_filtrados) < self.min_activos:
            print(f"❌ No hay suficientes activos prometedores después del filtrado. Solo hay {len(activos_filtrados) if activos_filtrados is not None else 0} activos.")
            return None

        resultado_opt = self.optimizar_rapido(activos_filtrados)

        if resultado_opt:
            pesos, activos = resultado_opt
            recomendaciones = self.generar_recomendaciones(pesos, activos)
            return recomendaciones
        else:
            print(f"\n❌ El proceso de optimización no pudo completarse.")
            return None

# --- Uso del solver ---
if __name__ == "__main__":
    solver = OptimaBattleQuickSolver()
    # Asegúrate de que 'Ronda1.xlsx' esté en el mismo directorio que este script,
    # o proporciona la ruta completa al archivo.
    recomendaciones_finales = solver.ejecutar_solver_completo('Ronda1.xlsx')

    if recomendaciones_finales:
        print("\n\n🎯 ACCIONES FINALES PARA INVERTIR (RESUMEN):")
        print("="*60)
        # Ordenar por monto de inversión descendente para mayor claridad
        recomendaciones_finales.sort(key=lambda x: x['monto_inversion'], reverse=True)
        for i, rec in enumerate(recomendaciones_finales, 1):
            if rec['num_acciones'] > 0: # Solo mostrar si se compra al menos una acción
                print(f"{i}. Activo ID: {rec['activo_id']} | Sector: {rec['sector']} | "
                      f"Invertir: S/.{rec['monto_inversion']:,.0f} | Acciones: {rec['num_acciones']}")
                # print(f"   Peso: {rec['peso_porcentaje']:.2f}% | Retorno Exp: {rec['retorno_esperado']:.2f}% | Volatilidad: {rec['volatilidad']:.2f}% | Beta: {rec['beta']:.2f}")
        print("="*60)
    else:
        print("\nNo se pudieron generar recomendaciones de inversión.")