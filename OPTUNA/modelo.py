import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Modelos y métricas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(csv_path):
    """Carga y prepara los datos"""
    # Cargar datos
    try:
        df = pd.read_csv(csv_path, encoding='cp1252', delimiter=';')
    except:
        df = pd.read_csv(csv_path, encoding='utf-8', delimiter=';')
    
    print("🔍 Información del Dataset:")
    print(f"   • Filas: {df.shape[0]}")
    print(f"   • Columnas: {df.shape[1]}")
    
    # Limpiar nombres de columnas
    df.columns = df.columns.str.strip()
    
    # Convertir fechas
    df['FECHA_CORTE'] = pd.to_datetime(df['FECHA_CORTE'])
    df['FECHA_EGRESO'] = pd.to_datetime(df['FECHA_EGRESO'], errors='coerce')
    
    # Crear variable objetivo: días desde el egreso
    df['DIAS_DESDE_EGRESO'] = (df['FECHA_CORTE'] - df['FECHA_EGRESO']).dt.days
    
    # Eliminar filas con fechas faltantes
    df = df.dropna(subset=['FECHA_EGRESO', 'DIAS_DESDE_EGRESO'])
    
    print(f"\n📊 Variable objetivo - Días desde egreso:")
    print(f"   • Promedio: {df['DIAS_DESDE_EGRESO'].mean():.1f} días")
    print(f"   • Mínimo: {df['DIAS_DESDE_EGRESO'].min()} días")
    print(f"   • Máximo: {df['DIAS_DESDE_EGRESO'].max()} días")
    
    return df

def prepare_features(df):
    """Prepara las características para el modelo"""
    # Características categóricas
    categorical_features = [
        'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 
        'ESCUELA_PROFESIONAL', 'PERIODO_EGRESO', 'SEDE'
    ]
    
    X = df[categorical_features].copy()
    y = df['DIAS_DESDE_EGRESO'].copy()
    
    # Codificar variables categóricas
    label_encoders = {}
    for col in categorical_features:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col].astype(str))
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n🔄 División de datos:")
    print(f"   • Entrenamiento: {X_train.shape[0]} muestras")
    print(f"   • Prueba: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test, label_encoders

def objective(trial, X_train, y_train):
    """Función objetivo para Optuna"""
    # Hiperparámetros a optimizar
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # Crear modelo
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    
    # Validación cruzada
    scores = cross_val_score(model, X_train, y_train, 
                           cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return -scores.mean()

def optimize_model(X_train, y_train, n_trials=50):
    """Optimiza hiperparámetros con Optuna"""
    print(f"\n🚀 Optimizando hiperparámetros ({n_trials} trials)...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), 
                  n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n✅ Mejor MSE encontrado: {study.best_value:.4f}")
    print(f"🎯 Mejores hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"   • {key}: {value}")
    
    # Entrenar modelo final
    best_model = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    return best_model, study.best_params

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evalúa el modelo"""
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred)
    }
    
    print(f"\n📈 Métricas del Modelo:")
    print(f"   • R² Entrenamiento: {metrics['train_r2']:.4f}")
    print(f"   • R² Prueba: {metrics['test_r2']:.4f}")
    print(f"   • MAE Entrenamiento: {metrics['train_mae']:.2f} días")
    print(f"   • MAE Prueba: {metrics['test_mae']:.2f} días")
    print(f"   • RMSE Prueba: {np.sqrt(metrics['test_mse']):.2f} días")
    
    return metrics, y_test_pred

def plot_results(df, model, X_test, y_test, y_pred, metrics):
    """Genera gráficos de resultados"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 Resultados del Modelo Random Forest - Egresados UNDC', 
                 fontsize=16, fontweight='bold')
    
    # 1. Distribución de días desde egreso
    axes[0, 0].hist(df['DIAS_DESDE_EGRESO'], bins=15, alpha=0.7, 
                    color='skyblue', edgecolor='black')
    axes[0, 0].set_title('📈 Distribución: Días desde Egreso')
    axes[0, 0].set_xlabel('Días')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Predicciones vs Valores Reales
    axes[0, 1].scatter(y_test, y_pred, alpha=0.7, color='coral', s=50)
    axes[0, 1].plot([y_test.min(), y_test.max()], 
                   [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_title(f'🎯 Predicciones vs Reales\n(R² = {metrics["test_r2"]:.3f})')
    axes[0, 1].set_xlabel('Valores Reales (días)')
    axes[0, 1].set_ylabel('Predicciones (días)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribución por Escuela Profesional
    escuela_counts = df['ESCUELA_PROFESIONAL'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(escuela_counts)))
    wedges, texts, autotexts = axes[1, 0].pie(escuela_counts.values, 
                                             labels=escuela_counts.index, 
                                             autopct='%1.1f%%',
                                             colors=colors)
    axes[1, 0].set_title('🎓 Distribución por Escuela')
    
    # 4. Importancia de Características
    feature_names = ['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 
                    'ESCUELA_PROF.', 'PERIODO_EGRESO', 'SEDE']
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    axes[1, 1].bar(range(len(importances)), importances[indices], 
                   color='gold', alpha=0.8, edgecolor='black')
    axes[1, 1].set_title('⭐ Importancia de Características')
    axes[1, 1].set_xticks(range(len(importances)))
    axes[1, 1].set_xticklabels([feature_names[i] for i in indices], 
                              rotation=45, ha='right')
    axes[1, 1].set_ylabel('Importancia')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Gráfico adicional: Días promedio por escuela
    plt.figure(figsize=(12, 6))
    avg_days = df.groupby('ESCUELA_PROFESIONAL')['DIAS_DESDE_EGRESO'].mean().sort_values()
    
    bars = plt.barh(range(len(avg_days)), avg_days.values, color='lightgreen', 
                    alpha=0.8, edgecolor='black')
    plt.yticks(range(len(avg_days)), 
               [label.replace('Ó', 'O').replace('Í', 'I') for label in avg_days.index])
    plt.xlabel('Días Promedio desde Egreso')
    plt.title('📅 Días Promedio desde Egreso por Escuela Profesional', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Agregar valores en las barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height()/2, 
                f'{width:.0f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main():
    """Función principal"""
    # Ruta del archivo CSV
    csv_path = 'dataset.csv'
    
    print("🎓 MODELO DE REGRESIÓN RANDOM FOREST - EGRESADOS UNDC")
    print("=" * 60)
    
    # Cargar datos
    df = load_data(csv_path)
    
    # Preparar características
    X_train, X_test, y_train, y_test, encoders = prepare_features(df)
    
    # Optimizar modelo
    best_model, best_params = optimize_model(X_train, y_train, n_trials=50)
    
    # Evaluar modelo
    metrics, y_pred = evaluate_model(best_model, X_train, X_test, y_train, y_test)
    
    # Generar gráficos
    plot_results(df, best_model, X_test, y_test, y_pred, metrics)
    
    print(f"\n✅ ¡Análisis completado!")
    print(f"🎯 El modelo puede predecir los días desde egreso con un R² de {metrics['test_r2']:.3f}")
    print(f"📏 Error promedio de {metrics['test_mae']:.1f} días")

if __name__ == "__main__":
    main()