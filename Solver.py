import numpy as np
import pandas as pd
from scipy.optimize import minimize

print("\n=== OPTIMIZADOR: ENTRADA DIRECTA POR TERMINAL ===")


def leer_bloque_texto(mensaje):
    """Permite al usuario pegar un bloque de texto multilínea en la terminal."""
    print(mensaje)
    print(
        "  (Pega tus datos, presiona Enter, escribe la palabra FIN y presiona Enter de nuevo)"
    )
    lineas = []
    while True:
        linea = input()
        if linea.strip().upper() == "FIN":
            break
        if linea.strip():
            lineas.append(linea.strip())
    return lineas


# ==========================================
# PASO 1: LEER RENDIMIENTOS (STATS)
# ==========================================
lineas_stats = leer_bloque_texto(
    "\nPASO 1: Pega aquí las columnas de Nombres y Medias (sin encabezados):"
)

nombres = []
retornos = []

try:
    for linea in lineas_stats:
        partes = linea.split()
        if len(partes) >= 2:
            nombres.append(partes[0])
            val_str = partes[-1].replace("%", "")
            retornos.append(float(val_str) / 100.0)

    retornos_array = np.array(retornos)
    num_activos = len(nombres)
    print(f"  [OK] {num_activos} activos leídos perfectamente: {nombres}")

except Exception as e:
    print(f"\n  [!] Hubo un error al procesar el texto: {e}")
    exit()

# ==========================================
# PASO 2: LEER MATRIZ DE COVARIANZAS
# ==========================================
lineas_matriz = leer_bloque_texto(
    "\nPASO 2: Pega aquí SOLO el bloque de números de tu matriz de covarianzas:"
)

filas_matriz = []

try:
    for linea in lineas_matriz:
        valores_fila = [float(v) for v in linea.split()]
        filas_matriz.append(valores_fila)

    cov_matrix = np.array(filas_matriz)

    if cov_matrix.shape != (num_activos, num_activos):
        print(f"\n  [!] Error: Leí una matriz de {cov_matrix.shape}.")
        print(f"      Debería ser cuadrada de {num_activos}x{num_activos}.")
        exit()

    print(f"  [OK] Matriz de {cov_matrix.shape} cargada exitosamente.")

except Exception as e:
    print(f"\n  [!] Hubo un error al procesar los números de la matriz: {e}")
    exit()

# ==========================================
# PASO 3: OPTIMIZACIÓN (SOLVERS)
# ==========================================
print("\nEjecutando algoritmos de optimización...")

restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
limites = tuple((0.0001, 1.0) for _ in range(num_activos))
pesos_iniciales = np.ones(num_activos) / num_activos


# MODIFICACIÓN CLAVE: Escalar la varianza multiplicando por 1,000,000
def min_riesgo(w):
    return (np.dot(w.T, np.dot(cov_matrix, w))) * 1000000


def max_rendimiento(w):
    return -np.dot(w.T, retornos_array)


# MODIFICACIÓN CLAVE: Añadir tol=1e-10 para mayor precisión
res_min = minimize(
    min_riesgo,
    pesos_iniciales,
    method="SLSQP",
    bounds=limites,
    constraints=restricciones,
    tol=1e-10,
)
res_max = minimize(
    max_rendimiento,
    pesos_iniciales,
    method="SLSQP",
    bounds=limites,
    constraints=restricciones,
    tol=1e-10,
)

# ==========================================
# PASO 4: RESULTADOS
# ==========================================
df_resultados = pd.DataFrame(
    {
        "Activo": nombres,
        "MinRisk (W)": np.round(res_min.x, 4),
        "MaxProfit (W)": np.round(res_max.x, 4),
    }
)

print("\n=== RESULTADOS FINALES ===")
print(df_resultados.to_string(index=False))

try:
    df_resultados.to_clipboard(index=False)
    print(
        "\n[ÉXITO] ¡Pesos óptimos copiados al portapapeles! Ve a Excel y presiona Ctrl+V."
    )
except Exception:
    print("\n[!] Copia manualmente la tabla de arriba.")
