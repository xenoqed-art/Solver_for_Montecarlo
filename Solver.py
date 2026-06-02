import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================
EPSILON = 0.01  # 1% para desigualdades estrictas (> 0)
TOLERANCIA = 1e-9


# ==========================================
# MÓDULO 1: INGESTA Y LIMPIEZA
# ==========================================
def limpiar_numero(texto):
    texto = texto.strip()
    if texto.endswith("-"):
        texto = "-" + texto[:-1]
    return float(texto)


def leer_bloque_texto(mensaje):
    print(mensaje)
    print("  (Pega datos, presiona Enter, escribe FIN y presiona Enter)")
    lineas = []
    while True:
        linea = input()
        if linea.strip().upper() == "FIN":
            break
        if linea.strip():
            lineas.append(linea.strip())
    return lineas


# ==========================================
# MÓDULO 2: NÚCLEO MATEMÁTICO
# ==========================================
def min_riesgo(w, cov_matrix):
    return (w.T @ cov_matrix @ w) * 1000000


def max_rendimiento(w, retornos):
    return -(w.T @ retornos) * 1000


# ==========================================
# MÓDULO 3: GESTOR DE REGLAS Y PUNTOS DE INICIO
# ==========================================
def generar_punto_partida(num_activos, limites_tupla):
    """Genera un x0 que respeta matemáticamente los límites fijos sin romper fronteras."""
    w0 = np.zeros(num_activos)
    capital_restante = 1.0

    # 1. Clavar los límites mínimos y los iguales
    for i in range(num_activos):
        minimo = limites_tupla[i][0]
        w0[i] = minimo
        capital_restante -= minimo

    # 2. Identificar qué activos tienen permitido recibir el capital sobrante
    activos_flexibles = [
        i
        for i in range(num_activos)
        if limites_tupla[i][1] - limites_tupla[i][0] > 1e-6
    ]

    # 3. Repartir el sobrante
    if capital_restante > 0 and activos_flexibles:
        porcion = capital_restante / len(activos_flexibles)
        for i in activos_flexibles:
            w0[i] += porcion

    # Omitimos divisiones extrañas para no corromper la memoria flotante de Python
    return w0


def solicitar_reglas(num_activos):
    """Interfaz interactiva que traduce texto a tensores lógicos."""
    limites = [[0.0, 1.0] for _ in range(num_activos)]
    restricciones = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    while True:
        if input("\n¿Agregar reglas a este escenario? (s/n): ").strip().lower() != "s":
            break

        try:
            indices_str = input("  1. Activos (ej. 1,2,3 o 7): ")
            idx_grupo = [int(x.strip()) for x in indices_str.split(",")]

            if any(i < 0 or i >= num_activos for i in idx_grupo):
                print(f"  [!] ERROR: Fuera de rango (0 a {num_activos - 1}).")
                continue

            tipo = input("  2. Condición ('<', '>', '='): ").strip()
            limite_pct = float(input("  3. Porcentaje límite (ej. 0 o 10): ")) / 100.0
            modo = (
                input("  4. ¿Aplicar a SUMA o CADA UNO? (suma/cada): ").strip().lower()
            )

            if modo == "cada":
                for i in idx_grupo:
                    if tipo in ["<", "<="]:
                        limites[i][1] = limite_pct
                    elif tipo in [">", ">="]:
                        limites[i][0] = EPSILON if limite_pct == 0.0 else limite_pct
                    elif tipo == "=":
                        limites[i][0] = limite_pct
                        limites[i][1] = limite_pct
                print(f"  [OK] LÍMITES ACTUALIZADOS (Individual)")

            else:
                if tipo in ["<", "<="]:
                    restricciones.append(
                        {
                            "type": "ineq",
                            "fun": lambda w, idx=idx_grupo, lim=limite_pct: lim
                            - np.sum(w[idx]),
                        }
                    )
                elif tipo in [">", ">="]:
                    lim_real = EPSILON if limite_pct == 0.0 else limite_pct
                    restricciones.append(
                        {
                            "type": "ineq",
                            "fun": lambda w, idx=idx_grupo, lim=lim_real: np.sum(w[idx])
                            - lim,
                        }
                    )
                elif tipo == "=":
                    restricciones.append(
                        {
                            "type": "eq",
                            "fun": lambda w, idx=idx_grupo, lim=limite_pct: np.sum(
                                w[idx]
                            )
                            - lim,
                        }
                    )
                print(f"  [OK] RESTRICCIÓN AÑADIDA (Conjuntos)")

        except Exception as e:
            print("  [!] Entrada inválida. Intenta de nuevo.")

    return tuple((l[0], l[1]) for l in limites), restricciones


# ==========================================
# MÓDULO 4: EL SIMULADOR PRINCIPAL
# ==========================================
def iniciar_simulador():
    print("\n=== INICIANDO SIMULADOR CUANTITATIVO MODULAR ===")

    lineas_stats = leer_bloque_texto("\n[PASO 1] Pega Nombres y Medias:")
    nombres, retornos = [], []
    try:
        for linea in lineas_stats:
            partes = linea.split()
            if len(partes) >= 2:
                nombres.append(partes[0])
                retornos.append(limpiar_numero(partes[-1].replace("%", "")) / 100.0)
        retornos_array = np.array(retornos)
        num_activos = len(nombres)
        print(f"  [OK] {num_activos} activos cargados.")
    except Exception as e:
        print(f"  [!] Error: {e}")
        return

    lineas_matriz = leer_bloque_texto("\n[PASO 2] Pega la Matriz de Covarianzas:")
    filas_matriz = []
    try:
        for linea in lineas_matriz:
            linea_limpia = (
                linea.replace("E- ", "E-")
                .replace("E+ ", "E+")
                .replace("e- ", "e-")
                .replace("e+ ", "e+")
            )
            filas_matriz.append([limpiar_numero(v) for v in linea_limpia.split()])
        cov_matrix = np.array(filas_matriz)
        if cov_matrix.shape != (num_activos, num_activos):
            raise ValueError("Tamaño incorrecto.")
        print(f"  [OK] Matriz cargada.")
    except Exception as e:
        print(f"  [!] Error en matriz: {e}")
        return

    numero_escenario = 1
    while True:
        print(f"\n" + "=" * 50)
        print(f"   ESCENARIO #{numero_escenario}")
        print("=" * 50)

        print("\n--- ÍNDICE DE ACTIVOS ---")
        for i, n in enumerate(nombres):
            print(f"  [{i}] {n}")

        limites_tupla, restricciones = solicitar_reglas(num_activos)

        print("\n[AUDITORÍA DE LÍMITES MATEMÁTICOS]")
        for i, n in enumerate(nombres):
            print(
                f"  {n}: Min {np.round(limites_tupla[i][0]*100, 2)}% | Max {np.round(limites_tupla[i][1]*100, 2)}%"
            )

        w0 = generar_punto_partida(num_activos, limites_tupla)

        print("\nEjecutando optimización...")
        res_min = minimize(
            min_riesgo,
            w0,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=limites_tupla,
            constraints=restricciones,
            tol=TOLERANCIA,
        )
        res_max = minimize(
            max_rendimiento,
            w0,
            args=(retornos_array,),
            method="SLSQP",
            bounds=limites_tupla,
            constraints=restricciones,
            tol=TOLERANCIA,
        )

        df_resultados = pd.DataFrame(
            {
                "Activo": nombres,
                "MinRisk (W)": np.round(res_min.x, 4),
                "MaxProfit (W)": np.round(res_max.x, 4),
            }
        )

        print(f"\n=== RESULTADOS: ESCENARIO #{numero_escenario} ===")
        print(df_resultados.to_string(index=False))

        try:
            df_resultados.to_clipboard(index=False)
            print("\n[ÉXITO] Copiado al portapapeles.")
        except Exception:
            pass

        if input("\n¿Correr otro escenario? (s/n): ").strip().lower() != "s":
            print("\nCerrando entorno.")
            break
        numero_escenario += 1


if __name__ == "__main__":
    iniciar_simulador()
