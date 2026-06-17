"""
main.py — Orquestador batch corregido.

CORRECCIONES respecto a la versión modificada:
  1. sheet_name=8  ← restaurado al valor original del proyecto
  2. np.genfromtxt reemplazado por np.loadtxt para archivos grandes
  3. estado_inicio forzable por parámetro para evitar que inferir_estado_inicial
     detecte N20A.csv cuando quieres correr N10 o N15
  4. Para N>=15: top_n reducido automáticamente para evitar colapso de RAM
"""

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.k_geomip import KGeoMIP
from src.controllers.strategies.k_qnodes import KQNodes
from src.controllers.strategies.q_nodes import QNodes

try:
    from src.controllers.strategies.phi import Phi
except Exception:
    Phi = None

import multiprocessing
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path

METHOD2_ROOT = Path(__file__).resolve().parents[1]
GEOMIP_ROOT  = Path(__file__).resolve().parents[3]


def convertir_a_binario(texto, n_bits=20):
    posiciones = "ABCDEFGHIJKLMNOPQRST"[:n_bits]
    binario = ["0"] * n_bits
    for letra in texto:
        if letra in posiciones:
            binario[posiciones.index(letra)] = "1"
    return "".join(binario)


def cargar_tpm(tpm_path: Path) -> np.ndarray:
    """
    Carga la TPM de forma eficiente usando Pandas.
    
    Pandas (engine='c') es mucho más rápido que np.loadtxt para archivos grandes
    y maneja correctamente la conversión de strings "0.00000000" a números.
    """
    n = int(re.search(r'N(\d+)', tpm_path.name).group(1))
    n_estados = 2 ** n
    size_mb = tpm_path.stat().st_size / (1024 ** 2)
    print(f"Cargando TPM: {tpm_path.name}  ({n_estados:,} filas, {size_mb:.0f} MB)...")

    # 1. Leer con Pandas. 
    # - header=None es CRÍTICO para que la primera fila no se vuelva el nombre de las columnas.
    # - dtype=np.float32 permite parsear "0.00000000" sin consumir mucha memoria.
    # - engine='c' fuerza la máxima velocidad de lectura.
    df = pd.read_csv(
        tpm_path, 
        header=None, 
        dtype=np.float32, 
        engine='c'
    )
    
    # 2. Convertir el DataFrame a un array de NumPy estricto en int8
    tpm = df.to_numpy(dtype=np.float32)
    
    print(f"TPM cargada: shape={tpm.shape}, tipo={tpm.dtype}")
    return tpm


def top_n_para_n(n: int, k: int) -> int | None:
    """
    Devuelve el top_n recomendado según el tamaño del sistema.

    Para N pequeño → exhaustivo (None).
    Para N grande → top_n conservador para no colapsar RAM.
    """
    if n <= 6:
        return None        # exhaustivo, S(2n,k) es manejable
    if n <= 8:
        return 100
    if n <= 10:
        return 50
    if n <= 15:
        return 20
    return 10              # N>=16: solo los 10 mejores candidatos geométricos


def ejecutar_con_tiempo(
    config_sistema,
    condiciones: str,
    alcance: str,
    mecanismo: str,
    resultado_queue,
    tpm: np.ndarray,
    k: int = 2,
    top_n: int | None = 20,
    estrategia: str = "kgeomip",
):
    """
    Ejecuta KGeoMIP o KQNodes en un proceso hijo aislado.
    El proceso padre puede matarlo con timeout si tarda demasiado.

    Args:
        estrategia: "kgeomip" (default) o "kqnodes".
    """
    try:
        if estrategia == "kqnodes":
            analizador = KQNodes(config_sistema, k=k)
            resultado = analizador.aplicar_estrategia(
                condiciones, alcance, mecanismo, tpm
            )
        else:
            analizador = KGeoMIP(config_sistema, k=k, top_n=top_n)
            resultado  = analizador.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)

        resultado_queue.put({
            "particion":  resultado.particion,
            "perdida":    str(resultado.perdida).replace('.', ','),
            "tiempo":     str(resultado.tiempo_ejecucion).replace('.', ','),
            "estrategia": estrategia,
        })
    except Exception as e:
        print(f"  [proceso hijo] Error: {e}")
        resultado_queue.put({
            "particion":  None,
            "perdida":    None,
            "tiempo":     None,
            "estrategia": estrategia,
        })


def resolver_tpm_path(estado_inicio: str) -> Path:
    """Busca el CSV de la TPM para el tamaño de estado indicado."""
    sample_name = f"N{len(estado_inicio)}A.csv"
    candidates = (
        METHOD2_ROOT / "src" / ".samples" / sample_name,
        METHOD2_ROOT / ".samples" / sample_name,
        GEOMIP_ROOT  / "data" / "samples" / sample_name,
    )
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No se encontró '{sample_name}'. "
        f"Busqué en: {', '.join(str(c) for c in candidates)}"
    )


def inferir_estado_inicial(max_n: int = 15) -> str:
    """
    Infiere el estado inicial del CSV más grande disponible,
    con un límite max_n para evitar seleccionar N20 accidentalmente.

    CORRECCIÓN: el parámetro max_n evita que si tienes N20A.csv en
    samples, el sistema lo seleccione cuando quieres correr N10 o N15.
    """
    sample_dirs = (
        METHOD2_ROOT / "src" / ".samples",
        METHOD2_ROOT / ".samples",
        GEOMIP_ROOT  / "data" / "samples",
    )
    pattern = re.compile(r"N(\d+)[A-Z]\.csv$")
    available_sizes = []

    for sample_dir in sample_dirs:
        if not sample_dir.exists():
            continue
        for f in sample_dir.glob("N*.csv"):
            m = pattern.match(f.name)
            if m:
                n = int(m.group(1))
                if n <= max_n:          # ← solo considerar hasta max_n
                    available_sizes.append(n)

    if not available_sizes:
        raise FileNotFoundError(
            f"No hay archivos TPM disponibles con N<={max_n}."
        )

    n_bits = max(available_sizes)
    print(f"Estado inicial inferido: N={n_bits}")
    return "1" + ("0" * (n_bits - 1))


def ejecutar_desde_excel(
    ruta_excel: Path,
    ruta_salida: Path,
    inicio: int = 0,
    cantidad: int = 50,
    estado_inicio: str | None = None,
    condiciones: str | None = None,
    k: int = 2,
    timeout_seg: int = 3600,
    max_n: int = 15,          # limita qué N se selecciona automáticamente
    estrategia: str = "kgeomip",
):
    df = pd.read_excel(
        ruta_excel,
        sheet_name=0,          
        usecols="B",
        skiprows=3,
        names=["Subsistema"],
    )
    filas = df["Subsistema"].dropna().tolist()
    filas = filas[inicio : inicio + cantidad]

    # ── Estado inicial y TPM ──────────────────────────────────────────────
    estado_inicio = estado_inicio or inferir_estado_inicial(max_n=max_n)
    condiciones   = condiciones   or ("1" * len(estado_inicio))
    n             = len(estado_inicio)
    tpm_path      = resolver_tpm_path(estado_inicio)

    # usar loadtxt en vez de genfromtxt ──────────────────
    tpm = cargar_tpm(tpm_path)

    top_n = top_n_para_n(n, k)
    print(f"Configuración: N={n}, k={k}, top_n={top_n}, filas={len(filas)}")

    resultados = []

    for i, fila in enumerate(filas, start=inicio + 1):
        partes = fila.split("|")
        if len(partes) != 2:
            continue

        alcance   = convertir_a_binario(partes[0][: len(partes[0]) - 3], n_bits=n)
        mecanismo = convertir_a_binario(partes[1][: len(partes[1]) - 1], n_bits=n)
        print(f"Iteración {i} — Alcance: {alcance}  Mecanismo: {mecanismo}")

        config_sistema = Manager(estado_inicial=estado_inicio)
        resultado_queue = multiprocessing.Queue()

        proceso = multiprocessing.Process(
            target=ejecutar_con_tiempo,
            args=(
                config_sistema,
                condiciones,
                alcance,
                mecanismo,
                resultado_queue,
                tpm,
                k,
                top_n,
                estrategia,
            ),
        )
        proceso.start()
        proceso.join(timeout=timeout_seg)

        if proceso.is_alive():
            print(f"  Iteración {i}: timeout ({timeout_seg}s). Terminando proceso...")
            proceso.terminate()
            proceso.join()
            resultado = {"perdida": None, "tiempo": None, "particion": None}
        else:
            resultado = (
                resultado_queue.get()
                if not resultado_queue.empty()
                else {"perdida": None, "tiempo": None, "particion": None}
            )

        resultados.append({
            "Iteración":              i,
            "Alcance":                alcance,
            "Mecanismo":              mecanismo,
            "k":                      k,
            "top_n":                  top_n,
            "Partición":              resultado["particion"],
            "Pérdida":                resultado["perdida"],
            "Tiempo de ejecución (s)": resultado["tiempo"],
        })

    df_resultados = pd.DataFrame(resultados)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"Resultados guardados en {ruta_salida}")


def iniciar():
    ruta_entrada = Path(
        os.getenv(
            "GEOMIP_INPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "DatosPruebas2026_1 (1).xlsx"),
        )
    )
    ruta_salida = Path(
        os.getenv(
            "GEOMIP_OUTPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "resultados_KGeoMIP.xlsx"),
        )
    )

    # ── Controlar N, k y estrategia desde variables de entorno ───────────
    # En Windows CMD:
    #   set KGEOMIP_K=3 && set KGEOMIP_MAX_N=15 && set KGEOMIP_ESTRATEGIA=kqnodes && python exec.py
    # En Linux:
    #   KGEOMIP_K=3 KGEOMIP_MAX_N=15 KGEOMIP_ESTRATEGIA=kqnodes python3 exec.py
    k          = int(os.getenv("KGEOMIP_K",          "2"))
    max_n      = int(os.getenv("KGEOMIP_MAX_N",      "15"))
    estrategia = str(os.getenv("KGEOMIP_ESTRATEGIA", "kgeomip")).lower()

    ejecutar_desde_excel(
        ruta_entrada,
        ruta_salida,
        k=k,
        max_n=max_n,
        estrategia=estrategia,
    )