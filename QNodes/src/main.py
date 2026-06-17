import multiprocessing
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.controllers.manager import Manager
from src.strategies.force import BruteForce
from src.strategies.q_nodes import QNodes
from src.strategies.k_qnodes import KQNodes


QNODES_ROOT = Path(__file__).resolve().parents[1]


def convertir_a_binario(texto: str, n_bits: int = 10) -> str:
    posiciones = "ABCDEFGHIJKLMNOPQRST"[:n_bits]
    binario = ["0"] * n_bits
    for letra in texto:
        if letra in posiciones:
            binario[posiciones.index(letra)] = "1"
    return "".join(binario)


def ejecutar_con_tiempo(
    tpm: np.ndarray,
    estado_inicial: str,
    condicion: str,
    alcance: str,
    mecanismo: str,
    resultado_queue,
    k: int = 2,
    estrategia: str = "kqnodes",
):
    """Ejecuta KQNodes (o QNodes) en un proceso hijo aislado."""
    try:
        if estrategia == "qnodes":
            analizador = QNodes(tpm)
        else:
            analizador = KQNodes(tpm, k=k)

        resultado = analizador.aplicar_estrategia(
            estado_inicial, condicion, alcance, mecanismo
        )
        resultado_queue.put({
            "particion":  resultado.particion,
            "perdida":    str(resultado.perdida).replace(".", ","),
            "tiempo":     str(resultado.tiempo_ejecucion).replace(".", ","),
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


def ejecutar_desde_excel(
    ruta_excel: Path,
    ruta_salida: Path,
    sheet_name: str = "10A-Elementos",
    inicio: int = 0,
    cantidad: int = 10,
    estado_inicio: str | None = None,
    condiciones: str | None = None,
    k: int = 2,
    timeout_seg: int = 3600,
    estrategia: str = "kqnodes",
):
    """
    Lee la hoja `sheet_name` del Excel de pruebas.

    Estructura de la hoja:
      Fila 0: "Estado inicial" | "<bits estado>" | ...
      Fila 3: "#Prueba" | "Alcance o Purview (t+1)" | "Mecanismo(t)" | ...
      Fila 4+: numero | letras alcance (ej. ABCDEFGHIJ) | letras mecanismo | ...
    """
    # Leer estado inicial desde la celda B0 de la hoja
    hdr = pd.read_excel(ruta_excel, sheet_name=sheet_name, nrows=1, header=None)
    estado_bits = str(hdr.iloc[0, 1]).strip()   # "1000000000" para N=10
    n_bits = len(estado_bits)

    if estado_inicio is None:
        estado_inicio = estado_bits
    else:
        n_bits = len(estado_inicio)

    condiciones = condiciones or ("1" * n_bits)

    # Leer filas de datos (skiprows=4 salta las 4 filas de cabecera)
    df = pd.read_excel(
        ruta_excel,
        sheet_name=sheet_name,
        skiprows=4,
        header=None,
        usecols=[0, 1, 2],
        names=["Prueba", "Alcance", "Mecanismo"],
    )
    df = df.dropna(subset=["Alcance", "Mecanismo"])
    filas = df[["Alcance", "Mecanismo"]].values.tolist()
    filas = filas[inicio: inicio + cantidad]

    gestor = Manager(estado_inicio)
    print(f"Cargando TPM desde: {gestor.tpm_filename}")
    tpm = gestor.cargar_red()
    print(f"TPM cargada: shape={tpm.shape}")
    print(f"Configuracion: N={n_bits}, k={k}, filas={len(filas)}, estrategia={estrategia}")

    resultados = []

    for i, (alc_raw, mec_raw) in enumerate(filas, start=inicio + 1):
        alcance = convertir_a_binario(
            re.sub(r"[^A-Z]", "", str(alc_raw).upper()), n_bits=n_bits
        )
        mecanismo = convertir_a_binario(
            re.sub(r"[^A-Z]", "", str(mec_raw).upper()), n_bits=n_bits
        )
        print(f"Iteracion {i} -- Alcance: {alcance}  Mecanismo: {mecanismo}")

        resultado_queue = multiprocessing.Queue()
        proceso = multiprocessing.Process(
            target=ejecutar_con_tiempo,
            args=(
                tpm,
                estado_inicio,
                condiciones,
                alcance,
                mecanismo,
                resultado_queue,
                k,
                estrategia,
            ),
        )
        proceso.start()
        proceso.join(timeout=timeout_seg)

        if proceso.is_alive():
            print(f"  Iteracion {i}: timeout ({timeout_seg}s). Terminando proceso...")
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
            "Iteracion":               i,
            "Alcance":                 alcance,
            "Mecanismo":               mecanismo,
            "k":                       k,
            "Particion":               resultado["particion"],
            "Perdida":                 resultado["perdida"],
            "Tiempo de ejecucion (s)": resultado["tiempo"],
        })

    df_resultados = pd.DataFrame(resultados)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"Resultados guardados en {ruta_salida}")


def iniciar():
    """
    Punto de entrada principal.

    Variables de entorno:
        KQNODES_INPUT_XLSX  — ruta al Excel de pruebas (default: tests/DatosPruebas2026_1 (2).xlsx)
        KQNODES_OUTPUT_XLSX — ruta del Excel de salida  (default: results/resultados_KQNodes.xlsx)
        KQNODES_K           — numero de particiones k   (default: 2)
        KQNODES_CANTIDAD    — filas a procesar           (default: 10)
        KQNODES_ESTRATEGIA  — "kqnodes" o "qnodes"      (default: kqnodes)
    """
    ruta_entrada = Path(
        os.getenv(
            "KQNODES_INPUT_XLSX",
            str(QNODES_ROOT / "tests" / "DatosPruebas2026_1 (2).xlsx"),
        )
    )
    ruta_salida = Path(
        os.getenv(
            "KQNODES_OUTPUT_XLSX",
            str(QNODES_ROOT / "results" / "resultados_KQNodes.xlsx"),
        )
    )

    k = int(os.getenv("KQNODES_K", "2"))
    cantidad = int(os.getenv("KQNODES_CANTIDAD", "10"))
    estrategia = str(os.getenv("KQNODES_ESTRATEGIA", "kqnodes")).lower()
    sheet = str(os.getenv("KQNODES_SHEET", "10A-Elementos"))

    ejecutar_desde_excel(
        ruta_entrada,
        ruta_salida,
        sheet_name=sheet,
        cantidad=cantidad,
        k=k,
        estrategia=estrategia,
    )
