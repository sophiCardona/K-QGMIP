"""
Tests de integracion para KQNodes en el proyecto QNodes.

Cobertura:
    1. test_aplanar_vertices     — estructuras simples, listas y anidadas
    2. test_residual             — eliminacion correcta del corte
    3. test_smoke_k2             — k=2 ejecuta sin error, phi >= 0
    4. test_k3                   — k=3 retorna phi numerico
    5. test_k4                   — k=4 retorna phi numerico
    6. test_k_mayor_que_nodos    — k > |V| retorna Solution(perdida=inf)

Uso desde la raiz del proyecto QNodes/:
    cd QNodes
    python -m tests.test_kqnodes

O desde la raiz del repo K-QGMIP:
    cd QNodes
    python -m pytest tests/test_kqnodes.py -v
"""

import sys
from pathlib import Path

import numpy as np

# Asegurar que src/ sea visible
_QNODES_ROOT = Path(__file__).resolve().parents[1]
if str(_QNODES_ROOT) not in sys.path:
    sys.path.insert(0, str(_QNODES_ROOT))

from src.controllers.manager import Manager
from src.models.base.application import aplicacion
from src.strategies.k_qnodes import KQNodes
from src.constants.base import INFTY_POS


def _cargar_tpm(n: int, variante: str = "A") -> np.ndarray:
    gestor = Manager("1" * n)
    return gestor.cargar_red()


def _estrategia(n: int, k: int, variante: str = "A") -> tuple:
    """Devuelve (estrategia, tpm) para un sistema de n nodos."""
    tpm = _cargar_tpm(n, variante)
    return KQNodes(tpm, k=k), tpm


def test_aplanar_vertices():
    """_aplanar_vertices maneja tuplas simples, listas y estructuras anidadas."""
    tpm = _cargar_tpm(2)
    kq = KQNodes(tpm, k=2)

    assert kq._aplanar_vertices((0, 1)) == [(0, 1)]
    assert kq._aplanar_vertices([(0, 1), (1, 2)]) == [(0, 1), (1, 2)]
    anidado = [(0, 0), [(1, 1), (0, 2)]]
    result = kq._aplanar_vertices(anidado)
    assert (0, 0) in result and (1, 1) in result and (0, 2) in result
    print("[OK] test_aplanar_vertices")


def test_residual():
    """_residual elimina correctamente los vertices del corte."""
    tpm = _cargar_tpm(2)
    kq = KQNodes(tpm, k=2)

    V = [(0, 0), (0, 1), (1, 0), (1, 1)]
    corte = ((0, 0),)
    res = kq._residual(V, corte)
    assert (0, 0) not in res
    assert len(res) == 3
    print("[OK] test_residual")


def test_smoke_k2():
    """KQNodes k=2 ejecuta sin error y retorna phi numerico positivo."""
    aplicacion.set_pagina_red_muestra("A")
    tpm = _cargar_tpm(4)
    estrategia = KQNodes(tpm, k=2)
    sol = estrategia.aplicar_estrategia("1000", "1111", "1111", "1111")

    assert sol.perdida != INFTY_POS, "k=2: perdida no deberia ser inf"
    assert float(sol.perdida) >= 0.0, "k=2: phi debe ser >= 0"
    assert sol.particion, "k=2: debe haber representacion de particion"
    print(f"[OK] test_smoke_k2  phi={sol.perdida:.6f}")


def test_k3():
    """KQNodes k=3 retorna Solution con phi numerico sobre N4A."""
    aplicacion.set_pagina_red_muestra("A")
    tpm = _cargar_tpm(4)
    estrategia = KQNodes(tpm, k=3)
    sol = estrategia.aplicar_estrategia("1000", "1111", "1111", "1111")

    assert sol.perdida != INFTY_POS, "k=3: perdida no deberia ser inf"
    assert float(sol.perdida) >= 0.0, "k=3: phi debe ser >= 0"
    print(f"[OK] test_k3  phi={sol.perdida:.6f}  particion={len(sol.particion)} chars")


def test_k4():
    """KQNodes k=4 retorna Solution con phi numerico sobre N4A."""
    aplicacion.set_pagina_red_muestra("A")
    tpm = _cargar_tpm(4)
    estrategia = KQNodes(tpm, k=4)
    sol = estrategia.aplicar_estrategia("1000", "1111", "1111", "1111")

    assert sol.perdida != INFTY_POS, "k=4: perdida no deberia ser inf"
    assert float(sol.perdida) >= 0.0, "k=4: phi debe ser >= 0"
    print(f"[OK] test_k4  phi={sol.perdida:.6f}  particion={len(sol.particion)} chars")


def test_k_mayor_que_nodos():
    """k > |V| debe retornar Solution(perdida=inf) sin lanzar excepcion."""
    aplicacion.set_pagina_red_muestra("A")
    tpm = _cargar_tpm(3)
    estrategia = KQNodes(tpm, k=20)
    sol = estrategia.aplicar_estrategia("100", "111", "111", "111")

    assert sol.perdida == INFTY_POS, "k > |V|: debe retornar perdida=inf"
    print("[OK] test_k_mayor_que_nodos -> Solution(perdida=inf)")


if __name__ == "__main__":
    tests = [
        test_aplanar_vertices,
        test_residual,
        test_smoke_k2,
        test_k3,
        test_k4,
        test_k_mayor_que_nodos,
    ]
    errores = []
    for t in tests:
        try:
            t()
        except Exception as e:
            errores.append((t.__name__, e))
            print(f"[FAIL] {t.__name__}: {e}")

    print("\n" + "-" * 50)
    if errores:
        print(f"FALLOS: {len(errores)}/{len(tests)}")
        sys.exit(1)
    else:
        print(f"TODOS LOS TESTS PASARON ({len(tests)}/{len(tests)})")
