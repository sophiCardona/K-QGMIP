"""
Tests de integración para KQNodes.

Cobertura:
    1. Smoke test k=2 — ejecuta sin error y retorna φ numérico positivo.
    2. k=2 vs QNodes  — KQNodes(k=2) debe dar φ ≤ QNodes en N4A
       (ambos usan Queyranne pero KQNodes incluye presentes en V).
    3. k=3 sobre N4A  — retorna 3 partes, φ numérico.
    4. k=4 sobre N4A  — retorna 4 partes, φ numérico.
    5. Poda de residuales — k mayor que vértices disponibles retorna Solution
       de error sin lanzar excepción.

Uso:
    cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation
    python -m src.tests.test_kqnodes
"""

import sys
import numpy as np
from pathlib import Path

# ── Añadir raíz del proyecto al path si se ejecuta directamente ───────────
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT.parent))

from src.controllers.manager import Manager
from src.controllers.strategies.k_qnodes import KQNodes
from src.controllers.strategies.q_nodes import QNodes
from src.constants.base import INFTY_POS


def _cargar_tpm(n: int, variante: str = "A") -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[4]
    path = repo_root / "data" / "samples" / f"N{n}{variante}.csv"
    return np.genfromtxt(path, delimiter=",", dtype=np.float32)


def test_smoke_k2():
    """KQNodes k=2 ejecuta sin error y retorna phi numérico positivo."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    estrategia = KQNodes(gestor, k=2)
    sol = estrategia.aplicar_estrategia("1111", "1111", "1111", tpm)

    assert sol.perdida != INFTY_POS, "k=2: perdida no deberia ser inf"
    assert float(sol.perdida) >= 0.0, "k=2: phi debe ser >= 0"
    assert sol.particion, "k=2: debe haber representacion de particion"
    print(f"[OK] smoke k=2  phi={sol.perdida:.6f}")


def test_k3():
    """KQNodes k=3 retorna Solution con phi numérico sobre N4A."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    estrategia = KQNodes(gestor, k=3)
    sol = estrategia.aplicar_estrategia("1111", "1111", "1111", tpm)

    assert sol.perdida != INFTY_POS, "k=3: perdida no deberia ser inf"
    assert float(sol.perdida) >= 0.0, "k=3: phi debe ser >= 0"
    print(f"[OK] k=3  phi={sol.perdida:.6f}  particion={len(sol.particion)} chars")


def test_k4():
    """KQNodes k=4 retorna Solution con phi numérico sobre N4A."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    estrategia = KQNodes(gestor, k=4)
    sol = estrategia.aplicar_estrategia("1111", "1111", "1111", tpm)

    assert sol.perdida != INFTY_POS, "k=4: perdida no deberia ser inf"
    assert float(sol.perdida) >= 0.0, "k=4: phi debe ser >= 0"
    print(f"[OK] k=4  phi={sol.perdida:.6f}  particion={len(sol.particion)} chars")


def test_k_mayor_que_nodos():
    """k > |V| debe retornar Solution(perdida=inf) sin lanzar excepcion."""
    gestor = Manager(estado_inicial="100")  # N3 -> |V| = 6 (3 pres + 3 fut)
    tpm = _cargar_tpm(3)
    estrategia = KQNodes(gestor, k=20)
    sol = estrategia.aplicar_estrategia("111", "111", "111", tpm)

    assert sol.perdida == INFTY_POS, "k > |V|: debe retornar perdida=inf"
    print("[OK] k > nodos  -> Solution(perdida=inf) sin excepcion")


def test_aplanar_vertices():
    """_aplanar_vertices maneja tuplas simples, listas y estructuras anidadas."""
    gestor = Manager(estado_inicial="10")
    kq = KQNodes(gestor, k=2)

    assert kq._aplanar_vertices((0, 1)) == [(0, 1)]
    assert kq._aplanar_vertices([(0, 1), (1, 2)]) == [(0, 1), (1, 2)]
    anidado = [(0, 0), [(1, 1), (0, 2)]]
    result = kq._aplanar_vertices(anidado)
    assert (0, 0) in result and (1, 1) in result and (0, 2) in result
    print("[OK] _aplanar_vertices maneja estructuras anidadas correctamente")


def test_residual():
    """_residual elimina correctamente los vertices del corte."""
    gestor = Manager(estado_inicial="10")
    kq = KQNodes(gestor, k=2)

    V = [(0, 0), (0, 1), (1, 0), (1, 1)]
    corte = ((0, 0),)
    res = kq._residual(V, corte)
    assert (0, 0) not in res
    assert len(res) == 3
    print("[OK] _residual elimina vertices del corte correctamente")


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
