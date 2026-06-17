"""
Tests de integracion para KGeoMIP con multiples valores de k sobre N4A.

Cobertura:
    1. GeometricSIA (baseline k=2 BFS)  — phi >= 0
    2. KGeoMIP k=2 (exhaustivo)         — phi == 0.375000 sobre N4A
    3. KGeoMIP k=3 (exhaustivo)         — phi == 1.000000 sobre N4A
    4. KGeoMIP k=4 (exhaustivo)         — phi == 1.875000 sobre N4A
    5. KGeoMIP k=5 (k > n_futuros)     — phi == inf sin excepcion
    6. KGeoMIP k=2 con top_n=1          — phi >= 0 (filtro activo)
    7. Invarianza H-3                   — costo_geometrico identico para dos
                                          particiones distintas del mismo sistema

Uso:
    cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation
    python -m src.tests.test_kgeomip_complete
"""

import sys
import numpy as np
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT.parent))

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.k_geomip import KGeoMIP
from src.constants.base import INFTY_POS


def _cargar_tpm(n: int, variante: str = "A") -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[4]
    return np.genfromtxt(
        repo_root / "data" / "samples" / f"N{n}{variante}.csv",
        delimiter=",",
        dtype=np.float32,
    )


def test_baseline_geomip():
    """GeometricSIA produce phi numerico >= 0 sobre N4A."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    sol = GeometricSIA(gestor).aplicar_estrategia("1111", "1111", "1111", tpm)

    assert sol.perdida != INFTY_POS, "GeometricSIA: no debe retornar inf"
    assert float(sol.perdida) >= 0.0
    print(f"[OK] baseline GeometricSIA  phi={sol.perdida:.6f}")


def test_k2_exhaustivo():
    """KGeoMIP k=2 top_n=None produce phi=0.375000 sobre N4A."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    sol = KGeoMIP(gestor, k=2, top_n=None).aplicar_estrategia(
        "1111", "1111", "1111", tpm
    )

    assert sol.perdida != INFTY_POS
    assert abs(float(sol.perdida) - 0.375) < 1e-4, (
        f"k=2: phi esperado 0.375000, obtenido {sol.perdida:.6f}"
    )
    print(f"[OK] KGeoMIP k=2  phi={sol.perdida:.6f}")


def test_k3_exhaustivo():
    """KGeoMIP k=3 top_n=None produce phi=1.000000 sobre N4A."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    sol = KGeoMIP(gestor, k=3, top_n=None).aplicar_estrategia(
        "1111", "1111", "1111", tpm
    )

    assert sol.perdida != INFTY_POS
    assert abs(float(sol.perdida) - 1.0) < 1e-4, (
        f"k=3: phi esperado 1.000000, obtenido {sol.perdida:.6f}"
    )
    print(f"[OK] KGeoMIP k=3  phi={sol.perdida:.6f}")


def test_k4_exhaustivo():
    """KGeoMIP k=4 top_n=None produce phi=1.875000 sobre N4A."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    sol = KGeoMIP(gestor, k=4, top_n=None).aplicar_estrategia(
        "1111", "1111", "1111", tpm
    )

    assert sol.perdida != INFTY_POS
    assert abs(float(sol.perdida) - 1.875) < 1e-4, (
        f"k=4: phi esperado 1.875000, obtenido {sol.perdida:.6f}"
    )
    print(f"[OK] KGeoMIP k=4  phi={sol.perdida:.6f}")


def test_k_mayor_que_futuros():
    """k > n_futuros retorna Solution(perdida=inf) sin lanzar excepcion."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    sol = KGeoMIP(gestor, k=5, top_n=None).aplicar_estrategia(
        "1111", "1111", "1111", tpm
    )

    assert sol.perdida == INFTY_POS, (
        f"k=5 > n_futuros=4: debe retornar perdida=inf, obtuvo {sol.perdida}"
    )
    print("[OK] KGeoMIP k=5 > n_futuros -> Solution(perdida=inf)")


def test_top_n_activo():
    """KGeoMIP con top_n=1 ejecuta sin error y retorna phi numerico."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    sol = KGeoMIP(gestor, k=2, top_n=1).aplicar_estrategia(
        "1111", "1111", "1111", tpm
    )

    assert sol.perdida != INFTY_POS
    assert float(sol.perdida) >= 0.0
    print(f"[OK] KGeoMIP k=2 top_n=1  phi={sol.perdida:.6f}")


def test_invarianza_h3():
    """
    H-3 documentado: _costo_geometrico es invariante respecto a la particion.

    Verifica que dos particiones distintas del mismo sistema producen el mismo
    costo geometrico, documentando el comportamiento actual (no es un error
    a corregir en esta version sino una limitacion conocida).
    """
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    estrategia = KGeoMIP(gestor, k=2, top_n=None)

    # Preparar subsistema para que tabla_transiciones este disponible
    estrategia.sia_preparar_subsistema("1111", "1111", "1111", tpm)
    estrategia._flat_data = [ncubo.data.ravel() for ncubo in estrategia.sia_subsistema.ncubos]
    dims = estrategia.sia_subsistema.dims_ncubos
    estrategia.estado_inicial = estrategia.sia_subsistema.estado_inicial[dims]
    estrategia.estado_final = 1 - estrategia.estado_inicial
    estrategia.idx_ncubos = list(range(len(estrategia.sia_subsistema.indices_ncubos)))
    estrategia.caminos = {0: [estrategia.estado_inicial.tolist()]}
    estrategia.tabla_transiciones[
        tuple(estrategia.caminos[0][0]), tuple(estrategia.caminos[0][0])
    ] = [0.0] * len(estrategia.sia_subsistema.indices_ncubos)
    for nivel in range(1, len(estrategia.estado_inicial) + 1):
        estrategia.calcular_costos_nivel(estrategia.estado_final, nivel)

    # Dos particiones distintas del mismo sistema N4A (4 futuros: 0,1,2,3)
    particion_a = ((0,), (1, 2, 3))   # singleton + grupo grande
    particion_b = ((0, 1), (2, 3))    # dos grupos iguales

    costo_a = estrategia._costo_geometrico(particion_a)
    costo_b = estrategia._costo_geometrico(particion_b)

    assert abs(costo_a - costo_b) < 1e-9, (
        f"H-3: se esperaba invarianza pero costo_a={costo_a:.6f} != costo_b={costo_b:.6f}"
    )
    print(
        f"[OK] H-3 confirmado: _costo_geometrico invariante "
        f"(costo_a={costo_a:.6f} == costo_b={costo_b:.6f})"
    )


if __name__ == "__main__":
    tests = [
        test_baseline_geomip,
        test_k2_exhaustivo,
        test_k3_exhaustivo,
        test_k4_exhaustivo,
        test_k_mayor_que_futuros,
        test_top_n_activo,
        test_invarianza_h3,
    ]
    errores = []
    for t in tests:
        try:
            t()
        except Exception as e:
            errores.append((t.__name__, e))
            print(f"[FAIL] {t.__name__}: {e}")

    print("\n" + "-" * 60)
    if errores:
        print(f"FALLOS: {len(errores)}/{len(tests)}")
        sys.exit(1)
    else:
        print(f"TODOS LOS TESTS PASARON ({len(tests)}/{len(tests)})")
