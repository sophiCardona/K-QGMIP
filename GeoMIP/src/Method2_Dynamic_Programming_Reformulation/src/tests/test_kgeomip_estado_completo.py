"""
Tests de integracion para KGeoMIP con subsistemas parciales (condicionados).

Complementa test_kgeomip_complete.py verificando que KGeoMIP funciona
correctamente cuando alcance y mecanismo NO son el sistema completo,
usando condicionamiento y substraccion de variables.

Cobertura:
    1. Subsistema parcial (alcance="1110", mecanismo="1110") — k=2
    2. Subsistema parcial — k=3
    3. Subsistema parcial — k=4 (k == n_futuros: S(3,3)=1, trivial)
    4. Subsistema simetrico (|alcance|==|mecanismo|) — k=2, verifica H-1
    5. Particion formateada contiene lineas de futuros y presentes

Uso:
    cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation
    python -m src.tests.test_kgeomip_estado_completo
"""

import sys
import numpy as np
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT.parent))

from src.controllers.manager import Manager
from src.controllers.strategies.k_geomip import KGeoMIP
from src.constants.base import INFTY_POS


def _cargar_tpm(n: int, variante: str = "A") -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[4]
    return np.genfromtxt(
        repo_root / "data" / "samples" / f"N{n}{variante}.csv",
        delimiter=",",
        dtype=np.float32,
    )


def test_subsistema_parcial_k2():
    """KGeoMIP k=2 sobre subsistema parcial (3 futuros, 3 presentes) retorna phi valido."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    # alcance="1110" -> 3 futuros activos; mecanismo="1110" -> 3 presentes activos
    sol = KGeoMIP(gestor, k=2, top_n=None).aplicar_estrategia(
        "1111", "1110", "1110", tpm
    )

    assert sol.perdida != INFTY_POS, "subsistema parcial k=2: no debe retornar inf"
    assert float(sol.perdida) >= 0.0
    assert sol.particion and len(sol.particion) > 0
    print(f"[OK] subsistema parcial k=2  phi={sol.perdida:.6f}")


def test_subsistema_parcial_k3():
    """KGeoMIP k=3 sobre subsistema parcial con 3 futuros: S(3,3)=1 particion trivial."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    sol = KGeoMIP(gestor, k=3, top_n=None).aplicar_estrategia(
        "1111", "1110", "1110", tpm
    )

    assert sol.perdida != INFTY_POS, "subsistema parcial k=3: no debe retornar inf"
    assert float(sol.perdida) >= 0.0
    # Con S(3,3)=1, solo existe una particion (singletons): trivialmente valida
    print(f"[OK] subsistema parcial k=3 (trivial S(3,3)=1)  phi={sol.perdida:.6f}")


def test_k_igual_n_futuros():
    """k == n_futuros es el caso limite: una particion de singletons, siempre valido."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    # 4 futuros activos, k=4 -> S(4,4)=1 (todos singletons)
    sol = KGeoMIP(gestor, k=4, top_n=None).aplicar_estrategia(
        "1111", "1111", "1111", tpm
    )

    assert sol.perdida != INFTY_POS, "k==n_futuros: debe encontrar la unica particion"
    assert float(sol.perdida) >= 0.0
    print(f"[OK] k=n_futuros=4 (S(4,4)=1)  phi={sol.perdida:.6f}")


def test_subsistema_simetrico_k2():
    """
    Subsistema simetrico (|alcance|==|mecanismo|) con k=2.

    Verifica que la ausencia del bug H-1 en KGeoMIP (que no usa
    deteccion dinamica de roles) produce phi valido incluso cuando
    el subsistema tiene el mismo numero de futuros y presentes.
    """
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    # Sistema completo 4x4: |alcance|=4 == |mecanismo|=4
    sol = KGeoMIP(gestor, k=2, top_n=None).aplicar_estrategia(
        "1111", "1111", "1111", tpm
    )

    assert sol.perdida != INFTY_POS
    assert abs(float(sol.perdida) - 0.375) < 1e-4, (
        f"simetrico k=2: phi esperado 0.375, obtenido {sol.perdida:.6f}"
    )
    print(f"[OK] subsistema simetrico k=2  phi={sol.perdida:.6f} (H-1 no aplica a KGeoMIP)")


def test_formato_particion():
    """La particion formateada tiene exactamente dos lineas no vacias."""
    gestor = Manager(estado_inicial="1000")
    tpm = _cargar_tpm(4)
    sol = KGeoMIP(gestor, k=3, top_n=None).aplicar_estrategia(
        "1111", "1111", "1111", tpm
    )

    assert sol.particion, "la particion no debe ser vacia"
    lineas = [l for l in sol.particion.split("\n") if l.strip()]
    assert len(lineas) == 2, (
        f"la particion debe tener 2 lineas (futuros/presentes), obtuvo {len(lineas)}: {lineas}"
    )
    # Primera linea: futuros (mayusculas); segunda linea: presentes (minusculas)
    assert "|" in lineas[0] and "|" in lineas[1]
    print(f"[OK] formato particion k=3: 2 lineas correctas")
    print(f"     {lineas[0]}")
    print(f"     {lineas[1]}")


if __name__ == "__main__":
    tests = [
        test_subsistema_parcial_k2,
        test_subsistema_parcial_k3,
        test_k_igual_n_futuros,
        test_subsistema_simetrico_k2,
        test_formato_particion,
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
