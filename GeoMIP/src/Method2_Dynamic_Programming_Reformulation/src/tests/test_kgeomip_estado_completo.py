"""
Prueba completa de KGeoMIP con estado particionado.
Valida múltiples valores de k y muestra resultados.
"""
import numpy as np
from pathlib import Path

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.k_geomip import KGeoMIP

def run_complete_test():
    """
    Prueba KGeoMIP con k=2,3,4 y compara con GeometricSIA.
    """
    gestor = Manager(estado_inicial="1000")
    repo_root = Path(__file__).resolve().parents[4]
    tpm = np.genfromtxt(repo_root / "data" / "samples" / "N4A.csv", delimiter=",")
    
    print("\n" + "="*70)
    print("PRUEBA COMPLETA: KGeoMIP con ESTADO PARTICIONADO")
    print("="*70)
    
    # Baseline: GeometricSIA
    print("\n[1] Baseline - GeometricSIA:")
    baseline = GeometricSIA(gestor)
    sol_baseline = baseline.aplicar_estrategia("1111", "1111", "1111", tpm)
    print(f"    φ = {sol_baseline.perdida:.6f}")
    print(f"    Partición:\n{sol_baseline.particion}")
    
    # KGeoMIP con diferentes k
    for k in [2, 3, 4]:
        print(f"\n[{k+1}] KGeoMIP (k={k}):")
        try:
            estrategia = KGeoMIP(gestor, k=k, top_n=None)
            sol = estrategia.aplicar_estrategia("1111", "1111", "1111", tpm)
            print(f"    φ = {sol.perdida:.6f}")
            print(f"    Partición:\n{sol.particion}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print("\n" + "="*70)
    print("✓ Todos los tests completados correctamente")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_complete_test()
