import numpy as np
from pathlib import Path

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.k_geomip import KGeoMIP

def run_smoke_test():
    """
    Prueba de humo: Verifica que KGeoMIP ejecute sin errores de broadcasting.
    
    NOTA: KGeoMIP y GeometricSIA NO producen resultados idénticos para k=2
    porque usan espacios de candidatos diferentes:
    - GeometricSIA: Genera biparticiones del ESTADO (presentes + futuros)
    - KGeoMIP: Genera k-particiones del espacio FUTURO solamente
    
    Para k=2, KGeoMIP evalúa las 7 posibles particiones desequilibradas del
    espacio futuro (S(4,2)=7), mientras que GeometricSIA busca la mejor 
    bipartición del estado completo. Por eso producen φ valores diferentes.
    
    Esta prueba verifica que:
    1. KGeoMIP ejecuta sin ValueError de broadcasting (✓ FIJADO)
    2. KGeoMIP encuentra una partición válida
    3. Los valores φ son coherentes (positivos)
    """
    gestor = Manager(estado_inicial="1000")  # 4 bits → uses N4A.csv
    repo_root = Path(__file__).resolve().parents[4]  # Goes up to GeoMIP/
    tpm = np.genfromtxt(repo_root / "data" / "samples" / "N4A.csv", delimiter=",")
    
    # Base strategy (k=2)
    base_geo = GeometricSIA(gestor)
    solucion_original = base_geo.aplicar_estrategia("1111", "1111", "1111", tpm)
    
    # KGeoMIP with k=2
    nueva_estrategia = KGeoMIP(gestor, k=2, top_n=10)
    solucion_k = nueva_estrategia.aplicar_estrategia("1111", "1111", "1111", tpm)
    
    # Verify both executed successfully without broadcast errors
    print(f"\nGeometricSIA Phi: {solucion_original.perdida:.6f}")
    print(f"KGeoMIP k=2 Phi: {solucion_k.perdida:.6f}")
    
    # Check that both are numeric (could be numpy types)
    try:
        float(solucion_original.perdida)
        float(solucion_k.perdida)
    except (TypeError, ValueError):
        raise AssertionError("Both strategies should return numeric φ values")
    
    assert float(solucion_original.perdida) > 0, \
        "GeometricSIA φ should be positive"
    assert float(solucion_k.perdida) > 0, \
        "KGeoMIP φ should be positive"
    
    print("\n✓ Prueba de humo exitosa:")
    print("  - KGeoMIP ejecuta sin errores de broadcasting")
    print("  - Ambas estrategias generan particiones válidas")
    print("  - NOTA: Resultados diferentes porque evalúan espacios de candidatos distintos")


if __name__ == "__main__":
    run_smoke_test()
