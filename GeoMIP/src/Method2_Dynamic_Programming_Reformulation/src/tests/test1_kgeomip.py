from src.controllers.manager import Manager
from src.controllers.strategies.KGeoMIP import KGeoMIP
from src.controllers.strategies.geometric import GeometricSIA
from numpy import np

def run_smoke_test():
    # 1. Carga de un dataset estándar de 4 nodos (ej. N4A.csv) [3, 4]
    gestor = Manager(tpm_path="data/samples/N4A.csv", estado_inicial=np.array([0, 1, 2, 3]))
    
    # 2. Ejecución con la base original (k=2 por defecto)
    base_geo = GeometricSIA(gestor)
    solucion_original = base_geo.aplicar_estrategia()
    
    # 3. Ejecución con tu nueva clase KGeoMIP configurada para k=2
    # El límite_evaluacion se hereda para asegurar que el atajo sea el mismo [5]
    nueva_estrategia = KGeoMIP(gestor, k=2, limite_evaluacion=10)
    solucion_k = nueva_estrategia.aplicar_estrategia()
    
    # 4. Verificación de Correctitud [6]
    print(f"Original Phi: {solucion_original.phi} | K-Partita Phi: {solucion_k.phi}")
    assert solucion_original.phi == solucion_k.phi, "ERROR: Los valores de Phi no coinciden"
    assert solucion_original.partition == solucion_k.partition, "ERROR: Las particiones no coinciden"
    
    print("¡Prueba de humo exitosa! KGeoMIP es consistente con el caso base.")

if __name__ == "__main__":
    run_smoke_test()