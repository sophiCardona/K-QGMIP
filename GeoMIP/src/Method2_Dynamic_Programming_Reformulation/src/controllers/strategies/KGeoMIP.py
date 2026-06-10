import numpy as np
from src.models.base.sia import SIA
from src.controllers.strategies.geometric import GeometricSIA
from src.funcs.base import emd_efecto
from src.models.core.solution import Solution

class KGeoMIP(SIA):
    def __init__(self, gestor, k: int, limite_evaluacion: int = 10):
        super().__init__(gestor)
        self.k = k
        self.limite_evaluacion = limite_evaluacion
        self.n_nodes = len(gestor.estado_inicial)
        
        # REUTILIZACIÓN: Instanciamos GeoMIP para obtener su tabla de costos
        self.geo_base = GeometricSIA(gestor)
        self.tabla_costos = self.geo_base.tabla 

    def aplicar_estrategia(self) -> Solution:
        """
        Punto de entrada principal que orquestra el filtrado geométrico
        y la búsqueda de la k-MIP.
        """
        nodos = list(range(self.n_nodes))
        
        # 1. Generar Candidatas (Stirling)
        candidatas = list(self._generar_k_particiones(nodos, self.k))
        
        # 2. Filtrado Geométrico (El Atajo)
        puntuaciones = []
        for p in candidatas:
            costo = self._calcular_costo_geometrico(p)
            puntuaciones.append((costo, p))
        
        # Ordenar por menor costo (mejores 'grietas' en el hipercubo)
        puntuaciones.sort(key=lambda x: x)
        top_candidatas = puntuaciones[:self.limite_evaluacion]
        
        # 3. Evaluación Exacta (EMD)
        mejor_phi = float('inf')
        mejor_particion = None
        
        for _, particion in top_candidatas:
            phi = self._evaluar_phi_k(particion)
            if phi < mejor_phi:
                mejor_phi = phi
                mejor_particion = particion
        
        return Solution(mejor_particion, mejor_phi)

    def _evaluar_phi_k(self, particion):
        """Reconstrucción por producto tensorial de k términos."""
        distribucion_original = self.gestor.tpm
        
        # Marginalizamos cada parte S_i usando la infraestructura de N-Cubo
        marginales = [self.gestor.sistema.obtener_marginal(S) for S in particion]
        
        # Producto tensorial iterativo: P(S1) ⊗ P(S2) ⊗ ... ⊗ P(Sk)
        reconstruccion = marginales
        for i in range(1, self.k):
            reconstruccion = np.kron(reconstruccion, marginales[i])
            
        return emd_efecto(distribucion_original, reconstruccion)

    def _calcular_costo_geometrico(self, particion):
        """Suma de inercia t(i, j) para transiciones que rompen la partición."""
        costo_total = 0
        for (estado_i, estado_j), costo in self.tabla_costos.items():
            if self._es_transicion_particionada(estado_i, estado_j, particion):
                costo_total += costo
        return costo_total

    def _es_transicion_particionada(self, i, j, particion):
        diferencia = i ^ j
        nodos_cambiados = [idx for idx in range(self.n_nodes) if (diferencia >> idx) & 1]
        
        partes_afectadas = set()
        for nodo in nodos_cambiados:
            for idx, subconjunto in enumerate(particion):
                if nodo in subconjunto:
                    partes_afectadas.add(idx)
        return len(partes_afectadas) > 1

    def _generar_k_particiones(self, nodos, k):
        # Implementación recursiva de Stirling para k-subconjuntos
        if k == 1: yield [nodos]; return
        if len(nodos) == k: yield [[n] for n in nodos]; return
        
        primer = nodos
        resto = nodos[1:]
        for p in self._generar_k_particiones(resto, k - 1):
            yield [[primer]] + p
        for p in self._generar_k_particiones(resto, k):
            for i in range(k):
                res = [list(g) for g in p]
                res[i].append(primer)
                yield res