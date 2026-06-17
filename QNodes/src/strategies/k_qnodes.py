"""
KQNodes — Extensión submodular para k-Particiones de Mínima Información.

Extiende QNodes (algoritmo de Queyranne, k=2) al caso general de k
particiones mediante un greedy iterativo de k-1 pasadas sucesivas.

Herencia:
    KQNodes → QNodes → SIA

Reutiliza sin modificar:
    - sia_preparar_subsistema()     [SIA]
    - algorithm(vertices)           [QNodes — Queyranne completo]
    - nodes_complement(nodes)       [QNodes]

Añade / sobreescribe:
    - __init__                      parámetro k
    - aplicar_estrategia            greedy iterativo k-1 cortes
    - funcion_submodular            override con cache memoria_delta inter-iteraciones
    - _reiniciar_memoria            reinicia memoria_grupo_candidato entre pasadas
    - _residual                     calcula residual tras un corte
    - _aplanar_vertices             aplana estructuras anidadas de Queyranne
    - _producto_tensorial_k_partes  construye D_SP por asignacion posicional
    - _formatear_k_particion        representacion legible de la k-MIP

Correcciones:
    H-1: Roles EFECTO/ACTUAL determinados directamente por v[0] == EFFECT/ACTUAL,
         eliminando heuristica dinamica que fallaba en subsistemas simetricos.
    H-2: Residual demasiado pequeño (|V_res| <= 2) se poda con extraccion directa;
         ultima parte residual siempre se agrega al final.
"""

import time
from typing import Union

import numpy as np

from src.strategies.q_nodes import QNodes
from src.middlewares.slogger import SafeLogger
from src.funcs.iit import emd_efecto, ABECEDARY, LOWER_ABECEDARY
from src.models.core.solution import Solution
from src.constants.base import (
    TYPE_TAG,
    INFTY_POS,
    EFFECT,
    ACTUAL,
    VOID_STR,
)
from src.constants.models import (
    DUMMY_ARR,
    ERROR_PARTITION,
    KQNODES_LABEL,
    KQNODES_TAG,
    KQNODES_ANALYSIS_TAG,
)


class KQNodes(QNodes):
    """
    Extension submodular para encontrar la k-MIP.

    Para k=2 es identico a QNodes (Queyranne exacto, optimo garantizado).
    Para k>2 aplica Queyranne k-1 veces sucesivas sobre el sistema residual
    (greedy iterativo, heuristica de alta calidad sin garantia de optimo global).

    Complejidad:
        k=2  -> O(n^3)          optimo garantizado
        k>2  -> O((k-1) * n^3)  greedy heuristico

    Args:
        tpm (np.ndarray): Matriz de probabilidad de transicion shape (2^N, N).
        k (int): Numero de partes >= 2.
    """

    def __init__(self, tpm: np.ndarray, k: int = 2) -> None:
        if k < 2:
            raise ValueError(f"k debe ser >= 2, recibido: {k}")
        super().__init__(tpm)
        self.k = k
        # memoria_delta ya existe en QNodes.__init__ — se reutiliza como cache
        # inter-iteraciones (persiste entre las k-1 pasadas de Queyranne).
        self.logger = SafeLogger(KQNODES_TAG)

    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ) -> Solution:
        """
        Encuentra la k-MIP usando el enfoque submodular greedy.

        Args:
            estado_inicial: estado de bits del sistema completo.
            condicion:      bits en '0' condicionan la variable.
            alcance:        bits en '0' sustraen la variable del futuro.
            mecanismo:      bits en '0' sustraen la variable del presente.

        Returns:
            Solution con la k-MIP encontrada (phi, particion, distribuciones, tiempo).
        """
        # PASO 1: Preparar subsistema
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        # PASO 2: Construir vertices
        futuro = tuple(
            (EFFECT, int(i)) for i in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, int(j)) for j in self.sia_subsistema.dims_ncubos
        )

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size
        self.indices_alcance = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos
        self.tiempos = (
            np.zeros(self.n, dtype=np.int8),
            np.zeros(self.m, dtype=np.int8),
        )

        V = list(presente + futuro)
        self.vertices = set(presente + futuro)

        # PASO 3: Validar k
        if len(V) < self.k:
            self.logger.error(
                f"k={self.k} > n_nodos={len(V)}. No es posible generar esa cantidad de partes."
            )
            return Solution(
                estrategia=KQNODES_LABEL,
                perdida=INFTY_POS,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=np.array(DUMMY_ARR, dtype=np.float32),
                particion=ERROR_PARTITION,
                tiempo_total=0.0,
                quiere_hablar=False,
            )

        # PASO 4: Greedy iterativo — k-1 cortes de Queyranne
        partes: list = []
        V_res: list = list(V)

        for iteracion in range(1, self.k):
            self.logger.critic(
                f"KQNodes k={self.k}: iteracion {iteracion}/{self.k - 1} "
                f"-- {len(V_res)} vertices restantes"
            )

            if len(V_res) <= 2:
                # H-2: residual demasiado pequeno, extraccion directa
                corte = (V_res[0],)
            else:
                self._reiniciar_memoria()
                self.vertices = set(map(tuple, V_res))
                corte = self.algorithm(list(V_res))

            partes.append(corte)
            V_res = self._residual(V_res, corte)

            if not V_res:
                self.logger.critic("Residual vacio antes de completar k partes.")
                break

        # H-2: siempre agregar el residual final como ultima parte
        if V_res:
            partes.append(tuple(V_res))

        self.logger.critic(f"KQNodes: {len(partes)} partes encontradas.")

        # PASO 5: Evaluar phi
        D_SP = self._producto_tensorial_k_partes(partes)
        perdida = emd_efecto(D_SP, self.sia_dists_marginales)
        fmt = self._formatear_k_particion(partes)

        self.logger.critic(f"k-MIP encontrada: phi={perdida:.6f}")

        # PASO 6: Retornar solucion
        return Solution(
            estrategia=KQNODES_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=D_SP,
            particion=fmt,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            quiere_hablar=False,
        )

    def funcion_submodular(
        self,
        deltas: Union[tuple, list],
        omegas: list,
    ):
        """
        Override de QNodes.funcion_submodular.

        El EMD individual del delta se cachea en memoria_delta (que persiste
        entre las k-1 pasadas de Queyranne), evitando recomputar biparticiones
        para deltas que reaparecen en iteraciones posteriores.

        H-1: Roles EFECTO/ACTUAL determinados directamente por v[0] == EFFECT/ACTUAL.

        Returns:
            (emd_union, emd_delta, dist_delta)
        """
        # Extraer indices de delta (H-1: usar EFFECT/ACTUAL directamente)
        planos_delta = self._aplanar_vertices(deltas)
        idxs_efecto_delta = sorted(v[1] for v in planos_delta if v[0] == EFFECT)
        dims_actual_delta = sorted(v[1] for v in planos_delta if v[0] == ACTUAL)

        clave_delta = (tuple(dims_actual_delta), tuple(idxs_efecto_delta))

        # Evaluacion individual con cache inter-iteraciones
        if clave_delta in self.memoria_delta:
            emd_delta, dist_delta = self.memoria_delta[clave_delta]
        else:
            P_delta = self.sia_subsistema.bipartir(
                np.array(idxs_efecto_delta, dtype=np.int8),
                np.array(dims_actual_delta, dtype=np.int8),
            )
            dist_delta = P_delta.distribucion_marginal()
            emd_delta = emd_efecto(dist_delta, self.sia_dists_marginales)
            self.memoria_delta[clave_delta] = (emd_delta, dist_delta)

            if emd_delta == 0.0:
                return INFTY_POS, emd_delta, dist_delta

        # Evaluacion de la union delta union omega
        idxs_efecto_union = list(idxs_efecto_delta)
        dims_actual_union = list(dims_actual_delta)

        for omega in omegas:
            for v in self._aplanar_vertices(omega):
                if v[0] == EFFECT:
                    idxs_efecto_union.append(v[1])
                else:
                    dims_actual_union.append(v[1])

        P_union = self.sia_subsistema.bipartir(
            np.array(idxs_efecto_union, dtype=np.int8),
            np.array(dims_actual_union, dtype=np.int8),
        )
        dist_union = P_union.distribucion_marginal()
        emd_union = emd_efecto(dist_union, self.sia_dists_marginales)

        return emd_union, emd_delta, dist_delta

    def _reiniciar_memoria(self) -> None:
        """
        Reinicia memoria_grupo_candidato entre pasadas del greedy.
        memoria_delta NO se reinicia: es el cache inter-iteraciones.
        """
        self.memoria_grupo_candidato = {}
        self.clave_submodular = [], []

    def _residual(self, vertices_todos: list, corte) -> list:
        """Devuelve vertices_todos menos los vertices del corte."""
        conjunto_corte = set(map(tuple, self._aplanar_vertices(corte)))
        residual = []
        for v in vertices_todos:
            componentes = set(map(tuple, self._aplanar_vertices(v)))
            if not componentes.intersection(conjunto_corte):
                residual.append(v)
        return residual

    def _aplanar_vertices(self, v) -> list:
        """
        Convierte estructuras potencialmente anidadas (resultado de los
        par_candidato generados por Queyranne) a lista plana de tuplas (tiempo, indice).
        """
        # Caso base: tupla (int, int) — vertice simple
        if (
            isinstance(v, tuple)
            and len(v) == 2
            and isinstance(v[0], (int, np.integer))
            and isinstance(v[1], (int, np.integer))
        ):
            return [v]

        # Caso recursivo
        resultado = []
        for elemento in v:
            resultado.extend(self._aplanar_vertices(elemento))
        return resultado

    def _producto_tensorial_k_partes(self, partes: list) -> np.ndarray:
        """
        Construye D_SP de shape (m,) por asignacion posicional (scatter).

        H-1 corregido: roles EFECTO/ACTUAL determinados directamente por
        v[0] == EFFECT o v[0] == ACTUAL.

        Args:
            partes: lista de k grupos de vertices (posiblemente anidados).

        Returns:
            np.ndarray de shape (m,) con la distribucion del sistema partido.
        """
        indices_ncubos = self.sia_subsistema.indices_ncubos
        dims_ncubos = self.sia_subsistema.dims_ncubos

        arr_presentes_todos = dims_ncubos.copy()
        nodo_a_prob: dict[int, float] = {}

        for parte in partes:
            planos = self._aplanar_vertices(parte)

            futuros_reales = [v[1] for v in planos if v[0] == EFFECT]
            presentes_reales = [v[1] for v in planos if v[0] == ACTUAL]

            if not futuros_reales:
                continue

            arr_presentes = (
                np.array(presentes_reales, dtype=np.int8)
                if presentes_reales
                else arr_presentes_todos
            )
            arr_futuros = np.array(futuros_reales, dtype=np.int8)

            parte_obj = self.sia_subsistema.bipartir(arr_futuros, arr_presentes)
            dist_parte = parte_obj.distribucion_marginal()

            for i, nodo_real in enumerate(futuros_reales):
                nodo_a_prob[int(nodo_real)] = float(dist_parte[i])

        D_SP = np.array(
            [nodo_a_prob.get(int(n), 0.0) for n in indices_ncubos],
            dtype=np.float32,
        )
        return D_SP

    def _formatear_k_particion(self, partes: list) -> str:
        """
        Representacion legible de la k-MIP:

            |Futuros_1||Futuros_2||...||Futuros_k|
            |Presentes_1||Presentes_2||...||Presentes_k|
        """
        linea_top = ""
        linea_bot = ""

        for parte in partes:
            planos = self._aplanar_vertices(parte)

            futuros_reales = [v[1] for v in planos if v[0] == EFFECT]
            presentes_reales = [v[1] for v in planos if v[0] == ACTUAL]

            str_fut = (
                ",".join(ABECEDARY[int(f)] for f in futuros_reales)
                if futuros_reales else VOID_STR
            )
            str_pre = (
                ",".join(LOWER_ABECEDARY[int(p)] for p in presentes_reales)
                if presentes_reales else VOID_STR
            )

            ancho = max(len(str_fut), len(str_pre)) + 2
            linea_top += f"|{str_fut:^{ancho}}|"
            linea_bot += f"|{str_pre:^{ancho}}|"

        return f"{linea_top}\n{linea_bot}\n"
