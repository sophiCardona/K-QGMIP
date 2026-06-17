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
    - __init__                      parámetro k y memoria_delta
    - aplicar_estrategia            greedy iterativo k-1 cortes
    - funcion_submodular            override con caché memoria_delta
    - _reiniciar_memoria            reinicia solo memoria_particiones
    - _residual                     calcula residual tras un corte
    - _aplanar_vertices             aplana estructuras anidadas de Queyranne
    - _producto_tensorial_k_partes  construye D_SP por asignación posicional
    - _formatear_k_particion        representación legible de la k-MIP

Correcciones respecto a versiones anteriores:
    H-1: Bug de inversión de roles causales en _producto_tensorial_k_partes.
         Se usa EFECTO/ACTUAL directamente en lugar de heurística dinámica.
    H-2: Bug de degeneración del número de partes (residual vacío antes de k
         partes). Se añade logging y la última parte residual siempre se agrega.
"""

import time
from typing import Union

import numpy as np

from src.controllers.manager import Manager
from src.controllers.strategies.q_nodes import QNodes
from src.funcs.base import emd_efecto, ABECEDARY, LOWER_ABECEDARY
from src.middlewares.profile import profile, profiler_manager
from src.middlewares.slogger import SafeLogger
from src.models.core.solution import Solution
from src.constants.base import (
    TYPE_TAG,
    NET_LABEL,
    INFTY_POS,
    EFECTO,
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
    Extensión submodular para encontrar la k-MIP.

    Para k=2 es idéntico a QNodes (Queyranne exacto, óptimo garantizado).
    Para k>2 aplica Queyranne k-1 veces sucesivas sobre el sistema residual
    (greedy iterativo, heurística de alta calidad sin garantía de óptimo global).

    Complejidad:
        k=2  → O(n³)          óptimo garantizado
        k>2  → O((k-1) × n³)  greedy heurístico

    Args:
        gestor (Manager): Gestor de datos del proyecto base.
        k (int): Número de partes ≥ 2.
    """

    def __init__(self, gestor: Manager, k: int = 2) -> None:
        if k < 2:
            raise ValueError(f"k debe ser >= 2, recibido: {k}")
        super().__init__(gestor)
        self.k = k
        # Caché inter-iteraciones: persiste entre las k-1 pasadas de Queyranne.
        # Clave: (tuple(dims_ACTUAL), tuple(idxs_EFECTO))
        # Valor: (φ_δ: float, dist_δ: np.ndarray)
        self.memoria_delta: dict = {}
        self.logger = SafeLogger(KQNODES_TAG)

    @profile(context={TYPE_TAG: KQNODES_ANALYSIS_TAG})
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
            estado_inicial: cadena de bits de longitud N.
            condicion:  bits en '0' condicionan la variable.
            alcance:    bits en '0' sustraen la variable del futuro.
            mecanismo:  bits en '0' sustraen la variable del presente.

        Returns:
            Solution con la k-MIP encontrada (φ, partición, distribuciones, tiempo).
        """
        # ── PASO 1: Preparar subsistema ────────────────────────────────────
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)

        # ── PASO 2: Construir vértices ─────────────────────────────────────
        F = [(EFECTO, int(i)) for i in self.sia_subsistema.indices_ncubos]
        P = [(ACTUAL,  int(j)) for j in self.sia_subsistema.dims_ncubos]
        V = P + F

        self.m = self.sia_subsistema.indices_ncubos.size
        self.n = self.sia_subsistema.dims_ncubos.size
        self.indices_alcance   = self.sia_subsistema.indices_ncubos
        self.indices_mecanismo = self.sia_subsistema.dims_ncubos

        import numpy as _np
        self.tiempos = (
            _np.zeros(self.n, dtype=_np.int8),
            _np.zeros(self.m, dtype=_np.int8),
        )

        # ── PASO 3: Validar k ──────────────────────────────────────────────
        n_nodos = len(V)
        if n_nodos < self.k:
            self.logger.error(
                f"k={self.k} > n_nodos={n_nodos}. "
                "No es posible generar esa cantidad de partes."
            )
            return Solution(
                estrategia=KQNODES_LABEL,
                perdida=INFTY_POS,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=np.array(DUMMY_ARR, dtype=np.float32),
                particion=ERROR_PARTITION,
                tiempo_total=0.0,
                hablar=False,
            )

        # ── PASO 4: Greedy iterativo — k-1 cortes de Queyranne ────────────
        partes: list = []
        V_res: list = list(V)

        for iteracion in range(1, self.k):
            self.logger.critic(
                f"KQNodes k={self.k}: iteración {iteracion}/{self.k - 1} "
                f"— {len(V_res)} vértices restantes"
            )

            if len(V_res) <= 2:
                # Poda: sistema residual demasiado pequeño para Queyranne.
                # H-2: extrae singleton directo para evitar min() sobre dict vacío.
                corte = (V_res[0],)
            else:
                self._reiniciar_memoria()
                self.vertices = set(map(tuple, V_res))
                corte = self.algorithm(list(V_res))

            partes.append(corte)
            V_res = self._residual(V_res, corte)

            if not V_res:
                self.logger.critic("Residual vacío antes de completar k partes.")
                break

        # La última parte = todo el residual restante (H-2: siempre se agrega)
        if V_res:
            partes.append(tuple(V_res))

        self.logger.critic(f"KQNodes: {len(partes)} partes encontradas.")

        # ── PASO 5: Evaluar φ ──────────────────────────────────────────────
        D_SP   = self._producto_tensorial_k_partes(partes)
        perdida = emd_efecto(D_SP, self.sia_dists_marginales)
        fmt     = self._formatear_k_particion(partes)

        self.logger.critic(f"k-MIP encontrada: φ={perdida:.6f}")

        # ── PASO 6: Retornar solución ──────────────────────────────────────
        return Solution(
            estrategia=KQNODES_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=D_SP,
            particion=fmt,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            hablar=False,
        )

    # ── Override: funcion_submodular con memoria_delta ──────────────────────

    def funcion_submodular(
        self,
        deltas: Union[tuple, list],
        omegas: list,
    ):
        """
        Evalúa EMD(δ) y EMD(δ∪Ω) con caché inter-iteraciones (memoria_delta).

        Override de QNodes.funcion_submodular. La lógica es idéntica, pero
        el EMD individual del delta se cachea en memoria_delta (que persiste
        entre las k-1 pasadas de Queyranne), evitando recomputar biparticiones
        para deltas que reaparecen en iteraciones posteriores.

        Returns:
            (φ_unión, φ_delta, dist_delta)
        """
        # ── Extraer índices de δ ───────────────────────────────────────────
        planos_delta = self._aplanar_vertices(deltas)
        idxs_efecto_delta = sorted(v[1] for v in planos_delta if v[0] == EFECTO)
        dims_actual_delta  = sorted(v[1] for v in planos_delta if v[0] == ACTUAL)

        clave_delta = (tuple(dims_actual_delta), tuple(idxs_efecto_delta))

        # ── Evaluación individual de δ (con caché) ────────────────────────
        if clave_delta in self.memoria_delta:
            emd_delta, dist_delta = self.memoria_delta[clave_delta]
        else:
            P_delta   = self.sia_subsistema.bipartir(
                np.array(idxs_efecto_delta, dtype=np.int8),
                np.array(dims_actual_delta,  dtype=np.int8),
            )
            dist_delta = P_delta.distribucion_marginal()
            emd_delta  = emd_efecto(dist_delta, self.sia_dists_marginales)

            if emd_delta == 0.0:
                self.memoria_delta[clave_delta] = (emd_delta, dist_delta)
                # Atajo de salida temprana documentado en el pseudocódigo
                return INFTY_POS, emd_delta, dist_delta

            self.memoria_delta[clave_delta] = (emd_delta, dist_delta)

        # ── Evaluación de la unión δ ∪ Ω ──────────────────────────────────
        idxs_efecto_union = list(idxs_efecto_delta)
        dims_actual_union  = list(dims_actual_delta)

        for omega in omegas:
            for v in self._aplanar_vertices(omega):
                if v[0] == EFECTO:
                    idxs_efecto_union.append(v[1])
                else:
                    dims_actual_union.append(v[1])

        P_union   = self.sia_subsistema.bipartir(
            np.array(idxs_efecto_union, dtype=np.int8),
            np.array(dims_actual_union,  dtype=np.int8),
        )
        dist_union = P_union.distribucion_marginal()
        emd_union  = emd_efecto(dist_union, self.sia_dists_marginales)

        return emd_union, emd_delta, dist_delta

    # ── Subrutinas de soporte ───────────────────────────────────────────────

    def _reiniciar_memoria(self) -> None:
        """
        Reinicia solo memoria_particiones entre iteraciones del greedy.
        memoria_delta NO se reinicia: es el caché inter-iteraciones.
        """
        self.memoria_particiones = {}

    def _residual(self, vertices_todos: list, corte) -> list:
        """
        Devuelve vertices_todos menos los vértices del corte.

        Args:
            vertices_todos: lista de vértices activos.
            corte: tupla o lista (posiblemente anidada) con los vértices extraídos.

        Returns:
            Lista de vértices residuales.
        """
        conjunto_corte = set(map(tuple, self._aplanar_vertices(corte)))
        residual = []
        for v in vertices_todos:
            componentes = set(map(tuple, self._aplanar_vertices(v)))
            if not componentes.intersection(conjunto_corte):
                residual.append(v)
        return residual

    def _aplanar_vertices(self, v) -> list:
        """
        Convierte estructuras potencialmente anidadas (listas de listas de tuplas,
        resultado de los par_candidato generados por Queyranne) a una lista plana
        de tuplas (tiempo, índice).

        Casos manejados:
            - Tupla simple  (tiempo, índice)  → [(tiempo, índice)]
            - Lista o tupla de vértices/grupos → recursión sobre cada elemento
        """
        # Caso base: tupla (int, int) — vértice simple
        if (
            isinstance(v, tuple)
            and len(v) == 2
            and isinstance(v[0], (int, np.integer))
            and isinstance(v[1], (int, np.integer))
        ):
            return [v]

        # Caso recursivo: lista, tupla de grupos, o par_candidato anidado
        resultado = []
        for elemento in v:
            resultado.extend(self._aplanar_vertices(elemento))
        return resultado

    def _producto_tensorial_k_partes(self, partes: list) -> np.ndarray:
        """
        Construye D_SP de shape (m,) por asignación posicional (scatter).

        El nombre es un residuo histórico: NO aplica np.kron. En su lugar
        asigna la probabilidad marginal de cada nodo futuro en su posición
        canónica dentro de Sub.indices_ncubos, produciendo un vector comparable
        con sia_dists_marginales mediante emd_efecto.

        Corrección H-1: los roles EFECTO/ACTUAL se determinan directamente por
        v[0] == EFECTO (1) o v[0] == ACTUAL (0), eliminando la heurística
        dinámica que fallaba en subsistemas simétricos.

        Args:
            partes: lista de k grupos de vértices (posiblemente anidados).

        Returns:
            np.ndarray de shape (m,) con la distribución del sistema partido.
        """
        indices_ncubos = self.sia_subsistema.indices_ncubos
        dims_ncubos    = self.sia_subsistema.dims_ncubos
        m = indices_ncubos.size

        # mapa: índice_real_futuro → probabilidad OFF
        nodo_a_prob: dict[int, float] = {}

        # Todos los presentes del subsistema, en índices reales
        arr_presentes_todos = dims_ncubos.copy()

        for parte in partes:
            planos = self._aplanar_vertices(parte)

            # H-1 corregido: usar constantes EFECTO/ACTUAL directamente
            futuros_reales  = [v[1] for v in planos if v[0] == EFECTO]
            presentes_reales = [v[1] for v in planos if v[0] == ACTUAL]

            if not futuros_reales:
                continue

            # Si la parte no tiene presentes propios, usar todos los del subsistema
            arr_presentes = (
                np.array(presentes_reales, dtype=np.int8)
                if presentes_reales
                else arr_presentes_todos
            )
            arr_futuros = np.array(futuros_reales, dtype=np.int8)

            parte_obj = self.sia_subsistema.bipartir(arr_futuros, arr_presentes)
            dist_parte = parte_obj.distribucion_marginal()

            # dist_parte[i] corresponde al i-ésimo futuro de arr_futuros
            for i, nodo_real in enumerate(futuros_reales):
                nodo_a_prob[int(nodo_real)] = float(dist_parte[i])

        # Reconstruir en el orden canónico de indices_ncubos (coincide con D_Sub)
        D_SP = np.array(
            [nodo_a_prob.get(int(n), 0.0) for n in indices_ncubos],
            dtype=np.float32,
        )
        return D_SP

    def _formatear_k_particion(self, partes: list) -> str:
        """
        Representación legible de la k-MIP:

            |Futuros_1||Futuros_2||...||Futuros_k|
            |Presentes_1||Presentes_2||...||Presentes_k|
        """
        linea_top = ""
        linea_bot = ""

        indices_ncubos = self.sia_subsistema.indices_ncubos
        dims_ncubos    = self.sia_subsistema.dims_ncubos

        for parte in partes:
            planos = self._aplanar_vertices(parte)

            futuros_reales  = [v[1] for v in planos if v[0] == EFECTO]
            presentes_reales = [v[1] for v in planos if v[0] == ACTUAL]

            str_fut = (
                ",".join(ABECEDARY[int(f)] for f in futuros_reales)
                if futuros_reales else VOID_STR
            )
            str_pre = (
                ",".join(LOWER_ABECEDARY[int(p)] for p in presentes_reales)
                if presentes_reales else VOID_STR
            )

            ancho      = max(len(str_fut), len(str_pre)) + 2
            linea_top += f"|{str_fut:^{ancho}}|"
            linea_bot += f"|{str_pre:^{ancho}}|"

        return f"{linea_top}\n{linea_bot}\n"
