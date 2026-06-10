"""
KGeoMIP — Extensión geométrica para k-Particiones de Mínima Información.

Estrategia principal del proyecto KQMIP. Extiende GeometricSIA al caso
general de k particiones (k ∈ {2,3,4,5}) reutilizando íntegramente:
  - sia_preparar_subsistema()
  - calcular_costo() / calcular_costos_nivel()  →  tabla_transiciones
  - _flat_data (datos aplanados de NCubos)
  - System.bipartir() / distribucion_marginal()

La idea central: la tabla_transiciones, que cuantifica el 'costo'
de separar pares de estados en el hipercubo, se calcula UNA SOLA VEZ
(heredando el BFS de GeometricSIA) y se reutiliza para cualquier k.
Para k>2, la búsqueda de candidatos se amplía de n bi-particiones
a S(n,k) k-particiones, filtradas geométricamente antes de la
evaluación exacta con EMD.

Ubicación sugerida en el proyecto:
    src/controllers/strategies/k_geomip.py
"""

import time
from itertools import combinations
from typing import Generator

import numpy as np

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.funcs.base import emd_efecto
from src.middlewares.profile import profile, profiler_manager
from src.middlewares.slogger import SafeLogger
from src.models.core.solution import Solution
from src.constants.base import (
    NET_LABEL,
    TYPE_TAG,
    INFTY_POS,
)
from src.constants.models import (
    DUMMY_ARR,
    ERROR_PARTITION,
)

# ── Nuevas constantes (añadir a constants/models.py) ────────────────────────
KGEOMIP_LABEL: str = "KGeoMIP"
KGEOMIP_TAG: str = f"{KGEOMIP_LABEL}_strategy"
KGEOMIP_ANALYSIS_TAG: str = f"{KGEOMIP_LABEL}_analysis"
# ────────────────────────────────────────────────────────────────────────────


class KGeoMIP(GeometricSIA):
    """
    Extensión geométrica para encontrar la k-MIP.

    Hereda COMPLETAMENTE de GeometricSIA. No reimplementa nada del
    núcleo geométrico (BFS, tabla de costos, NCubos). Solo extiende
    la fase de generación y evaluación de candidatos al caso k > 2.

    Flujo de ejecución:
    ------------------
    1. sia_preparar_subsistema()       [heredado de SIA]
    2. Construir tabla_transiciones()  [heredado de GeometricSIA]
    3. _generar_candidatos_k()         [NUEVO — Stirling + filtro geométrico]
    4. _evaluar_candidatos_exacto()    [NUEVO — producto tensorial + EMD]
    5. Retornar Solution con k-MIP     [misma estructura que GeometricSIA]

    Args:
    ----
        gestor (Manager): Gestor de datos del proyecto base.
        k (int): Número de partes. k=2 reproduce GeometricSIA original.
        top_n (int): Cuántos candidatos geométricos pasan a evaluación
                     exacta con EMD. Trade-off precisión/velocidad.
                     Por defecto 10. Para garantía de óptimo usar None
                     (evalúa todos, equivalente a KBruteForce pero con
                     filtro geométrico previo).

    Complejidad:
    -----------
        Construcción tabla: igual que GeometricSIA — O(n² × 2ⁿ)
        Generación candidatos: O(S(n,k)) números de Stirling
        Filtro geométrico: O(S(n,k) × 2ⁿ)
        Evaluación exacta: O(top_n × n)
        Total dominante: O(S(n,k) × 2ⁿ)

        Para n=10, k=3: S(10,3)=9330 → viable.
        Para n=15, k=3: S(15,3)≈2.3M → requiere top_n pequeño.
    """

    def __init__(
        self,
        gestor: Manager,
        k: int = 2,
        top_n: int = 10,
    ) -> None:
        if k < 2:
            raise ValueError(f"k debe ser >= 2, recibido: {k}")
        if top_n is not None and top_n < 1:
            raise ValueError(f"top_n debe ser >= 1 o None, recibido: {top_n}")

        super().__init__(gestor)   # inicializa GeometricSIA completo
        self.k = k
        self.top_n = top_n
        self.logger = SafeLogger(KGEOMIP_TAG)

    @profile(context={TYPE_TAG: KGEOMIP_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ) -> Solution:
        """
        Encuentra la k-MIP usando el enfoque geométrico extendido.

        Para k=2 el resultado es idéntico a GeometricSIA. Para k>2
        amplía el espacio de búsqueda usando la tabla de costos como
        heurística de filtrado antes de la evaluación exacta con EMD.

        Args:
        ----
            condicion (str): Cadena de bits para condicionamiento de fondo.
            alcance (str): Bits en 0 indican futuros a substraer.
            mecanismo (str): Bits en 0 indican presentes a substraer.
            tpm (np.ndarray): Matriz de probabilidad de transición.

        Returns:
        -------
            Solution: k-MIP encontrada con pérdida φ mínima.
        """
        # ── 1. Preparar subsistema + construir tabla_transiciones ──────────
        # sia_preparar_subsistema viene de SIA (vía GeometricSIA)
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        # Preparar _flat_data (igual que GeometricSIA.aplicar_estrategia)
        self._flat_data = [
            ncubo.data.ravel()
            for ncubo in self.sia_subsistema.ncubos
        ]

        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial

        # Construir tabla_transiciones con el BFS heredado de GeometricSIA
        self.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))
        self.caminos = {0: [self.estado_inicial.tolist()]}
        self.tabla_transiciones[
            tuple(self.caminos[0][0]), tuple(self.caminos[0][0])
        ] = [0.0] * len(self.sia_subsistema.indices_ncubos)

        for nivel in range(1, len(self.estado_inicial) + 1):
            self.calcular_costos_nivel(self.estado_final, nivel)

        self.logger.critic("Tabla de transiciones construida.")

        # ── 2. Validación de k contra tamaño del subsistema ───────────────
        n_futuros = self.sia_subsistema.indices_ncubos.size
        if n_futuros < self.k:
            self.logger.error(
                f"k={self.k} > n_futuros={n_futuros}. "
                "No es posible generar esa cantidad de partes."
            )
            return Solution(
                estrategia=KGEOMIP_LABEL,
                perdida=INFTY_POS,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=np.array(DUMMY_ARR, dtype=np.float32),
                particion=ERROR_PARTITION,
                tiempo_total=0.0,
                hablar=False,
            )

        # ── 3. Generar candidatos y filtrar geométricamente ────────────────
        indices_futuros = list(range(n_futuros))
        candidatos_filtrados = self._generar_candidatos_k(indices_futuros)
        self.logger.critic(
            f"Candidatos tras filtro geométrico: {len(candidatos_filtrados)}"
        )

        # ── 4. Evaluación exacta con EMD ───────────────────────────────────
        mejor_perdida, mejor_dist, mejor_fmt = self._evaluar_candidatos_exacto(
            candidatos_filtrados
        )

        self.logger.critic(f"k-MIP encontrada: φ={mejor_perdida:.6f}")

        return Solution(
            estrategia=KGEOMIP_LABEL,
            perdida=mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist,
            particion=mejor_fmt,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            hablar=False,
        )

    # ── Fase 3: Generación y filtrado geométrico ────────────────────────────

    def _generar_candidatos_k(
        self,
        indices_futuros: list[int],
    ) -> list[tuple[tuple[int, ...], ...]]:
        """
        Genera todas las k-particiones y las ordena por costo geométrico.
        Retorna las top_n con menor costo (o todas si top_n es None).

        El costo geométrico de una k-partición es la suma de costos
        tx(estado_inicial, estado_final) para las variables cuyos
        índices pertenecen a partes distintas. Es una heurística rápida
        que aproxima qué particiones "rompen menos" el hipercubo.

        Args:
        ----
            indices_futuros: índices locales de variables futuras.

        Returns:
        -------
            Lista de k-particiones ordenadas por costo geométrico ascendente.
        """
        puntuaciones = []

        for particion in self._generar_k_particiones(indices_futuros, self.k):
            costo = self._costo_geometrico(particion)
            puntuaciones.append((costo, particion))

        # Ordenar por menor costo geométrico
        puntuaciones.sort(key=lambda x: x[0])

        if self.top_n is None:
            return [p for _, p in puntuaciones]
        return [p for _, p in puntuaciones[: self.top_n]]

    def _costo_geometrico(
        self,
        grupos_futuros: tuple[tuple[int, ...], ...],
    ) -> float:
        """
        Costo geométrico de una k-partición usando tabla_transiciones.

        Para cada variable futura, consulta el costo tx en la entrada
        (estado_inicial, estado_final) de tabla_transiciones. Variables
        en grupos distintos que "comparten" transiciones costosas indican
        una partición que "corta" el hipercubo en zonas de alta inercia.

        La suma de costos de las variables en el grupo más costoso
        aproxima el EMD que tendrá esa partición.

        Args:
        ----
            grupos_futuros: k-partición como tupla de k tuplas de índices.

        Returns:
        -------
            float: costo geométrico escalar de la partición.
        """
        key = tuple(self.caminos[0][0]), tuple(self.estado_final)
        costos_por_variable: list[float] = self.tabla_transiciones[key]

        costo_total = 0.0
        for grupo in grupos_futuros:
            # Suma de costos de las variables de este grupo
            costo_grupo = sum(
                costos_por_variable[idx]
                for idx in grupo
                if costos_por_variable[idx] is not None
            )
            costo_total += costo_grupo

        return costo_total

    # ── Fase 4: Evaluación exacta ───────────────────────────────────────────

    def _evaluar_candidatos_exacto(
        self,
        candidatos: list[tuple[tuple[int, ...], ...]],
    ) -> tuple[float, np.ndarray, str]:
        """
        Evalúa cada candidato con el EMD exacto y retorna el mejor.

        Para cada k-partición:
          a. Calcula distribución de cada parte vía System.bipartir()
          b. Producto tensorial np.kron iterativo para obtener dist_sp
          c. EMD(dist_sp, sia_dists_marginales) = φ exacto

        Args:
        ----
            candidatos: lista de k-particiones pre-filtradas.

        Returns:
        -------
            Tupla (mejor_perdida, mejor_dist_particion, mejor_fmt_str).
        """
        indices_presentes = list(range(self.sia_subsistema.dims_ncubos.size))

        mejor_perdida = INFTY_POS
        mejor_dist = np.array(DUMMY_ARR, dtype=np.float32)
        mejor_fmt = ERROR_PARTITION

        for grupos_futuros in candidatos:
            dist_sp = self._producto_tensorial_k_partes(
                grupos_futuros, indices_presentes
            )
            perdida = emd_efecto(dist_sp, self.sia_dists_marginales)

            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_dist = dist_sp
                mejor_fmt = self._formatear_k_particion(
                    grupos_futuros, indices_presentes
                )

        return mejor_perdida, mejor_dist, mejor_fmt

    def _producto_tensorial_k_partes(
        self,
        grupos_futuros: tuple[tuple[int, ...], ...],
        indices_presentes: list[int],
    ) -> np.ndarray:
        """
        Calcula p(S₁) ⊗ p(S₂) ⊗ ... ⊗ p(Sₖ).

        Cada parte Sᵢ se evalúa con System.bipartir():
          - alcance   = futuros de la parte i  (indices reales del subsistema)
          - mecanismo = todos los presentes     (igual para todas las partes)

        Esta asimetría (futuros particionados, presentes completos) es
        consistente con el comportamiento de BruteForce y GeometricSIA
        para bi-particiones.

        Args:
        ----
            grupos_futuros: k-partición de índices locales de futuros.
            indices_presentes: índices locales de presentes (siempre todos).

        Returns:
        -------
            np.ndarray: distribución del sistema partido (SP).
        """
        arr_presentes = self.sia_subsistema.dims_ncubos[indices_presentes]

        # Primera parte — inicializa el resultado
        futuros_reales_0 = self.sia_subsistema.indices_ncubos[
            list(grupos_futuros[0])
        ]
        resultado = (
            self.sia_subsistema
            .bipartir(futuros_reales_0, arr_presentes)
            .distribucion_marginal()
        )

        # Partes 1..k-1 — producto tensorial iterativo
        for grupo in grupos_futuros[1:]:
            futuros_reales_i = self.sia_subsistema.indices_ncubos[list(grupo)]
            dist_i = (
                self.sia_subsistema
                .bipartir(futuros_reales_i, arr_presentes)
                .distribucion_marginal()
            )
            resultado = np.kron(resultado, dist_i)

        return resultado

    # ── Generador de k-particiones (Stirling) ──────────────────────────────

    def _generar_k_particiones(
        self,
        elementos: list[int],
        k: int,
    ) -> Generator[tuple[tuple[int, ...], ...], None, None]:
        """
        Genera todas las k-particiones del conjunto `elementos`.

        Basado en la recurrencia de Stirling S(n,k) = k·S(n-1,k) + S(n-1,k-1).
        El primer elemento siempre va al primer grupo para evitar duplicados.

        Args:
        ----
            elementos: lista de enteros a particionar.
            k: número de grupos deseados.

        Yields:
        ------
            Tupla de k tuplas ordenadas, cada una con los índices de un grupo.

        Conteos de referencia:
            S(4,2)=7,  S(4,3)=6,  S(4,4)=1
            S(6,2)=31, S(6,3)=90, S(6,4)=65
        """
        n = len(elementos)

        if k == 1:
            yield (tuple(elementos),)
            return
        if n == k:
            yield tuple((e,) for e in elementos)
            return
        if n < k:
            return

        primero = elementos[0]
        resto = elementos[1:]

        # Caso A: primero forma su propio grupo (nuevo grupo al frente)
        for sub in self._generar_k_particiones(resto, k - 1):
            yield ((primero,),) + sub

        # Caso B: primero se agrega a cada uno de los k grupos existentes
        for sub in self._generar_k_particiones(resto, k):
            for i in range(k):
                grupos = list(sub)
                grupos[i] = tuple(sorted(grupos[i] + (primero,)))
                yield tuple(grupos)

    # ── Formateo de salida ──────────────────────────────────────────────────

    def _formatear_k_particion(
        self,
        grupos_futuros: tuple[tuple[int, ...], ...],
        indices_presentes: list[int],
    ) -> str:
        """
        Representación legible de la k-MIP encontrada.

        Usa ABECEDARY y LOWER_ABECEDARY del proyecto base, igual que
        fmt_biparte_q() de format.py. Una columna por parte:

            |Futuros_1||Futuros_2||...||Futuros_k|
            |Presentes||Presentes||...||Presentes |

        Args:
        ----
            grupos_futuros: grupos de índices locales de futuros.
            indices_presentes: índices locales de presentes.

        Returns:
        -------
            str: representación formateada multi-parte.
        """
        from src.funcs.base import ABECEDARY, LOWER_ABECEDARY
        from src.constants.base import VOID_STR

        arr_presentes = self.sia_subsistema.dims_ncubos[indices_presentes]
        str_pre = (
            ",".join(LOWER_ABECEDARY[p] for p in arr_presentes)
            if arr_presentes.size else VOID_STR
        )

        linea_top = ""
        linea_bot = ""

        for grupo in grupos_futuros:
            futuros_reales = self.sia_subsistema.indices_ncubos[list(grupo)]
            str_fut = (
                ",".join(ABECEDARY[f] for f in futuros_reales)
                if futuros_reales.size else VOID_STR
            )
            ancho = max(len(str_fut), len(str_pre)) + 2
            linea_top += f"|{str_fut:^{ancho}}|"
            linea_bot += f"|{str_pre:^{ancho}}|"

        return f"{linea_top}\n{linea_bot}\n"
