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
    4. _evaluar_candidatos_exacto()    [NUEVO — concatenate por nodo + EMD]
    5. Retornar Solution con k-MIP     [misma estructura que GeometricSIA]

    ── CORRECCIÓN v2: _distribucion_k_partes() ──────────────────────────────
    distribucion_marginal() retorna UN ESCALAR POR NODO (prob OFF del nodo),
    produciendo un vector de shape (n_ncubos,). emd_efecto compara este
    vector contra sia_dists_marginales que también tiene shape (n_nodos,).

    Por tanto la distribución del sistema partido SP debe construirse
    con np.concatenate (apilar escalares por nodo) y NO con np.kron
    (que expande al espacio de 2^n estados y produce shapes incompatibles
    para particiones desiguales como (1,3) o (1,1,2)).

    Ejemplo concreto para n=4, k=2, partición ({A,B}, {C,D}):
        dist_parte1 = bipartir({A,B}, presentes).dist_marginal()
                    = [p_OFF_A, p_OFF_B]          shape (2,)
        dist_parte2 = bipartir({C,D}, presentes).dist_marginal()
                    = [p_OFF_C, p_OFF_D]          shape (2,)
        dist_SP     = concatenate([d1, d2])
                    = [p_OFF_A, p_OFF_B, p_OFF_C, p_OFF_D]  shape (4,) ✓
        sia_dists   = [p_OFF_A, p_OFF_B, p_OFF_C, p_OFF_D]  shape (4,) ✓
        emd_efecto  = sum|dist_SP - sia_dists|  ← compatible

    Con np.kron(d1, d2) = shape (4,) para (2,2) PERO shape (3,) para (1,3)
    → crash para cualquier partición desigual.
    ─────────────────────────────────────────────────────────────────────────

    Args:
    ----
        gestor (Manager): Gestor de datos del proyecto base.
        k (int): Número de partes. k=2 reproduce GeometricSIA original.
        top_n (int): Cuántos candidatos geométricos pasan a evaluación
                     exacta con EMD. Trade-off precisión/velocidad.
                     Por defecto 10. Para garantía de óptimo usar None.
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

        super().__init__(gestor)
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

        Para k=2 el resultado es comparable a GeometricSIA. Para k>2
        amplía el espacio de búsqueda usando la tabla de costos como
        heurística de filtrado antes de la evaluación exacta con EMD.
        """
        # ── 1. Preparar subsistema + construir tabla_transiciones ──────────
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        self._flat_data = [
            ncubo.data.ravel()
            for ncubo in self.sia_subsistema.ncubos
        ]

        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final   = 1 - self.estado_inicial

        self.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))
        self.caminos = {0: [self.estado_inicial.tolist()]}
        self.tabla_transiciones[
            tuple(self.caminos[0][0]), tuple(self.caminos[0][0])
        ] = [0.0] * len(self.sia_subsistema.indices_ncubos)

        for nivel in range(1, len(self.estado_inicial) + 1):
            self.calcular_costos_nivel(self.estado_final, nivel)

        self.logger.critic("Tabla de transiciones construida.")

        # ── 2. Validar k ──────────────────────────────────────────────────
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
        Genera todas las k-particiones, las ordena por costo geométrico
        y retorna las top_n mejores (o todas si top_n es None).
        """
        puntuaciones = []
        for particion in self._generar_k_particiones(indices_futuros, self.k):
            costo = self._costo_geometrico(particion)
            puntuaciones.append((costo, particion))

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

        Consulta la entrada (estado_inicial → estado_final) de la tabla
        y suma los costos tx de las variables de cada grupo.
        """
        key = tuple(self.caminos[0][0]), tuple(self.estado_final)
        costos_por_variable: list[float] = self.tabla_transiciones[key]

        costo_total = 0.0
        for grupo in grupos_futuros:
            costo_total += sum(
                costos_por_variable[idx]
                for idx in grupo
                if costos_por_variable[idx] is not None
            )
        return costo_total

    # ── Fase 4: Evaluación exacta ───────────────────────────────────────────

    def _evaluar_candidatos_exacto(
        self,
        candidatos: list[tuple[tuple[int, ...], ...]],
    ) -> tuple[float, np.ndarray, str]:
        """
        Evalúa cada candidato con EMD exacto y retorna el mejor.
        """
        indices_presentes = list(range(self.sia_subsistema.dims_ncubos.size))

        mejor_perdida = INFTY_POS
        mejor_dist    = np.array(DUMMY_ARR, dtype=np.float32)
        mejor_fmt     = ERROR_PARTITION

        for grupos_futuros in candidatos:
            dist_sp = self._distribucion_k_partes(grupos_futuros, indices_presentes)
            perdida = emd_efecto(dist_sp, self.sia_dists_marginales)

            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_dist    = dist_sp
                mejor_fmt     = self._formatear_k_particion(
                    grupos_futuros, indices_presentes
                )

        return mejor_perdida, mejor_dist, mejor_fmt

    def _distribucion_k_partes(
        self,
        grupos_futuros: tuple[tuple[int, ...], ...],
        indices_presentes: list[int],
    ) -> np.ndarray:
        """
        Construye la distribución del sistema partido SP para una k-partición.

        ── Por qué concatenate y no kron ────────────────────────────────────
        distribucion_marginal() retorna UN ESCALAR POR NODO (prob de OFF),
        produciendo shape (n_ncubos_en_la_parte,).

        emd_efecto(u, v) = sum|u_i - v_i| donde u_i y v_i son la prob OFF
        del nodo i. Ambos vectores deben tener shape (n_nodos_total,).

        La distribución SP se construye apilando los escalares de cada parte
        EN EL ORDEN CORRECTO DE LOS ÍNDICES REALES de los nodos futuros.
        np.concatenate hace exactamente eso.

        np.kron produciría el espacio de 2^n estados (correcto para emd_causal
        con distancia Hamming), pero emd_efecto opera sobre n escalares, no
        sobre 2^n estados. Usar kron aquí es matemáticamente incorrecto y
        además produce shapes variables que crashean para particiones desiguales.

        Ejemplo n=4, k=2, grupos=((0,1),(2,3)):
            bipartir([A,B], presentes).dist_marginal() → [p_A, p_B]  shape (2,)
            bipartir([C,D], presentes).dist_marginal() → [p_C, p_D]  shape (2,)
            concatenate → [p_A, p_B, p_C, p_D]  shape (4,)  ← coincide con sia

        Args:
        ----
            grupos_futuros: k-partición de índices locales de futuros.
                            Cada tupla contiene índices locales (0-based).
            indices_presentes: índices locales de presentes (siempre todos).

        Returns:
        -------
            np.ndarray shape (n_futuros_total,): distribución SP ordenada
            por índice real de nodo futuro, lista para emd_efecto.
        """
        arr_presentes = self.sia_subsistema.dims_ncubos[indices_presentes]

        # Acumular (indice_real, valor_escalar) para ordenar correctamente
        # antes de concatenar, garantizando que el orden coincide con
        # sia_dists_marginales (que sigue el orden de indices_ncubos)
        nodo_a_prob: dict[int, float] = {}

        for grupo in grupos_futuros:
            # Convertir índices locales a índices reales del subsistema
            futuros_reales = self.sia_subsistema.indices_ncubos[list(grupo)]
            arr_futuros    = np.array(futuros_reales, dtype=np.int8)

            dist_parte = (
                self.sia_subsistema
                .bipartir(arr_futuros, arr_presentes)
                .distribucion_marginal()
            )
            # dist_parte[i] corresponde al nodo futuros_reales[i]
            for i, nodo_real in enumerate(futuros_reales):
                nodo_a_prob[int(nodo_real)] = float(dist_parte[i])

        # Reconstruir en el orden canónico de indices_ncubos del subsistema
        # para que coincida con sia_dists_marginales
        dist_sp = np.array(
            [nodo_a_prob[int(n)] for n in self.sia_subsistema.indices_ncubos],
            dtype=np.float32,
        )
        return dist_sp

    # ── Generador de k-particiones (Stirling) ──────────────────────────────

    def _generar_k_particiones(
        self,
        elementos: list[int],
        k: int,
    ) -> Generator[tuple[tuple[int, ...], ...], None, None]:
        """
        Genera todas las k-particiones del conjunto `elementos`.

        Implementación recursiva: S(n,k) = k·S(n-1,k) + S(n-1,k-1).
        Fija el primer elemento en el primer grupo para evitar duplicados.

        Conteos verificados:
            S(3,2)=3,  S(4,2)=7,  S(4,3)=6,  S(6,3)=90,  S(10,3)=9330
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
        resto   = elementos[1:]

        for sub in self._generar_k_particiones(resto, k - 1):
            yield ((primero,),) + sub

        for sub in self._generar_k_particiones(resto, k):
            for i in range(k):
                grupos    = list(sub)
                grupos[i] = tuple(sorted(grupos[i] + (primero,)))
                yield tuple(grupos)

    # ── Formateo de salida ──────────────────────────────────────────────────

    def _formatear_k_particion(
        self,
        grupos_futuros: tuple[tuple[int, ...], ...],
        indices_presentes: list[int],
    ) -> str:
        """
        Representación legible de la k-MIP:

            |Futuros_1||Futuros_2||...||Futuros_k|
            |Presentes||Presentes||...||Presentes |
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
            ancho      = max(len(str_fut), len(str_pre)) + 2
            linea_top += f"|{str_fut:^{ancho}}|"
            linea_bot += f"|{str_pre:^{ancho}}|"

        return f"{linea_top}\n{linea_bot}\n"