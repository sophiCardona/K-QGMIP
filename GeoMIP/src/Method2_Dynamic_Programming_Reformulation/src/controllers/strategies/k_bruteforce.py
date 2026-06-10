"""
KBruteForce — Fuerza bruta para k-particiones.

Rol en el proyecto: VALIDADOR. No es la estrategia principal.
Encuentra la k-MIP óptima garantizada por búsqueda exhaustiva.
Límite práctico: n <= 6 nodos, k <= 4. Para sistemas mayores
el tiempo de ejecución se vuelve intratable.

Hereda de GeometricSIA para reutilizar:
- sia_preparar_subsistema()
- sia_subsistema (System ya preparado)
- sia_dists_marginales (distribución del sistema completo)
- sia_tiempo_inicio

Ubicación sugerida en el proyecto:
    src/controllers/strategies/k_bruteforce.py
"""

import time
from itertools import combinations
from typing import Generator

import numpy as np

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.funcs.base import emd_efecto
from src.funcs.format import fmt_biparte_q
from src.middlewares.profile import profile, profiler_manager
from src.middlewares.slogger import SafeLogger
from src.models.core.solution import Solution
from src.constants.base import (
    NET_LABEL,
    TYPE_TAG,
    INFTY_POS,
    ACTUAL,
    EFECTO,
)
from src.constants.models import (
    DUMMY_ARR,
    ERROR_PARTITION,
)

# ── Nuevas constantes (añadir a constants/models.py) ────────────────────────
KBRUTEFORCE_LABEL: str = "KBruteForce"
KBRUTEFORCE_TAG: str = f"{KBRUTEFORCE_LABEL}_strategy"
KBRUTEFORCE_ANALYSIS_TAG: str = f"{KBRUTEFORCE_LABEL}_analysis"
# ────────────────────────────────────────────────────────────────────────────


class KBruteForce(GeometricSIA):
    """
    Búsqueda exhaustiva de la k-Partición de Mínima Información (k-MIP).

    Evalúa TODAS las k-particiones posibles del sistema y retorna
    aquella con menor pérdida EMD. Garantiza el óptimo global.

    Hereda de GeometricSIA en vez de SIA directamente para reutilizar
    sia_preparar_subsistema() y la infraestructura de NCubos.

    Args:
    ----
        gestor (Manager): Gestor de datos del proyecto base.
        k (int): Número de partes de la partición. k=2 reproduce
                 BruteForce original. k in {2,3,4,5}.

    Complejidad:
    -----------
        Temporal: O(S(n,k) × n) donde S(n,k) es el número de Stirling
                  del segundo tipo. Para n=6, k=3: S(6,3)=90 particiones.
        Espacial: O(S(n,k)) para almacenar candidatos.

    Límite práctico recomendado: n <= 6.
    """

    def __init__(self, gestor: Manager, k: int = 2) -> None:
        if k < 2:
            raise ValueError(f"k debe ser >= 2, recibido: {k}")
        super().__init__(gestor)
        self.k = k
        self.logger = SafeLogger(KBRUTEFORCE_TAG)

    @profile(context={TYPE_TAG: KBRUTEFORCE_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ) -> Solution:
        """
        Encuentra la k-MIP por búsqueda exhaustiva sobre todas las
        k-particiones del subsistema.

        Args:
        ----
            condicion (str): Cadena de bits para condicionamiento de fondo.
            alcance (str): Bits en 0 indican dimensiones a substraer del futuro.
            mecanismo (str): Bits en 0 indican dimensiones a substraer del presente.
            tpm (np.ndarray): Matriz de probabilidad de transición.

        Returns:
        -------
            Solution: Objeto con la k-MIP encontrada, su pérdida φ y tiempo.
        """
        # ── 1. Preparar subsistema (heredado de SIA vía GeometricSIA) ──────
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        futuros_todos = self.sia_subsistema.indices_ncubos   # variables futuras
        presentes_todos = self.sia_subsistema.dims_ncubos    # variables presentes
        n_futuros = futuros_todos.size
        n_presentes = presentes_todos.size

        if n_futuros < self.k:
            self.logger.error(
                f"k={self.k} > n_futuros={n_futuros}. "
                "No es posible generar esa cantidad de partes."
            )
            return Solution(
                estrategia=KBRUTEFORCE_LABEL,
                perdida=INFTY_POS,
                distribucion_subsistema=self.sia_dists_marginales,
                distribucion_particion=np.array(DUMMY_ARR, dtype=np.float32),
                particion=ERROR_PARTITION,
                tiempo_total=0.0,
                hablar=False,
            )

        # ── 2. Búsqueda exhaustiva ─────────────────────────────────────────
        mejor_perdida = INFTY_POS
        mejor_dist_particion = np.array(DUMMY_ARR, dtype=np.float32)
        mejor_particion_fmt = ERROR_PARTITION

        indices_futuros = list(range(n_futuros))
        indices_presentes = list(range(n_presentes))

        n_evaluadas = 0
        for grupos_futuros in self._generar_k_particiones(indices_futuros, self.k):
            # Para cada agrupación de futuros, los presentes van completos
            # a cada parte (comportamiento equivalente al base para k=2)
            dist_sp = self._producto_tensorial_k_partes(
                grupos_futuros, indices_presentes
            )
            perdida = emd_efecto(dist_sp, self.sia_dists_marginales)
            n_evaluadas += 1

            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_dist_particion = dist_sp
                mejor_particion_fmt = self._formatear_k_particion(
                    grupos_futuros, indices_presentes
                )

        self.logger.critic(
            f"KBruteForce k={self.k}: {n_evaluadas} particiones evaluadas. "
            f"φ={mejor_perdida:.6f}"
        )

        return Solution(
            estrategia=KBRUTEFORCE_LABEL,
            perdida=mejor_perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=mejor_dist_particion,
            particion=mejor_particion_fmt,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            hablar=False,
        )

    # ── Núcleo matemático ───────────────────────────────────────────────────

    def _producto_tensorial_k_partes(
        self,
        grupos_futuros: list[tuple[int, ...]],
        indices_presentes: list[int],
    ) -> np.ndarray:
        """
        Calcula p(S₁) ⊗ p(S₂) ⊗ ... ⊗ p(Sₖ) para una k-partición.

        Cada parte Sᵢ se evalúa llamando a System.bipartir() con:
          - alcance  = futuros_parte_i  (los nodos futuros de esta parte)
          - mecanismo = presentes_todos  (todos los presentes, siempre)

        Esto es exactamente lo que hace BruteForce para k=2, extendido
        iterativamente con np.kron para k partes.

        Args:
        ----
            grupos_futuros: lista de k tuplas con índices de futuros por parte.
            indices_presentes: lista de índices de presentes del subsistema.

        Returns:
        -------
            np.ndarray: vector de distribución del sistema partido (SP).
        """
        arr_presentes = np.array(indices_presentes, dtype=np.int8)

        # Primera parte inicializa el resultado
        arr_futuros_0 = self.sia_subsistema.indices_ncubos[
            list(grupos_futuros[0])
        ]
        resultado = (
            self.sia_subsistema
            .bipartir(arr_futuros_0, arr_presentes)
            .distribucion_marginal()
        )

        # Producto tensorial iterativo para partes 1..k-1
        for grupo in grupos_futuros[1:]:
            arr_futuros_i = self.sia_subsistema.indices_ncubos[list(grupo)]
            dist_i = (
                self.sia_subsistema
                .bipartir(arr_futuros_i, arr_presentes)
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

        Implementación recursiva basada en la recurrencia de Stirling:
            S(n,k) = k·S(n-1,k) + S(n-1,k-1)

        Para evitar duplicados, el primer elemento siempre se asigna
        al primer grupo.

        Args:
        ----
            elementos: lista de enteros a particionar.
            k: número de grupos.

        Yields:
        ------
            Tupla de k tuplas, cada una con los índices de un grupo.

        Ejemplos:
        --------
            list(_generar_k_particiones([0,1,2], 2))
            → [((0,1),(2,)), ((0,2),(1,)), ((0,),(1,2))]  — 3 particiones = S(3,2)

            list(_generar_k_particiones([0,1,2], 3))
            → [((0,),(1,),(2,))]                          — 1 partición = S(3,3)
        """
        n = len(elementos)

        # Casos base
        if k == 1:
            yield (tuple(elementos),)
            return
        if n == k:
            yield tuple((e,) for e in elementos)
            return
        if n < k:
            return  # imposible particionar n elementos en k > n grupos

        primero = elementos[0]
        resto = elementos[1:]

        for sub in self._generar_k_particiones(resto, k - 1):
            # primero forma un grupo propio (grupo nuevo al frente)
            yield ((primero,),) + sub

        for sub in self._generar_k_particiones(resto, k):
            # primero se añade a cada uno de los k grupos existentes
            for i in range(k):
                grupos = list(sub)
                grupos[i] = tuple(sorted(grupos[i] + (primero,)))
                yield tuple(grupos)

    # ── Formateo de salida ──────────────────────────────────────────────────

    def _formatear_k_particion(
        self,
        grupos_futuros: list[tuple[int, ...]],
        indices_presentes: list[int],
    ) -> str:
        """
        Produce una representación legible de la k-partición encontrada,
        usando letras del abecedario consistentes con el resto del proyecto.

        Formato por parte:
            |Futuros_i|
            |Presentes|

        Args:
        ----
            grupos_futuros: grupos de índices de variables futuras.
            indices_presentes: índices de variables presentes.

        Returns:
        -------
            str: representación formateada de la k-partición.
        """
        from src.funcs.base import ABECEDARY, LOWER_ABECEDARY
        from src.constants.base import VOID_STR

        linea_top = ""
        linea_bot = ""

        arr_presentes = self.sia_subsistema.dims_ncubos[indices_presentes]

        for grupo in grupos_futuros:
            arr_futuros_i = self.sia_subsistema.indices_ncubos[list(grupo)]
            str_fut = (
                ",".join(ABECEDARY[f] for f in arr_futuros_i)
                if arr_futuros_i.size else VOID_STR
            )
            str_pre = (
                ",".join(LOWER_ABECEDARY[p] for p in arr_presentes)
                if arr_presentes.size else VOID_STR
            )
            ancho = max(len(str_fut), len(str_pre)) + 2
            linea_top += f"|{str_fut:^{ancho}}|"
            linea_bot += f"|{str_pre:^{ancho}}|"

        return f"{linea_top}\n{linea_bot}\n"
