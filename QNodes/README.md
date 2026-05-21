# QNodes

Este directorio contiene la base clasica del proyecto para el analisis de MIP/IIT. Incluye el punto de entrada de ejecucion, la implementacion principal en `src/`, las redes de ejemplo y la documentacion tecnica auxiliar.

## Contenido principal

- `exec.py`: punto de entrada del aplicativo.
- `src/`: codigo fuente con controladores, estrategias, modelos, utilidades y middlewares.
- `src/.samples/`: redes de ejemplo utilizadas por la ejecucion.
- `.docs/`: documentacion interna, guias, soluciones y materiales de apoyo.
- `tests/`: pruebas del proyecto.
- `pyproject.toml`: definicion del proyecto y dependencias.

## Estructura

```text
QNodes/
├── exec.py
├── pyproject.toml
├── pyphi_config.yml
├── LICENSE
├── .docs/
│   ├── application.md
│   ├── Generalidades_Proyecto_ADA24B___V1_2_0.pdf
│   ├── Taller_Final.pdf
│   ├── .diagrams/
│   ├── .errors/
│   ├── .solutions/
│   ├── .strategies/
│   └── .study/
├── src/
│   ├── main.py
│   ├── constants/
│   ├── controllers/
│   ├── funcs/
│   ├── middlewares/
│   ├── models/
│   └── strategies/
└── tests/
```

## Ejecucion

Desde la carpeta `QNodes/`:

```bash
uv sync
uv run exec.py
```

El archivo `exec.py` activa el profiling, selecciona una red de muestra y delega la ejecucion en `src/main.py`.

## Flujo de trabajo

La configuracion inicial se define en `src/main.py`, donde se establecen el estado inicial y los patrones binarios que usa la estrategia `BruteForce`.

La clase `Application` en `src/models/base/application.py` centraliza parametros globales como:

- semilla aleatoria
- pagina de la red de muestra
- distancia metrica
- notacion de indexado
- tiempo de EMD
- activacion del profiling

## Documentacion interna

La carpeta `.docs/` contiene material de uso y diagnostico. En particular, `application.md` explica:

- como ejecutar el programa
- como usar redes de muestra
- como aplicar estrategias puntuales
- como interpretar los resultados de profiling y logs
- recomendaciones cuando PyPhi requiere dependencias de compilacion

## Notas

- Las redes de ejemplo se cargan desde `src/.samples/`.
- Si PyPhi falla por compilacion nativa, puede ser necesario instalar Microsoft C++ Build Tools.
- Para comprender la ejecucion general del proyecto, conviene revisar primero `exec.py` y luego `src/main.py`.
