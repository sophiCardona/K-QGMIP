# GeoMIP

Este directorio agrupa los datos, el codigo fuente y los resultados asociados al proyecto GeoMIP.

## Contenido principal

- `data/`: matrices TPM de ejemplo y utilidades para generar nuevos datos.
- `src/`: implementacion del flujo principal del proyecto.
- `results/`: archivos de salida y resultados experimentales.
- `Dataset_Description.md`: descripcion del dataset y su interpretacion.

## Estructura

```text
GeoMIP/
├── data/
│   ├── creation.py
│   └── samples/
│       ├── N3A.csv
│       ├── N3B.csv
│       ├── N4A.csv
│       ├── N4B.csv
│       ├── N4C.csv
│       ├── N5A.csv
│       ├── N5B.csv
│       ├── N6A.csv
│       ├── N8A.csv
│       ├── N10A.csv
│       ├── N15A.csv
│       └── N15B.csv
├── src/
│   └── Method2_Dynamic_Programming_Reformulation/
│       ├── exec.py
│       ├── pyphi_config.yml
│       ├── pyproject.toml
│       ├── review/
│       │   └── profiling/
│       │       └── NET15A/
│       │           └── 19_03_2026/
│       │               ├── 11hrs/
│       │               │   └── aplicar_estrategia.html
│       │               └── 12hrs/
│       │                   └── aplicar_estrategia.html
│       ├── src/
│       │   ├── main.py
│       │   ├── constants/
│       │   │   ├── base.py
│       │   │   ├── error.py
│       │   │   └── models.py
│       │   ├── controllers/
│       │   │   ├── manager.py
│       │   │   └── strategies/
│       │   │       ├── force.py
│       │   │       ├── geometric.py
│       │   │       ├── phi.py
│       │   │       └── q_nodes.py
│       │   ├── funcs/
│       │   │   ├── base.py
│       │   │   ├── format.py
│       │   │   └── system.py
│       │   ├── middlewares/
│       │   │   ├── profile.py
│       │   │   └── slogger.py
│       │   ├── models/
│       │   │   ├── base/
│       │   │   │   ├── application.py
│       │   │   │   └── sia.py
│       │   │   ├── core/
│       │   │   │   ├── ncube.py
│       │   │   │   ├── solution.py
│       │   │   │   └── system.py
│       │   │   └── enums/
│       │   │       ├── distance.py
│       │   │       └── notation.py
│       │   └── video/
│       │       ├── hyper-v0.py
│       │       ├── hyper-v1.py
│       │       ├── hyper-v2.py
│       │       ├── hyper-v3.py
│       │       ├── hyper-v4.py
│       │       ├── hyper-v5.py
│       │       ├── hyper-v6.py
│       │       ├── hyper-v7.py
│       │       └── hyper-v8.py
└── results/
```

## Datos

Los archivos `data/samples/*.csv` contienen matrices TPM para redes de prueba de distintos tamanos.
El archivo `data/creation.py` sirve para generar nuevas redes o matrices sinteticas.

## Codigo fuente

La implementacion principal disponible en este arbol corresponde a `Method2_Dynamic_Programming_Reformulation`.
Desde `exec.py` se lanza la ejecucion del metodo y se conecta con el resto de modulos definidos en `src/`.

## Resultados y revision

La carpeta `results/` concentra salidas y archivos derivados del proceso experimental.
La ruta `review/profiling/` almacena reportes HTML de perfilado para casos concretos.

## Documentacion relacionada

- `Dataset_Description.md`: detalle del dataset, formatos y provenance.
- `../docs/`: documentos de apoyo del proyecto general.
