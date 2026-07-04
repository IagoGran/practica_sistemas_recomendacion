# Sistema de recomendación para playlist continuation

Este repositorio contiene una implementación de un sistema de recomendación para la tarea de playlist continuation usando el Million Playlist Dataset de Spotify. El objetivo es completar playlists incompletas recomendando canciones relevantes a partir de distintas estrategias de recomendación, con evaluación offline basada en métricas clásicas de ranking.

## Objetivo del proyecto

El proyecto aborda el problema de recomendar canciones para seguir una playlist parcialmente conocida. Para ello se implementan distintos enfoques:

- un baseline basado en popularidad global,
- modelos basados en vecindarios entre playlists y entre tracks,
- y variantes de PureSVD para comparar rendimiento y eficiencia.

## Enfoques implementados

### 1. Baseline de popularidad
Se recomienda una lista de canciones populares globalmente, filtrando aquellas que ya aparecen en la seed de la playlist.

### 2. Modelo basado en vecindarios
Se construye una recomendación a partir de similitud entre playlists y/o entre tracks, teniendo en cuenta la información de la seed.

### 3. PureSVD
Se experimenta con variantes de factorización matricial de tipo SVD para generar recomendaciones más robustas y con mejor calidad de ranking.

## Métricas de evaluación

El proyecto evalúa las recomendaciones con métricas offline estándar:

- R-Precision
- NDCG@500
- Clicks

Estas métricas permiten comparar la calidad de cada modelo sobre playlists de evaluación.

## Estructura del repositorio

```text
.
├── code/
│   ├── baseline_code_matrix.py
│   ├── neighborhood-based_recommendation.py
│   ├── run_iteration2_puresvd.py
│   └── global_utils/
│       ├── evaluation.py
│       ├── playlist_preprocessing.py
│       └── submission_writer.py
├── data/
│   ├── sample_submission.csv
│   ├── test_eval_playlists.json
│   └── test_input_playlists.json
├── doc/
├── spotify_train_dataset/
└── README.md
```

## Requisitos

Se recomienda trabajar con Python 3.9 o superior y un entorno virtual. Las dependencias principales del proyecto están relacionadas con:

- NumPy
- SciPy
- JSON / procesamiento de datos

## ▶Cómo ejecutar

1. Activa el entorno virtual del proyecto:

```powershell
RS_venv\Scripts\activate
```

2. Ejecuta cualquiera de los scripts principales desde la raíz del repositorio:

```powershell
python code\baseline_code_matrix.py
python code\neighborhood-based_recommendation.py
python code\run_iteration2_puresvd.py
```

3. Los resultados se generan en ficheros CSV/GZIP en la raíz del proyecto.

> Si tu estructura de datos difiere de la esperada, ajusta las rutas de entrada en los scripts al inicio de cada archivo.

## Datos

El proyecto utiliza datos del Million Playlist Dataset y ficheros de evaluación incluidos en la carpeta de datos del repositorio. Para ejecutar los scripts correctamente, es necesario disponer de la información de entrenamiento y de los playlists de prueba en las rutas indicadas.

## Autor

Iago Grandal del Río

## Notas

Este repositorio está orientado a un proyecto académico y de experimentación en sistemas de recomendación. La finalidad principal es comparar modelos de recomendación sobre un problema realista de playlist continuation.
