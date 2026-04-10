(SR_venv) PS C:\Universidad\4_ano\segundo_cuatri\sistemas_recomendacion\practica_sistemas_recomendacion> & c:\Universidad\4_ano\segundo_cuatri\sistemas_recomendacion\practica_sistemas_recomendacion\SR_venv\Scripts\python.exe c:/Universidad/4_ano/segundo_cuatri/sistemas_recomendacion/practica_sistemas_recomendacion/code/run_iteration2_puresvd.py
Playlists test: 10000

======================================================================
EJECUTANDO PURESVD - VARIANTE A (fit train + test)
======================================================================
INFO: Procesando train_dir para construir la matriz...
INFO: Train_dir procesado, matriz parcial construida.
INFO: Procesando test_dir para anadir playlists de test a la matriz...
INFO: Test_dir procesado, matriz completa construida.
INFO: Matriz construida con shape (1000000, 2262292) y 65048209 elementos no nulos.
[A] Construccion de matriz: 159.41s
[A] Shape matriz conjunta: (1000000, 2262292)
[A] NNZ: 65048209
[A] Entrenamiento: 111.92s
[A] Reordenando playlists de test segun test_input_file...
[A] Construyendo vista del modelo restringida a las filas de test...
[A] Lanzando recomendacion paralela...
[PureSVD-A] chunks: 40 | chunk_size: 250 | workers: 6 | backend: thread | item_block_size: 50000
[PureSVD-A] 1/40 | 250/10000 playlists | 15.55s transcurridos | 10.11 min restantes
[PureSVD-A] 2/40 | 500/10000 playlists | 15.64s transcurridos | 4.95 min restantes
[PureSVD-A] 3/40 | 750/10000 playlists | 15.89s transcurridos | 3.27 min restantes
[PureSVD-A] 4/40 | 1000/10000 playlists | 16.00s transcurridos | 2.40 min restantes
[PureSVD-A] 5/40 | 1250/10000 playlists | 16.11s transcurridos | 1.88 min restantes
[PureSVD-A] 6/40 | 1500/10000 playlists | 16.90s transcurridos | 1.60 min restantes
[PureSVD-A] 7/40 | 1750/10000 playlists | 30.84s transcurridos | 2.42 min restantes
[PureSVD-A] 8/40 | 2000/10000 playlists | 31.05s transcurridos | 2.07 min restantes
[PureSVD-A] 9/40 | 2250/10000 playlists | 31.28s transcurridos | 1.80 min restantes
[PureSVD-A] 10/40 | 2500/10000 playlists | 31.55s transcurridos | 1.58 min restantes
[PureSVD-A] 11/40 | 2750/10000 playlists | 31.77s transcurridos | 1.40 min restantes
[PureSVD-A] 12/40 | 3000/10000 playlists | 32.56s transcurridos | 1.27 min restantes
[PureSVD-A] 13/40 | 3250/10000 playlists | 44.34s transcurridos | 1.53 min restantes
[PureSVD-A] 14/40 | 3500/10000 playlists | 46.60s transcurridos | 1.44 min restantes
[PureSVD-A] 15/40 | 3750/10000 playlists | 46.69s transcurridos | 1.30 min restantes
[PureSVD-A] 16/40 | 4000/10000 playlists | 46.93s transcurridos | 1.17 min restantes
[PureSVD-A] 17/40 | 4250/10000 playlists | 47.30s transcurridos | 1.07 min restantes
[PureSVD-A] 18/40 | 4500/10000 playlists | 47.76s transcurridos | 0.97 min restantes
[PureSVD-A] 19/40 | 4750/10000 playlists | 58.68s transcurridos | 1.08 min restantes
[PureSVD-A] 20/40 | 5000/10000 playlists | 61.41s transcurridos | 1.02 min restantes
[PureSVD-A] 21/40 | 5250/10000 playlists | 61.71s transcurridos | 0.93 min restantes
[PureSVD-A] 22/40 | 5500/10000 playlists | 62.60s transcurridos | 0.85 min restantes
[PureSVD-A] 23/40 | 5750/10000 playlists | 62.94s transcurridos | 0.78 min restantes
[PureSVD-A] 24/40 | 6000/10000 playlists | 63.02s transcurridos | 0.70 min restantes
[PureSVD-A] 25/40 | 6250/10000 playlists | 73.36s transcurridos | 0.73 min restantes
[PureSVD-A] 26/40 | 6500/10000 playlists | 77.44s transcurridos | 0.69 min restantes
[PureSVD-A] 27/40 | 6750/10000 playlists | 77.68s transcurridos | 0.62 min restantes
[PureSVD-A] 28/40 | 7000/10000 playlists | 78.43s transcurridos | 0.56 min restantes
[PureSVD-A] 29/40 | 7250/10000 playlists | 78.80s transcurridos | 0.50 min restantes
[PureSVD-A] 30/40 | 7500/10000 playlists | 78.95s transcurridos | 0.44 min restantes
[PureSVD-A] 31/40 | 7750/10000 playlists | 88.33s transcurridos | 0.43 min restantes
[PureSVD-A] 32/40 | 8000/10000 playlists | 93.83s transcurridos | 0.39 min restantes
[PureSVD-A] 33/40 | 8250/10000 playlists | 93.92s transcurridos | 0.33 min restantes
[PureSVD-A] 34/40 | 8500/10000 playlists | 94.54s transcurridos | 0.28 min restantes
[PureSVD-A] 35/40 | 8750/10000 playlists | 94.70s transcurridos | 0.23 min restantes
[PureSVD-A] 36/40 | 9000/10000 playlists | 94.83s transcurridos | 0.18 min restantes
[PureSVD-A] 37/40 | 9250/10000 playlists | 99.93s transcurridos | 0.14 min restantes
[PureSVD-A] 38/40 | 9500/10000 playlists | 102.40s transcurridos | 0.09 min restantes
[PureSVD-A] 39/40 | 9750/10000 playlists | 102.46s transcurridos | 0.04 min restantes
[PureSVD-A] 40/40 | 10000/10000 playlists | 102.68s transcurridos | 0.00 min restantes
[A] Recomendacion: 102.69s

======================================================================
EJECUTANDO PURESVD - VARIANTE B (fit train + folding-in)
======================================================================
INFO: Procesando train_dir para construir la matriz...
INFO: Train_dir procesado, matriz parcial construida.
INFO: No se proporciono test_dir; solo se proceso train_dir.
INFO: Matriz construida con shape (990000, 2262292) y 64767209 elementos no nulos.
[B] Shape matriz train: (990000, 2262292)
[B] NNZ: 64767209
INFO: Procesando test_dir con vocabulario fijo de train...
INFO: Procesamiento de test_dir completado. Se han procesado 10000 playlists de test.
INFO: Matriz test construida con shape (10000, 2262292) y 281000 elementos no nulos.
[B] Construccion de matrices: 166.34s
[B] Shape matriz test fixed vocab: (10000, 2262292)
[B] NNZ test: 281000
[B] Entrenamiento: 111.27s
[B] Reordenando playlists de test segun test_input_file...
[B] Lanzando recomendacion paralela con folding-in...
[PureSVD-B] chunks: 40 | chunk_size: 250 | workers: 6 | backend: thread | item_block_size: 50000
[PureSVD-B] 1/40 | 250/10000 playlists | 12.74s transcurridos | 8.28 min restantes
[PureSVD-B] 2/40 | 500/10000 playlists | 12.78s transcurridos | 4.05 min restantes
[PureSVD-B] 3/40 | 750/10000 playlists | 12.82s transcurridos | 2.64 min restantes
[PureSVD-B] 4/40 | 1000/10000 playlists | 12.86s transcurridos | 1.93 min restantes
[PureSVD-B] 5/40 | 1250/10000 playlists | 15.35s transcurridos | 1.79 min restantes
[PureSVD-B] 6/40 | 1500/10000 playlists | 15.44s transcurridos | 1.46 min restantes
[PureSVD-B] 7/40 | 1750/10000 playlists | 26.07s transcurridos | 2.05 min restantes
[PureSVD-B] 8/40 | 2000/10000 playlists | 26.13s transcurridos | 1.74 min restantes
[PureSVD-B] 9/40 | 2250/10000 playlists | 26.18s transcurridos | 1.50 min restantes
[PureSVD-B] 10/40 | 2500/10000 playlists | 26.75s transcurridos | 1.34 min restantes
[PureSVD-B] 11/40 | 2750/10000 playlists | 30.14s transcurridos | 1.32 min restantes
[PureSVD-B] 12/40 | 3000/10000 playlists | 30.65s transcurridos | 1.19 min restantes
[PureSVD-B] 13/40 | 3250/10000 playlists | 39.64s transcurridos | 1.37 min restantes
[PureSVD-B] 14/40 | 3500/10000 playlists | 39.82s transcurridos | 1.23 min restantes
[PureSVD-B] 15/40 | 3750/10000 playlists | 39.89s transcurridos | 1.11 min restantes
[PureSVD-B] 16/40 | 4000/10000 playlists | 40.69s transcurridos | 1.02 min restantes
[PureSVD-B] 17/40 | 4250/10000 playlists | 44.77s transcurridos | 1.01 min restantes
[PureSVD-B] 18/40 | 4500/10000 playlists | 45.57s transcurridos | 0.93 min restantes
[PureSVD-B] 19/40 | 4750/10000 playlists | 52.97s transcurridos | 0.98 min restantes
[PureSVD-B] 20/40 | 5000/10000 playlists | 53.06s transcurridos | 0.88 min restantes
[PureSVD-B] 21/40 | 5250/10000 playlists | 53.37s transcurridos | 0.80 min restantes
[PureSVD-B] 22/40 | 5500/10000 playlists | 54.20s transcurridos | 0.74 min restantes
[PureSVD-B] 23/40 | 5750/10000 playlists | 60.56s transcurridos | 0.75 min restantes
[PureSVD-B] 24/40 | 6000/10000 playlists | 61.02s transcurridos | 0.68 min restantes
[PureSVD-B] 25/40 | 6250/10000 playlists | 65.70s transcurridos | 0.66 min restantes
[PureSVD-B] 26/40 | 6500/10000 playlists | 65.82s transcurridos | 0.59 min restantes
[PureSVD-B] 27/40 | 6750/10000 playlists | 66.11s transcurridos | 0.53 min restantes
[PureSVD-B] 28/40 | 7000/10000 playlists | 66.76s transcurridos | 0.48 min restantes
[PureSVD-B] 29/40 | 7250/10000 playlists | 72.61s transcurridos | 0.46 min restantes
[PureSVD-B] 30/40 | 7500/10000 playlists | 73.19s transcurridos | 0.41 min restantes
[PureSVD-B] 31/40 | 7750/10000 playlists | 76.24s transcurridos | 0.37 min restantes
[PureSVD-B] 32/40 | 8000/10000 playlists | 76.53s transcurridos | 0.32 min restantes
[PureSVD-B] 33/40 | 8250/10000 playlists | 76.86s transcurridos | 0.27 min restantes
[PureSVD-B] 34/40 | 8500/10000 playlists | 78.52s transcurridos | 0.23 min restantes
[PureSVD-B] 35/40 | 8750/10000 playlists | 88.51s transcurridos | 0.21 min restantes
[PureSVD-B] 36/40 | 9000/10000 playlists | 88.63s transcurridos | 0.16 min restantes
[PureSVD-B] 37/40 | 9250/10000 playlists | 89.79s transcurridos | 0.12 min restantes
[PureSVD-B] 38/40 | 9500/10000 playlists | 89.87s transcurridos | 0.08 min restantes
[PureSVD-B] 39/40 | 9750/10000 playlists | 89.97s transcurridos | 0.04 min restantes
[PureSVD-B] 40/40 | 10000/10000 playlists | 90.61s transcurridos | 0.00 min restantes
[B] Recomendacion: 90.61s

======================================================================
COMPARATIVA FINAL - ITERACION 2 PURESVD
======================================================================

PURESVD VARIANTE A (train + test)
Tiempo construccion     : 159.41s
Tiempo entrenamiento    : 111.92s
Tiempo recomendacion    : 102.69s
Tiempo total            : 374.03s
R-Precision             : 0.127532
NDCG@500                : 0.282845
Clicks                  : 7.773800

PURESVD VARIANTE B (train + folding-in)
Tiempo construccion     : 166.34s
Tiempo entrenamiento    : 111.27s
Tiempo recomendacion    : 90.61s
Tiempo total            : 368.23s
R-Precision             : 0.127588
NDCG@500                : 0.282955
Clicks                  : 7.774800

GANADOR
Por calidad general: PURESVD VARIANTE B

DIFERENCIAS
Delta R-Precision       : -0.000056
Delta NDCG@500          : -0.000110
Delta Clicks            : -0.001000
Delta Construccion      : -6.93s
Delta Entrenamiento     : 0.65s
Delta Recomendacion     : 12.07s
Delta Tiempo total      : 5.80s