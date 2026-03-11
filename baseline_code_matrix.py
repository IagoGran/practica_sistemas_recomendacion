import json
import os
from typing import List, Dict, Tuple, Set
import numpy as np
from scipy.sparse import csr_matrix
from math import log2
import gzip

def load_playlists_from_file(filepath: str) -> List[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("playlists", [])

def iter_playlists_from_dir(folder: str):
    for fn in os.listdir(folder):
        if fn.endswith(".json"):
            path = os.path.join(folder, fn)
            for pl in load_playlists_from_file(path):
                yield pl

def build_tracks_matrix(train_dir: str) -> Tuple[csr_matrix, Dict[str, int], Dict[int, int]]:
    """
    Matriz Playlist-Track:
      - filas = playlists
      - columnas = tracks
      - valor = 1 si el track aparece en la playlist
    Devuelve: (matrix, track_to_idx, playlist_id_to_row)
    """
    track_to_idx: Dict[str, int] = {}
    playlist_id_to_row: Dict[int, int] = {}

    rows = []
    cols = []
    values = []

    for pl in iter_playlists_from_dir(train_dir):
        playlist_id = pl.get("playlist_id")
        if playlist_id is None:
            continue

        # asigna índice de fila a cada playlist
        row = playlist_id_to_row.get(playlist_id)
        if row is None:
            row = len(playlist_id_to_row)
            playlist_id_to_row[playlist_id] = row

        # para evitar duplicados dentro de la misma playlist
        seen_in_playlist = set()

        for tr in pl.get("tracks", []):
            uri = tr.get("track_uri")
            if not uri or uri in seen_in_playlist:
                continue
            seen_in_playlist.add(uri)

            col = track_to_idx.get(uri)
            if col is None:
                col = len(track_to_idx)
                track_to_idx[uri] = col

            rows.append(row)
            cols.append(col)
            values.append(1)

    n_playlists = len(playlist_id_to_row)
    n_tracks = len(track_to_idx)

    X = csr_matrix(
        (np.array(values, dtype=np.uint8), (np.array(rows), np.array(cols))),
        shape=(n_playlists, n_tracks),
        dtype=np.uint8
    )
    X.sum_duplicates()  # por si acaso

    return X, track_to_idx, playlist_id_to_row

def popularity_from_matrix(X: csr_matrix, track_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Dada la matriz Playlist-Track, devuelve una lista de (track_uri, count) ordenada por count desc.
    """
    track_popularity = np.array(X.sum(axis=0)).flatten()  # suma por columnas
    idx_to_track = {idx: uri for uri, idx in track_to_idx.items()}
    popularity_list = [(idx_to_track[idx], count) for idx, count in enumerate(track_popularity)]
    # ordenamos por count desc
    popularity_list.sort(key=lambda x: x[1], reverse=True)
    return popularity_list

def build_global_popularity(train_dir: str) -> List[Tuple[str, int]]:
    X, track_to_idx, _ = build_tracks_matrix(train_dir)
    popularity_list = popularity_from_matrix(X, track_to_idx)
    return popularity_list

def recommend_for_playlist(seed: set, popularity_list: List[Tuple[str, int]], k: int = 500) -> List[str]:
    """
    Dada una playlist semilla (set de track_uri) y una lista de tracks ordenada por popularidad,
    devuelve las top-k recomendaciones (sin incluir los seeds).
    """
    recs = []
    for track_uri, _ in popularity_list:
        if track_uri not in seed:
            recs.append(track_uri)
        if len(recs) >= k:
            break
    return recs


# ----------------------------
# Submission writer
# ----------------------------

def write_submission_csv(results: Dict[int, List[str]],
                        out_csv_path: str,
                        team_name: str,
                        email: str,
                        add_spaces: bool = True,
                        sort_playlist_ids: bool = True):
    """
    Formato tipo sample_submission.csv
    team_info,Team,Email
    playlist_id, track1, ..., track500
    """
    sep = ", " if add_spaces else ","

    playlist_ids = sorted(results.keys()) if sort_playlist_ids else list(results.keys())

    with open(out_csv_path, "w", encoding="utf-8") as f:
        f.write(f"team_info{sep}{team_name}{sep}{email}\n\n")
        for playlist_id in playlist_ids:
            recs = results[playlist_id]
            if len(recs) != 500:
                raise ValueError(f"playlist_id {playlist_id}: esperado 500 recomendaciones, generado {len(recs)}")
            if len(set(recs)) != 500:
                raise ValueError(f"playlist_id {playlist_id}: hay duplicados en las recomendaciones")
            f.write(str(playlist_id) + sep + sep.join(recs) + "\n")

def gzip_file(in_path: str, out_path: str):
    with open(in_path, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
        f_out.writelines(f_in)


# ----------------------------
# Evaluation (offline, usando test_eval as gold)
# ----------------------------

def r_precision(recs: List[str], gold: Set[str]) -> float:
    """
    R-Precision: precisión en el top-R, donde R = |gold|.
    """
    R = len(gold)
    if R == 0:
        return 0.0
    topR = recs[:R]
    hits = sum(1 for t in topR if t in gold)
    return hits / R

def ndcg_at_k(recs: List[str], gold: Set[str], k: int = 500) -> float:
    """
    NDCG@k binario (relevancia 1 si está en gold).
    DCG = sum(rel_i / log2(i+2))
    IDCG = ideal con min(|gold|, k) unos.
    """
    k = min(k, len(recs))
    if not gold or k == 0:
        return 0.0

    dcg = 0.0
    for i in range(k):
        if recs[i] in gold:
            dcg += 1.0 / log2(i + 2)

    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0

def recommended_songs_clicks(recs: List[str], gold: Set[str]) -> float:
    """
    "Recommended Songs clicks" (como en el challenge):
    - Se asume que el usuario mira la lista en páginas de 10.
    - Devuelve el número de "clicks" hasta encontrar el primer relevante.
    clicks = floor(rank_first_relevant / 10)
    - Si no hay relevante en 500, clicks = 51 (convención típica MPD).
    """
    for idx, t in enumerate(recs):
        if t in gold:
            return idx // 10
    return 51.0

def build_gold_from_eval_playlists(test_eval_file: str) -> Dict[int, Set[str]]:
    """
    En tu dataset modificado, test_eval_playlists.json contiene la lista completa
    (seed + holdouts). Para evaluar necesitamos el gold (holdouts).

    Como el input puede venir con 0 tracks (title-only), usamos:
    gold = tracks completos del eval (toda la playlist) MINUS seed_del_input.
    """
    eval_playlists = load_playlists_from_file(test_eval_file)
    gold_by_playlist_id = {}
    for pl in eval_playlists:
        playlist_id = pl["playlist_id"]
        all_tracks = {tr["track_uri"] for tr in pl.get("tracks", []) if "track_uri" in tr}
        gold_by_playlist_id[playlist_id] = all_tracks
    return gold_by_playlist_id


# ----------------------------
# Main pipeline
# ----------------------------

def main():
    # Rutas (ajusta si quieres)
    train_dir = r"data\spotify_train_dataset\data"
    test_input_file = r"data\spotify_test_playlists\test_input_playlists.json"
    test_eval_file = r"data\spotify_test_playlists\test_eval_playlists.json"

    out_csv = "submission.csv"
    out_gz = "submission.csv.gz"

    team_name = "Iago Grandal del Río"
    email = "i.gdelrio@udc.es"

    # 1) Popularidad global
    popularity_list = build_global_popularity(train_dir)

    # 2) Lista popular
    popular_list = popularity_list[:10000]

    # 3) Generar recomendaciones SOLO para el input incompleto
    input_playlists = load_playlists_from_file(test_input_file)
    results: Dict[int, List[str]] = {}

    for pl in input_playlists:
        playlist_id = pl["playlist_id"]
        seed = {tr["track_uri"] for tr in pl.get("tracks", []) if "track_uri" in tr}

        recs = recommend_for_playlist(seed, popular_list, k=500)

        # Validaciones básicas del submission
        if len(recs) != 500:
            raise ValueError(f"playlist_id {playlist_id}: esperado 500 recomendaciones, generado {len(recs)}. "
                            f"Sube most_common() o revisa datos.")
        if len(set(recs)) != 500:
            raise ValueError(f"playlist_id {playlist_id}: duplicados en recomendaciones (no debería pasar).")
        if any(t in seed for t in recs):
            raise ValueError(f"playlist_id {playlist_id}: se coló un seed en recomendaciones (no debería pasar).")

        results[playlist_id] = recs

    # 4) Escribir submission + gzip
    write_submission_csv(results, out_csv, team_name, email, add_spaces=True, sort_playlist_ids=True)
    gzip_file(out_csv, out_gz)

    print(f"OK -> {out_csv} y {out_gz} generados. Playlists: {len(results)}")

    # 5) Evaluación offline
    # Construimos gold como: eval_tracks - seed_tracks (del input)
    gold_all = build_gold_from_eval_playlists(test_eval_file)

    # Mapa seed por playlist_id desde input
    seed_by_playlist_id = {
        pl["playlist_id"]: {tr["track_uri"] for tr in pl.get("tracks", []) if "track_uri" in tr}
        for pl in input_playlists
    }

    rp_list = []
    ndcg_list = []
    clicks_list = []

    for playlist_id, recs in results.items():
        all_eval_tracks = gold_all.get(playlist_id, set())
        seed = seed_by_playlist_id.get(playlist_id, set())
        gold = all_eval_tracks - seed  # holdouts esperados

        rp_list.append(r_precision(recs, gold))
        ndcg_list.append(ndcg_at_k(recs, gold, k=500))
        clicks_list.append(recommended_songs_clicks(recs, gold))

    avg_rp = sum(rp_list) / len(rp_list) if rp_list else 0.0
    avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    avg_clicks = sum(clicks_list) / len(clicks_list) if clicks_list else 0.0

    print("=== Offline evaluation (baseline popularidad) ===")
    print(f"R-Precision: {avg_rp:.6f}")
    print(f"NDCG@500  : {avg_ndcg:.6f}")
    print(f"Clicks    : {avg_clicks:.6f}")


if __name__ == "__main__":
    main()