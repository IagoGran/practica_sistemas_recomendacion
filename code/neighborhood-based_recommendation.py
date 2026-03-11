from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict

import numpy as np

from utils_1.playlist_processing import recommend_for_seed_playlist_fast, build_global_popularity
from global_utils.evaluation import r_precision, ndcg_at_k, recommended_songs_clicks, load_playlists_from_file, build_gold_from_eval_playlists
from global_utils.submission_writer import write_submission_csv, gzip_file

# Globals para workers
G_TRACK_TO_IDX = None
G_IDX_TO_TRACK = None
G_POPULARITY_LIST = None
G_X = None
G_XT = None
G_PLAYLIST_NORMS = None
G_K = 500
G_TOP_NEIGHBORS = 1000
G_MAX_PLAYLIST_FREQ = 50000

def init_worker(
    track_to_idx,
    idx_to_track,
    popularity_list,
    X,
    XT,
    playlist_norms,
    k,
    top_neighbors,
    max_playlist_freq_per_seed_track
):
    global G_TRACK_TO_IDX, G_IDX_TO_TRACK, G_POPULARITY_LIST
    global G_X, G_XT, G_PLAYLIST_NORMS
    global G_K, G_TOP_NEIGHBORS, G_MAX_PLAYLIST_FREQ

    G_TRACK_TO_IDX = track_to_idx
    G_IDX_TO_TRACK = idx_to_track
    G_POPULARITY_LIST = popularity_list
    G_X = X
    G_XT = XT
    G_PLAYLIST_NORMS = playlist_norms
    G_K = k
    G_TOP_NEIGHBORS = top_neighbors
    G_MAX_PLAYLIST_FREQ = max_playlist_freq_per_seed_track

def chunk_list(items: List[dict], chunk_size: int) -> List[List[dict]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def process_playlist_chunk(chunk: List[dict]) -> Dict[int, List[str]]:
    results = {}

    for pl in chunk:
        playlist_id = pl["pid"]
        seed = {tr["track_uri"] for tr in pl.get("tracks", []) if "track_uri" in tr}

        recs = recommend_for_seed_playlist_fast(
            seed_tracks=seed,
            track_to_idx=G_TRACK_TO_IDX,
            idx_to_track=G_IDX_TO_TRACK,
            popularity_list=G_POPULARITY_LIST,
            X=G_X,
            XT=G_XT,
            playlist_norms=G_PLAYLIST_NORMS,
            k=G_K,
            top_neighbors=G_TOP_NEIGHBORS,
            max_playlist_freq_per_seed_track=G_MAX_PLAYLIST_FREQ
        )

        # Validaciones
        if len(recs) != G_K:
            raise ValueError(
                f"pid {playlist_id}: esperado {G_K} recomendaciones, generado {len(recs)}."
            )
        if len(set(recs)) != G_K:
            raise ValueError(
                f"pid {playlist_id}: duplicados en recomendaciones."
            )
        if any(t in seed for t in recs):
            raise ValueError(
                f"pid {playlist_id}: se coló un seed en recomendaciones."
            )

        results[playlist_id] = recs

    return results

def main():
    import time

    train_dir = r"data\spotify_train_dataset\data"
    test_input_file = r"data\spotify_test_playlists\test_input_playlists.json"
    test_eval_file = r"data\spotify_test_playlists\test_eval_playlists.json"

    out_csv = "submission.csv"
    out_gz = "submission.csv.gz"

    team_name = "Iago Grandal del Río"
    email = "i.gdelrio@udc.es"

    t_build = time.time()
    X, track_to_idx, idx_to_track, playlist_id_to_row, row_to_playlist_id, popularity_list = build_global_popularity(train_dir)

    print("Shape matriz:", X.shape)
    print("NNZ:", X.nnz)

    XT = X.T.tocsr()
    playlist_norms = np.sqrt(X.sum(axis=1)).A1.astype(np.float32)

    print(f"Construcción/preparación completada en {time.time() - t_build:.2f}s")

    popular_list = popularity_list[:10000]

    input_playlists = load_playlists_from_file(test_input_file)
    print("Playlists test:", len(input_playlists))

    # Ajusta estos parámetros
    num_workers = 4
    chunk_size = 250

    chunks = chunk_list(input_playlists, chunk_size)
    print(f"Chunks: {len(chunks)} | chunk_size: {chunk_size} | workers: {num_workers}")

    results = {}
    t0 = time.time()
    completed_chunks = 0
    total_chunks = len(chunks)

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(
            track_to_idx,
            idx_to_track,
            popular_list,
            X,
            XT,
            playlist_norms,
            500,      # k
            1000,     # top_neighbors
            50000     # max_playlist_freq_per_seed_track
        )
    ) as executor:

        futures = [executor.submit(process_playlist_chunk, chunk) for chunk in chunks]

        for future in as_completed(futures):
            partial = future.result()
            results.update(partial)

            completed_chunks += 1
            processed = min(completed_chunks * chunk_size, len(input_playlists))
            elapsed = time.time() - t0
            avg_per_playlist = elapsed / processed if processed > 0 else 0.0
            remaining = avg_per_playlist * (len(input_playlists) - processed)

            print(
                f"Chunks {completed_chunks}/{total_chunks} | "
                f"playlists aprox: {processed}/{len(input_playlists)} | "
                f"transcurrido: {elapsed:.2f}s | "
                f"media: {avg_per_playlist:.4f}s/playlist | "
                f"restante estimado: {remaining/60:.2f} min"
            )

    # Submission
    write_submission_csv(results, out_csv, team_name, email, add_spaces=True, sort_pids=True)
    gzip_file(out_csv, out_gz)

    print(f"OK -> {out_csv} y {out_gz} generados. Playlists: {len(results)}")

    # Evaluación offline
    gold_all = build_gold_from_eval_playlists(test_eval_file)

    seed_by_pid = {
        pl["pid"]: {tr["track_uri"] for tr in pl.get("tracks", []) if "track_uri" in tr}
        for pl in input_playlists
    }

    rp_list = []
    ndcg_list = []
    clicks_list = []

    for pid, recs in results.items():
        all_eval_tracks = gold_all.get(pid, set())
        seed = seed_by_pid.get(pid, set())
        gold = all_eval_tracks - seed

        rp_list.append(r_precision(recs, gold))
        ndcg_list.append(ndcg_at_k(recs, gold, k=500))
        clicks_list.append(recommended_songs_clicks(recs, gold))

    avg_rp = sum(rp_list) / len(rp_list) if rp_list else 0.0
    avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    avg_clicks = sum(clicks_list) / len(clicks_list) if clicks_list else 0.0

    print("=== Offline evaluation ===")
    print(f"R-Precision: {avg_rp:.6f}")
    print(f"NDCG@500  : {avg_ndcg:.6f}")
    print(f"Clicks    : {avg_clicks:.6f}")

if __name__ == "__main__":
    main()