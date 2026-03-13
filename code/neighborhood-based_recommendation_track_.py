from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict
import time

from utils_1.playlist_processing import build_global_popularity
from utils_1.track_processing import (
    compute_track_frequencies,
    extract_unique_seed_track_idxs,
    init_worker_precompute,
    process_seed_chunk,
    recommend_from_precomputed_neighbors
)

from global_utils.playlist_preprocessing import load_playlists_from_file
from global_utils.evaluation import (
    r_precision,
    ndcg_at_k,
    recommended_songs_clicks,
    build_gold_from_eval_playlists
)
from global_utils.submission_writer import write_submission_csv, gzip_file


# ===============================
# Globals para recomendación final
# ===============================

G_TRACK_TO_IDX = None
G_IDX_TO_TRACK = None
G_POPULARITY_LIST = None
G_TRACK_FREQS = None
G_PRECOMPUTED_NEIGHBORS = None
G_N_PLAYLISTS = None
G_K = 500
G_USE_SEED_IDF_WEIGHT = True


def init_worker_recommend(
    track_to_idx,
    idx_to_track,
    popularity_list,
    track_freqs,
    precomputed_neighbors,
    n_playlists,
    k,
    use_seed_idf_weight
):
    global G_TRACK_TO_IDX, G_IDX_TO_TRACK, G_POPULARITY_LIST
    global G_TRACK_FREQS, G_PRECOMPUTED_NEIGHBORS, G_N_PLAYLISTS
    global G_K, G_USE_SEED_IDF_WEIGHT

    G_TRACK_TO_IDX = track_to_idx
    G_IDX_TO_TRACK = idx_to_track
    G_POPULARITY_LIST = popularity_list
    G_TRACK_FREQS = track_freqs
    G_PRECOMPUTED_NEIGHBORS = precomputed_neighbors
    G_N_PLAYLISTS = n_playlists
    G_K = k
    G_USE_SEED_IDF_WEIGHT = use_seed_idf_weight


def chunk_list(items: List, chunk_size: int) -> List[List]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def process_playlist_chunk(chunk: List[dict]) -> Dict[int, List[str]]:
    results = {}

    for pl in chunk:
        playlist_id = pl["pid"]
        seed = {
            tr["track_uri"]
            for tr in pl.get("tracks", [])
            if "track_uri" in tr
        }

        recs = recommend_from_precomputed_neighbors(
            seed_tracks=seed,
            track_to_idx=G_TRACK_TO_IDX,
            idx_to_track=G_IDX_TO_TRACK,
            popularity_list=G_POPULARITY_LIST,
            precomputed_neighbors=G_PRECOMPUTED_NEIGHBORS,
            track_freqs=G_TRACK_FREQS,
            n_playlists=G_N_PLAYLISTS,
            k=G_K,
            use_seed_idf_weight=G_USE_SEED_IDF_WEIGHT
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


def precompute_track_neighbors_parallel(
    unique_seed_track_idxs: List[int],
    X,
    XT,
    track_freqs,
    max_playlists_per_seed_track: int = 2000,
    min_cooccurrence: int = 2,
    min_similarity: float = 0.01,
    top_k_per_seed_track: int = 100,
    num_workers: int = 4,
    chunk_size: int = 1000
):
    seed_chunks = chunk_list(unique_seed_track_idxs, chunk_size)

    print(
        f"Precompute chunks: {len(seed_chunks)} | "
        f"chunk_size: {chunk_size} | workers: {num_workers}"
    )

    precomputed_neighbors = {}
    start = time.time()
    completed = 0
    total = len(seed_chunks)

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker_precompute,
        initargs=(
            X,
            XT,
            track_freqs,
            max_playlists_per_seed_track,
            min_cooccurrence,
            min_similarity,
            top_k_per_seed_track
        )
    ) as executor:

        futures = [
            executor.submit(process_seed_chunk, chunk)
            for chunk in seed_chunks
        ]

        for future in as_completed(futures):
            partial = future.result()
            precomputed_neighbors.update(partial)

            completed += 1
            processed = min(completed * chunk_size, len(unique_seed_track_idxs))
            elapsed = time.time() - start
            avg_per_seed = elapsed / processed if processed > 0 else 0.0
            remaining = avg_per_seed * (len(unique_seed_track_idxs) - processed)

            print(
                f"Precompute chunks {completed}/{total} | "
                f"seeds aprox: {processed}/{len(unique_seed_track_idxs)} | "
                f"transcurrido: {elapsed:.2f}s | "
                f"media: {avg_per_seed:.4f}s/seed | "
                f"restante estimado: {remaining/60:.2f} min"
            )

    return precomputed_neighbors


def main():
    train_dir = r"data\spotify_train_dataset\data"
    test_input_file = r"data\spotify_test_playlists\test_input_playlists.json"
    test_eval_file = r"data\spotify_test_playlists\test_eval_playlists.json"

    out_csv = "submission_track_track_precomputed.csv"
    out_gz = "submission_track_track_precomputed.csv.gz"

    team_name = "Iago Grandal del Río"
    email = "i.gdelrio@udc.es"

    # ===============================
    # 1) Construcción
    # ===============================
    t_build = time.time()

    X, track_to_idx, idx_to_track, playlist_id_to_row, row_to_playlist_id, popularity_list = build_global_popularity(train_dir)
    XT = X.T.tocsr()
    track_freqs = compute_track_frequencies(X)
    popular_list = popularity_list[:10000]

    print("Shape matriz:", X.shape)
    print("NNZ:", X.nnz)
    print(f"Construcción completada en {time.time() - t_build:.2f}s")

    # ===============================
    # 2) Carga test
    # ===============================
    input_playlists = load_playlists_from_file(test_input_file)
    print("Playlists test:", len(input_playlists))

    # Descomenta para pruebas rápidas
    # input_playlists = input_playlists[:1000]

    # ===============================
    # 3) Seeds únicos
    # ===============================
    unique_seed_track_idxs = extract_unique_seed_track_idxs(
        input_playlists=input_playlists,
        track_to_idx=track_to_idx
    )

    print("Tracks seed únicos:", len(unique_seed_track_idxs))

    # ===============================
    # 4) Precomputación paralela
    # ===============================
    t_precompute = time.time()

    precomputed_neighbors = precompute_track_neighbors_parallel(
        unique_seed_track_idxs=unique_seed_track_idxs,
        X=X,
        XT=XT,
        track_freqs=track_freqs,
        max_playlists_per_seed_track=2000,
        min_cooccurrence=2,
        min_similarity=0.01,
        top_k_per_seed_track=100,
        num_workers=2,          # prueba 2 primero
        chunk_size=1000
    )

    print(f"Precomputación completada en {time.time() - t_precompute:.2f}s")

    # ===============================
    # 5) Recomendación final
    # ===============================
    num_workers_recommend = 2
    chunk_size_recommend = 250
    k = 500
    use_seed_idf_weight = True

    chunks = chunk_list(input_playlists, chunk_size_recommend)

    print(
        f"Recommend chunks: {len(chunks)} | "
        f"chunk_size: {chunk_size_recommend} | "
        f"workers: {num_workers_recommend}"
    )

    results = {}
    t_recommend = time.time()
    completed_chunks = 0
    total_chunks = len(chunks)

    with ProcessPoolExecutor(
        max_workers=num_workers_recommend,
        initializer=init_worker_recommend,
        initargs=(
            track_to_idx,
            idx_to_track,
            popular_list,
            track_freqs,
            precomputed_neighbors,
            X.shape[0],
            k,
            use_seed_idf_weight
        )
    ) as executor:

        futures = [
            executor.submit(process_playlist_chunk, chunk)
            for chunk in chunks
        ]

        for future in as_completed(futures):
            partial = future.result()
            results.update(partial)

            completed_chunks += 1
            processed = min(completed_chunks * chunk_size_recommend, len(input_playlists))
            elapsed = time.time() - t_recommend
            avg_per_playlist = elapsed / processed if processed > 0 else 0.0
            remaining = avg_per_playlist * (len(input_playlists) - processed)

            print(
                f"Recommend chunks {completed_chunks}/{total_chunks} | "
                f"playlists aprox: {processed}/{len(input_playlists)} | "
                f"transcurrido: {elapsed:.2f}s | "
                f"media: {avg_per_playlist:.4f}s/playlist | "
                f"restante estimado: {remaining/60:.2f} min"
            )

    # ===============================
    # 6) Guardar submission
    # ===============================
    write_submission_csv(
        results,
        out_csv,
        team_name,
        email,
        add_spaces=True,
        sort_pids=True
    )
    gzip_file(out_csv, out_gz)

    print(f"OK -> {out_csv} y {out_gz} generados. Playlists: {len(results)}")

    # ===============================
    # 7) Evaluación offline
    # ===============================
    gold_all = build_gold_from_eval_playlists(test_eval_file)

    seed_by_pid = {
        pl["pid"]: {
            tr["track_uri"]
            for tr in pl.get("tracks", [])
            if "track_uri" in tr
        }
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

    print("=== Offline evaluation (track-track precomputed) ===")
    print(f"R-Precision: {avg_rp:.6f}")
    print(f"NDCG@500  : {avg_ndcg:.6f}")
    print(f"Clicks    : {avg_clicks:.6f}")


if __name__ == "__main__":
    main()