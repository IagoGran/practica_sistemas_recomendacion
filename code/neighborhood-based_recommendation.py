from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
import time

import numpy as np

from utils_1.playlist_processing import (
    build_global_popularity,
    recommend_for_seed_playlist_fast
)

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


# =========================================================
# Globals worker playlist-playlist
# =========================================================
G_PP_TRACK_TO_IDX = None
G_PP_IDX_TO_TRACK = None
G_PP_POPULARITY_LIST = None
G_PP_X = None
G_PP_XT = None
G_PP_PLAYLIST_NORMS = None
G_PP_K = 500
G_PP_TOP_NEIGHBORS = 1000
G_PP_MAX_PLAYLIST_FREQ = 50000


def init_worker_playlist(
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
    global G_PP_TRACK_TO_IDX, G_PP_IDX_TO_TRACK, G_PP_POPULARITY_LIST
    global G_PP_X, G_PP_XT, G_PP_PLAYLIST_NORMS
    global G_PP_K, G_PP_TOP_NEIGHBORS, G_PP_MAX_PLAYLIST_FREQ

    G_PP_TRACK_TO_IDX = track_to_idx
    G_PP_IDX_TO_TRACK = idx_to_track
    G_PP_POPULARITY_LIST = popularity_list
    G_PP_X = X
    G_PP_XT = XT
    G_PP_PLAYLIST_NORMS = playlist_norms
    G_PP_K = k
    G_PP_TOP_NEIGHBORS = top_neighbors
    G_PP_MAX_PLAYLIST_FREQ = max_playlist_freq_per_seed_track


def process_playlist_chunk_playlist_model(chunk: List[dict]) -> Dict[int, List[str]]:
    results = {}

    for pl in chunk:
        playlist_id = pl["pid"]
        seed = {tr["track_uri"] for tr in pl.get("tracks", []) if "track_uri" in tr}

        recs = recommend_for_seed_playlist_fast(
            seed_tracks=seed,
            track_to_idx=G_PP_TRACK_TO_IDX,
            idx_to_track=G_PP_IDX_TO_TRACK,
            popularity_list=G_PP_POPULARITY_LIST,
            X=G_PP_X,
            XT=G_PP_XT,
            playlist_norms=G_PP_PLAYLIST_NORMS,
            k=G_PP_K,
            top_neighbors=G_PP_TOP_NEIGHBORS,
            max_playlist_freq_per_seed_track=G_PP_MAX_PLAYLIST_FREQ
        )

        if len(recs) != G_PP_K:
            raise ValueError(f"[playlist-playlist] pid {playlist_id}: esperado {G_PP_K}, generado {len(recs)}")
        if len(set(recs)) != G_PP_K:
            raise ValueError(f"[playlist-playlist] pid {playlist_id}: duplicados")
        if any(t in seed for t in recs):
            raise ValueError(f"[playlist-playlist] pid {playlist_id}: seed colado")

        results[playlist_id] = recs

    return results


# =========================================================
# Globals worker track-track recommend final
# =========================================================
G_TR_TRACK_TO_IDX = None
G_TR_IDX_TO_TRACK = None
G_TR_POPULARITY_LIST = None
G_TR_TRACK_FREQS = None
G_TR_PRECOMPUTED_NEIGHBORS = None
G_TR_N_PLAYLISTS = None
G_TR_K = 500
G_TR_USE_SEED_IDF_WEIGHT = True


def init_worker_track_recommend(
    track_to_idx,
    idx_to_track,
    popularity_list,
    track_freqs,
    precomputed_neighbors,
    n_playlists,
    k,
    use_seed_idf_weight
):
    global G_TR_TRACK_TO_IDX, G_TR_IDX_TO_TRACK, G_TR_POPULARITY_LIST
    global G_TR_TRACK_FREQS, G_TR_PRECOMPUTED_NEIGHBORS, G_TR_N_PLAYLISTS
    global G_TR_K, G_TR_USE_SEED_IDF_WEIGHT

    G_TR_TRACK_TO_IDX = track_to_idx
    G_TR_IDX_TO_TRACK = idx_to_track
    G_TR_POPULARITY_LIST = popularity_list
    G_TR_TRACK_FREQS = track_freqs
    G_TR_PRECOMPUTED_NEIGHBORS = precomputed_neighbors
    G_TR_N_PLAYLISTS = n_playlists
    G_TR_K = k
    G_TR_USE_SEED_IDF_WEIGHT = use_seed_idf_weight


def process_playlist_chunk_track_model(chunk: List[dict]) -> Dict[int, List[str]]:
    results = {}

    for pl in chunk:
        playlist_id = pl["pid"]
        seed = {tr["track_uri"] for tr in pl.get("tracks", []) if "track_uri" in tr}

        recs = recommend_from_precomputed_neighbors(
            seed_tracks=seed,
            track_to_idx=G_TR_TRACK_TO_IDX,
            idx_to_track=G_TR_IDX_TO_TRACK,
            popularity_list=G_TR_POPULARITY_LIST,
            precomputed_neighbors=G_TR_PRECOMPUTED_NEIGHBORS,
            track_freqs=G_TR_TRACK_FREQS,
            n_playlists=G_TR_N_PLAYLISTS,
            k=G_TR_K,
            use_seed_idf_weight=G_TR_USE_SEED_IDF_WEIGHT
        )

        if len(recs) != G_TR_K:
            raise ValueError(f"[track-track] pid {playlist_id}: esperado {G_TR_K}, generado {len(recs)}")
        if len(set(recs)) != G_TR_K:
            raise ValueError(f"[track-track] pid {playlist_id}: duplicados")
        if any(t in seed for t in recs):
            raise ValueError(f"[track-track] pid {playlist_id}: seed colado")

        results[playlist_id] = recs

    return results


# =========================================================
# Helpers generales
# =========================================================
def chunk_list(items: List, chunk_size: int) -> List[List]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def evaluate_results(
    results: Dict[int, List[str]],
    input_playlists: List[dict],
    test_eval_file: str
) -> Dict[str, float]:
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

    return {
        "r_precision": sum(rp_list) / len(rp_list) if rp_list else 0.0,
        "ndcg": sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0,
        "clicks": sum(clicks_list) / len(clicks_list) if clicks_list else 0.0,
    }


def run_playlist_playlist_model(
    X,
    XT,
    playlist_norms,
    track_to_idx,
    idx_to_track,
    popular_list,
    input_playlists: List[dict],
    num_workers: int = 4,
    chunk_size: int = 500,
    k: int = 500,
    top_neighbors: int = 1000,
    max_playlist_freq_per_seed_track: int = 50000
) -> Dict[int, List[str]]:
    chunks = chunk_list(input_playlists, chunk_size)
    print(f"[playlist-playlist] chunks: {len(chunks)} | chunk_size: {chunk_size} | workers: {num_workers}")

    results = {}
    t0 = time.time()
    completed_chunks = 0
    total_chunks = len(chunks)

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker_playlist,
        initargs=(
            track_to_idx,
            idx_to_track,
            popular_list,
            X,
            XT,
            playlist_norms,
            k,
            top_neighbors,
            max_playlist_freq_per_seed_track
        )
    ) as executor:
        futures = [executor.submit(process_playlist_chunk_playlist_model, chunk) for chunk in chunks]

        for future in as_completed(futures):
            partial = future.result()
            results.update(partial)

            completed_chunks += 1
            processed = min(completed_chunks * chunk_size, len(input_playlists))
            elapsed = time.time() - t0
            avg_per_playlist = elapsed / processed if processed > 0 else 0.0
            remaining = avg_per_playlist * (len(input_playlists) - processed)

            print(
                f"[playlist-playlist] {completed_chunks}/{total_chunks} | "
                f"{processed}/{len(input_playlists)} playlists | "
                f"{elapsed:.2f}s transcurridos | "
                f"{remaining/60:.2f} min restantes"
            )

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
    print(f"[track-track precompute] chunks: {len(seed_chunks)} | chunk_size: {chunk_size} | workers: {num_workers}")

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
        futures = [executor.submit(process_seed_chunk, chunk) for chunk in seed_chunks]

        for future in as_completed(futures):
            partial = future.result()
            precomputed_neighbors.update(partial)

            completed += 1
            processed = min(completed * chunk_size, len(unique_seed_track_idxs))
            elapsed = time.time() - start
            avg_per_seed = elapsed / processed if processed > 0 else 0.0
            remaining = avg_per_seed * (len(unique_seed_track_idxs) - processed)

            print(
                f"[track-track precompute] {completed}/{total} | "
                f"{processed}/{len(unique_seed_track_idxs)} seeds | "
                f"{elapsed:.2f}s transcurridos | "
                f"{remaining/60:.2f} min restantes"
            )

    return precomputed_neighbors


def run_track_track_model(
    X,
    track_freqs,
    track_to_idx,
    idx_to_track,
    popular_list,
    input_playlists: List[dict],
    precomputed_neighbors,
    num_workers: int = 4,
    chunk_size: int = 250,
    k: int = 500,
    use_seed_idf_weight: bool = True
) -> Dict[int, List[str]]:
    chunks = chunk_list(input_playlists, chunk_size)
    print(f"[track-track recommend] chunks: {len(chunks)} | chunk_size: {chunk_size} | workers: {num_workers}")

    results = {}
    t0 = time.time()
    completed_chunks = 0
    total_chunks = len(chunks)

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker_track_recommend,
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
        futures = [executor.submit(process_playlist_chunk_track_model, chunk) for chunk in chunks]

        for future in as_completed(futures):
            partial = future.result()
            results.update(partial)

            completed_chunks += 1
            processed = min(completed_chunks * chunk_size, len(input_playlists))
            elapsed = time.time() - t0
            avg_per_playlist = elapsed / processed if processed > 0 else 0.0
            remaining = avg_per_playlist * (len(input_playlists) - processed)

            print(
                f"[track-track recommend] {completed_chunks}/{total_chunks} | "
                f"{processed}/{len(input_playlists)} playlists | "
                f"{elapsed:.2f}s transcurridos | "
                f"{remaining/60:.2f} min restantes"
            )

    return results


def main():
    train_dir = r"data\spotify_train_dataset\data"
    test_input_file = r"data\spotify_test_playlists\test_input_playlists.json"
    test_eval_file = r"data\spotify_test_playlists\test_eval_playlists.json"

    team_name = "Iago Grandal del Río"
    email = "i.gdelrio@udc.es"

    # =====================================================
    # 1) Construcción única
    # =====================================================
    t_build = time.time()
    X, track_to_idx, idx_to_track, _, _, popularity_list = build_global_popularity(train_dir)
    XT = X.T.tocsr()
    playlist_norms = np.sqrt(X.sum(axis=1)).A1.astype(np.float32)
    track_freqs = compute_track_frequencies(X)
    popular_list = popularity_list[:10000]

    print("Shape matriz:", X.shape)
    print("NNZ:", X.nnz)
    print(f"Construcción completada en {time.time() - t_build:.2f}s")

    # =====================================================
    # 2) Carga test
    # =====================================================
    input_playlists = load_playlists_from_file(test_input_file)
    print("Playlists test:", len(input_playlists))

    # Descomenta para pruebas rápidas
    # input_playlists = input_playlists[:1000]

    # =====================================================
    # 3) Ejecutar playlist-playlist
    # =====================================================
    print("\n" + "=" * 70)
    print("EJECUTANDO MODELO PLAYLIST-PLAYLIST")
    print("=" * 70)

    t_pp = time.time()
    results_pp = run_playlist_playlist_model(
        X=X,
        XT=XT,
        playlist_norms=playlist_norms,
        track_to_idx=track_to_idx,
        idx_to_track=idx_to_track,
        popular_list=popular_list,
        input_playlists=input_playlists,
        num_workers=4,
        chunk_size=500,
        k=500,
        top_neighbors=1000,
        max_playlist_freq_per_seed_track=50000
    )
    time_pp = time.time() - t_pp

    metrics_pp = evaluate_results(results_pp, input_playlists, test_eval_file)

    write_submission_csv(
        results_pp,
        "submission_playlist_playlist.csv",
        team_name,
        email,
        add_spaces=True,
        sort_pids=True
    )
    gzip_file("submission_playlist_playlist.csv", "submission_playlist_playlist.csv.gz")

    # =====================================================
    # 4) Ejecutar track-track precomputed
    # =====================================================
    print("\n" + "=" * 70)
    print("EJECUTANDO MODELO TRACK-TRACK PRECOMPUTED")
    print("=" * 70)

    unique_seed_track_idxs = extract_unique_seed_track_idxs(
        input_playlists=input_playlists,
        track_to_idx=track_to_idx
    )
    print("Tracks seed únicos:", len(unique_seed_track_idxs))

    t_pre = time.time()
    precomputed_neighbors = precompute_track_neighbors_parallel(
        unique_seed_track_idxs=unique_seed_track_idxs,
        X=X,
        XT=XT,
        track_freqs=track_freqs,
        max_playlists_per_seed_track=2000,
        min_cooccurrence=2,
        min_similarity=0.01,
        top_k_per_seed_track=100,
        num_workers=6,
        chunk_size=1000
    )
    time_pre = time.time() - t_pre

    t_tr = time.time()
    results_tr = run_track_track_model(
        X=X,
        track_freqs=track_freqs,
        track_to_idx=track_to_idx,
        idx_to_track=idx_to_track,
        popular_list=popular_list,
        input_playlists=input_playlists,
        precomputed_neighbors=precomputed_neighbors,
        num_workers=4,
        chunk_size=250,
        k=500,
        use_seed_idf_weight=True
    )
    time_tr_recommend = time.time() - t_tr
    time_tr_total = time_pre + time_tr_recommend

    metrics_tr = evaluate_results(results_tr, input_playlists, test_eval_file)

    write_submission_csv(
        results_tr,
        "submission_track_track_precomputed.csv",
        team_name,
        email,
        add_spaces=True,
        sort_pids=True
    )
    gzip_file("submission_track_track_precomputed.csv", "submission_track_track_precomputed.csv.gz")

    # =====================================================
    # 5) Comparativa final
    # =====================================================
    print("\n" + "=" * 70)
    print("COMPARATIVA FINAL")
    print("=" * 70)

    print("\nPLAYLIST-PLAYLIST")
    print(f"Tiempo modelo           : {time_pp:.2f}s")
    print(f"R-Precision             : {metrics_pp['r_precision']:.6f}")
    print(f"NDCG@500                : {metrics_pp['ndcg']:.6f}")
    print(f"Clicks                  : {metrics_pp['clicks']:.6f}")

    print("\nTRACK-TRACK PRECOMPUTED")
    print(f"Tiempo precompute       : {time_pre:.2f}s")
    print(f"Tiempo recommend        : {time_tr_recommend:.2f}s")
    print(f"Tiempo total modelo     : {time_tr_total:.2f}s")
    print(f"R-Precision             : {metrics_tr['r_precision']:.6f}")
    print(f"NDCG@500                : {metrics_tr['ndcg']:.6f}")
    print(f"Clicks                  : {metrics_tr['clicks']:.6f}")

    print("\nGANADOR")
    if metrics_pp["r_precision"] > metrics_tr["r_precision"]:
        print("Por calidad general: PLAYLIST-PLAYLIST")
    else:
        print("Por calidad general: TRACK-TRACK PRECOMPUTED")

    print("\nDIFERENCIAS")
    print(f"Δ R-Precision           : {metrics_pp['r_precision'] - metrics_tr['r_precision']:.6f}")
    print(f"Δ NDCG@500              : {metrics_pp['ndcg'] - metrics_tr['ndcg']:.6f}")
    print(f"Δ Clicks                : {metrics_pp['clicks'] - metrics_tr['clicks']:.6f}")
    print(f"Δ Tiempo total          : {time_pp - time_tr_total:.2f}s")


if __name__ == "__main__":
    main()