
from typing import List, Dict
from utils.playlist_processing import load_playlists_from_file, build_global_popularity, recommend_for_playlist
from utils.submission_writer import write_submission_csv, gzip_file
from utils.evaluation import r_precision, ndcg_at_k, recommended_songs_clicks, build_gold_from_eval_playlists

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
        pid = pl["pid"]
        seed = {tr["track_uri"] for tr in pl.get("tracks", []) if "track_uri" in tr}

        recs = recommend_for_playlist(seed, popular_list, k=500)

        # Validaciones básicas del submission
        if len(recs) != 500:
            raise ValueError(f"PID {pid}: esperado 500 recomendaciones, generado {len(recs)}. "
                            f"Sube most_common() o revisa datos.")
        if len(set(recs)) != 500:
            raise ValueError(f"PID {pid}: duplicados en recomendaciones (no debería pasar).")
        if any(t in seed for t in recs):
            raise ValueError(f"PID {pid}: se coló un seed en recomendaciones (no debería pasar).")

        results[pid] = recs

    # 4) Escribir submission + gzip
    write_submission_csv(results, out_csv, team_name, email, add_spaces=True, sort_pids=True)
    gzip_file(out_csv, out_gz)

    print(f"OK -> {out_csv} y {out_gz} generados. Playlists: {len(results)}")

    # 5) Evaluación offline
    # Construimos gold como: eval_tracks - seed_tracks (del input)
    gold_all = build_gold_from_eval_playlists(test_eval_file)

    # Mapa seed por pid desde input
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