from global_utils.playlist_preprocessing import load_playlists_from_file
from global_utils.submission_writer import gzip_file, write_submission_csv
from utils_2.evaluation.puresvd_eval import evaluate_results
from utils_2.runtime.puresvd_variants import run_variant_a, run_variant_b


def main() -> None:
    train_dir = r"data\spotify_train_dataset\data"
    test_dir = r"data\spotify_test_playlists\data"
    test_input_file = r"data\spotify_test_playlists\test_input_playlists.json"
    test_eval_file = r"data\spotify_test_playlists\test_eval_playlists.json"

    team_name = "Iago Grandal del Rio"
    email = "i.gdelrio@udc.es"

    num_factors = 128
    num_workers = 6
    chunk_size = 250
    top_k = 500
    min_playlist_length_train = 15
    max_playlist_length_train = 250
    item_block_size = 50_000
    parallel_backend = "thread"
    verbose = True

    input_playlists = load_playlists_from_file(test_input_file)
    print("Playlists test:", len(input_playlists))
    if verbose:
        print(
            "Filtro de playlists train para PureSVD:",
            f"min_len={min_playlist_length_train}",
            f"max_len={max_playlist_length_train}",
        )

    results_a, timings_a = run_variant_a(
        train_dir=train_dir,
        test_dir=test_dir,
        test_input_file=test_input_file,
        input_playlists=input_playlists,
        num_factors=num_factors,
        num_workers=num_workers,
        chunk_size=chunk_size,
        top_k=top_k,
        min_playlist_length_train=min_playlist_length_train,
        max_playlist_length_train=max_playlist_length_train,
        item_block_size=item_block_size,
        parallel_backend=parallel_backend,
        verbose=verbose,
    )
    metrics_a = evaluate_results(results_a, input_playlists, test_eval_file)

    write_submission_csv(
        results_a,
        "submission_puresvd_variant_a.csv",
        team_name,
        email,
        add_spaces=True,
        sort_pids=True,
    )
    gzip_file("submission_puresvd_variant_a.csv", "submission_puresvd_variant_a.csv.gz")

    results_b, timings_b = run_variant_b(
        train_dir=train_dir,
        test_dir=test_dir,
        test_input_file=test_input_file,
        input_playlists=input_playlists,
        num_factors=num_factors,
        num_workers=num_workers,
        chunk_size=chunk_size,
        top_k=top_k,
        min_playlist_length_train=min_playlist_length_train,
        max_playlist_length_train=max_playlist_length_train,
        item_block_size=item_block_size,
        parallel_backend=parallel_backend,
        verbose=verbose,
    )
    metrics_b = evaluate_results(results_b, input_playlists, test_eval_file)

    write_submission_csv(
        results_b,
        "submission_puresvd_variant_b.csv",
        team_name,
        email,
        add_spaces=True,
        sort_pids=True,
    )
    gzip_file("submission_puresvd_variant_b.csv", "submission_puresvd_variant_b.csv.gz")

    print("\n" + "=" * 70)
    print("COMPARATIVA FINAL - ITERACION 2 PURESVD")
    print("=" * 70)

    print("\nPURESVD VARIANTE A (train + test)")
    print(f"Tiempo construccion     : {timings_a['build_matrix']:.2f}s")
    print(f"Tiempo entrenamiento    : {timings_a['fit_model']:.2f}s")
    print(f"Tiempo recomendacion    : {timings_a['recommend']:.2f}s")
    print(f"Tiempo total            : {timings_a['total']:.2f}s")
    print(f"R-Precision             : {metrics_a['r_precision']:.6f}")
    print(f"NDCG@500                : {metrics_a['ndcg']:.6f}")
    print(f"Clicks                  : {metrics_a['clicks']:.6f}")

    print("\nPURESVD VARIANTE B (train + folding-in)")
    print(f"Tiempo construccion     : {timings_b['build_matrix']:.2f}s")
    print(f"Tiempo entrenamiento    : {timings_b['fit_model']:.2f}s")
    print(f"Tiempo recomendacion    : {timings_b['recommend']:.2f}s")
    print(f"Tiempo total            : {timings_b['total']:.2f}s")
    print(f"R-Precision             : {metrics_b['r_precision']:.6f}")
    print(f"NDCG@500                : {metrics_b['ndcg']:.6f}")
    print(f"Clicks                  : {metrics_b['clicks']:.6f}")

    print("\nGANADOR")
    if metrics_a["r_precision"] > metrics_b["r_precision"]:
        print("Por calidad general: PURESVD VARIANTE A")
    else:
        print("Por calidad general: PURESVD VARIANTE B")

    print("\nDIFERENCIAS")
    print(f"Delta R-Precision       : {metrics_a['r_precision'] - metrics_b['r_precision']:.6f}")
    print(f"Delta NDCG@500          : {metrics_a['ndcg'] - metrics_b['ndcg']:.6f}")
    print(f"Delta Clicks            : {metrics_a['clicks'] - metrics_b['clicks']:.6f}")
    print(f"Delta Construccion      : {timings_a['build_matrix'] - timings_b['build_matrix']:.2f}s")
    print(f"Delta Entrenamiento     : {timings_a['fit_model'] - timings_b['fit_model']:.2f}s")
    print(f"Delta Recomendacion     : {timings_a['recommend'] - timings_b['recommend']:.2f}s")
    print(f"Delta Tiempo total      : {timings_a['total'] - timings_b['total']:.2f}s")


if __name__ == "__main__":
    main()
