from typing import Dict, List

from global_utils.evaluation import (
    build_gold_from_eval_playlists,
    ndcg_at_k,
    r_precision,
    recommended_songs_clicks,
)


def evaluate_results(
    results: Dict[int, List[str]],
    input_playlists: List[dict],
    test_eval_file: str,
) -> Dict[str, float]:
    """
    Evaluate recommendation results for the ordered test playlists.

    Parameters
    ----------
    results:
        Mapping from playlist id to the ordered recommendation list.
    input_playlists:
        Ordered test playlists as they were queried.
    test_eval_file:
        JSON file containing the full evaluation playlists.

    Returns
    -------
    Dict[str, float]
        Mean values of ``r_precision``, ``ndcg`` and ``clicks``.
    """
    gold_all = build_gold_from_eval_playlists(test_eval_file)

    seed_by_pid = {
        playlist["pid"]: {
            track["track_uri"]
            for track in playlist.get("tracks", [])
            if "track_uri" in track
        }
        for playlist in input_playlists
    }

    rp_list = []
    ndcg_list = []
    clicks_list = []

    for pid, recommendations in results.items():
        all_eval_tracks = gold_all.get(pid, set())
        seed_tracks = seed_by_pid.get(pid, set())
        gold_tracks = all_eval_tracks - seed_tracks

        # This guards against accidental leakage of the visible seed tracks.
        if any(track in seed_tracks for track in recommendations):
            raise ValueError(
                f"[eval] pid {pid}: hay tracks seed colados en la recomendacion"
            )

        rp_list.append(r_precision(recommendations, gold_tracks))
        ndcg_list.append(ndcg_at_k(recommendations, gold_tracks, k=500))
        clicks_list.append(recommended_songs_clicks(recommendations, gold_tracks))

    return {
        "r_precision": sum(rp_list) / len(rp_list) if rp_list else 0.0,
        "ndcg": sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0,
        "clicks": sum(clicks_list) / len(clicks_list) if clicks_list else 0.0,
    }
