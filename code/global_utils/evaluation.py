from math import log2
from typing import List, Set, Dict
from utils.playlist_processing import load_playlists_from_file

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
    gold_by_pid = {}
    for pl in eval_playlists:
        pid = pl["pid"]
        all_tracks = {tr["track_uri"] for tr in pl.get("tracks", []) if "track_uri" in tr}
        gold_by_pid[pid] = all_tracks
    return gold_by_pid
