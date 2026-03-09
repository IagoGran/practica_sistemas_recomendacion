import gzip
from typing import List, Dict, Set

# ----------------------------
# Submission writer
# ----------------------------

def write_submission_csv(results: Dict[int, List[str]],
                        out_csv_path: str,
                        team_name: str,
                        email: str,
                        add_spaces: bool = True,
                        sort_pids: bool = True):
    """
    Formato tipo sample_submission.csv
    team_info,Team,Email
    pid, track1, ..., track500
    """
    sep = ", " if add_spaces else ","

    pids = sorted(results.keys()) if sort_pids else list(results.keys())

    with open(out_csv_path, "w", encoding="utf-8") as f:
        f.write(f"team_info{sep}{team_name}{sep}{email}\n\n")
        for pid in pids:
            recs = results[pid]
            if len(recs) != 500:
                raise ValueError(f"PID {pid}: esperado 500 recomendaciones, generado {len(recs)}")
            if len(set(recs)) != 500:
                raise ValueError(f"PID {pid}: hay duplicados en las recomendaciones")
            f.write(str(pid) + sep + sep.join(recs) + "\n")

def gzip_file(in_path: str, out_path: str):
    with open(in_path, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
        f_out.writelines(f_in)