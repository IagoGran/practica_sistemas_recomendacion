def main():
    # Rutas (ajusta si quieres)
    train_dir = r"data\spotify_train_dataset\data"
    test_input_file = r"data\spotify_test_playlists\test_input_playlists.json"
    test_eval_file = r"data\spotify_test_playlists\test_eval_playlists.json"
    
    out_csv = "submission.csv"
    out_gz = "submission.csv.gz"

    team_name = "Iago Grandal del Río"
    email = "i.gdelrio@udc.es"