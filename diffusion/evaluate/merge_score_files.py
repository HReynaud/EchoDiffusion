import os
import argparse
import time
import pandas as pd
from datetime import datetime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="Path to config file.")
    parser.add_argument("--chunks", type=int, default=1, help="Number of diffusion steps.")
    parser.add_argument("--rand_ef", action="store_true", help="Randomize EF input.", default=False)
    parser.add_argument("--del_prev", action="store_true", help="Delete temp files if they already exist.", default=False)
    args = parser.parse_args()


    causal_name = "counterfactual" if args.rand_ef else "factual"
    save_folder = os.path.join(args.model, "eval")
    csv_save_folder = csv_save_folder = os.path.join(save_folder, f"{causal_name}_csvs")
    file_names = [os.path.join(csv_save_folder, f"scores_{causal_name}_{c}.csv") for c in range(args.chunks)]


    if args.del_prev:
        print("Renaming previous files...")
        for f in file_names:
            if os.path.exists(f):
                os.rename(f, f + ".old")

    print("Waiting for all files to exist...")
    all_exist =  False
    while not all_exist:
        time.sleep(10)
        all_exist = all([os.path.exists(f) for f in file_names])

    print("All files exist, fusing...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dfs = [pd.read_csv(f) for f in file_names]
    df = pd.concat(dfs)
    final_path = os.path.join(csv_save_folder, f"scores_{causal_name}_{timestamp}.csv")
    df.to_csv(final_path, index=False)

    print(f"{len(df)} rows saved to {final_path}")