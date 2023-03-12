import os
import pandas as pd
import argparse
import numpy as np
from sklearn.metrics import r2_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to csv score file.")
    args = parser.parse_args()

    df = pd.read_csv(args.file)
    print(f"Loaded file {args.file.split('/')[-1]} with {df.shape[0]} rows.")
    df = df.sort_values(by=['FileName'])
    print(df.head(5))

    filename_0 = df["FileName"].values[0]
    num_repeat = df[df["FileName"]==filename_0].shape[0]

    print(f"Found {num_repeat} repeats per filename.")

    # get values
    gt_ef = df["GT EF"].values
    gen_ef = df["Gen EF"].values
    ref_ef = df["Ref EF"].values

    ssim = df["SSIM"].values

    lpips = df["LPIPS"].values

    print("Separated values.")

    # get best value per filename
    unique_gt_ef = gt_ef.reshape(-1, num_repeat).max(axis=1) # type: ignore

    tmp_best_idx = np.abs(gt_ef - gen_ef).reshape(-1, num_repeat).argmin(axis=1) # type: ignore type get index of lowest error
    best_gen_ef = gen_ef.reshape(-1, num_repeat)[np.arange(len(tmp_best_idx)),tmp_best_idx] # type: ignore get best value
    
    tmp_best_idx = np.abs(gt_ef - ref_ef).reshape(-1, num_repeat).argmin(axis=1) # type: ignore get index of lowest error
    best_ref_ef = ref_ef.reshape(-1, num_repeat)[np.arange(len(tmp_best_idx)),tmp_best_idx] # type: ignore get best value

    # get best ssim per filename
    best_ssim = ssim.reshape(-1, num_repeat).max(axis=1) # type: ignore 

    # get best lpips per filename
    best_lpips = lpips.reshape(-1, num_repeat).min(axis=1) # type: ignore

    print("Found best values.")

    ##################
    # compute scores #
    ##################
    
    # compute avg R2 scores
    r2_avg_gen = r2_score(gt_ef, gen_ef)
    r2_avg_ref = r2_score(gt_ef, ref_ef)

    # compute best R2 scores
    r2_best_gen = r2_score(unique_gt_ef, best_gen_ef)
    r2_best_ref = r2_score(unique_gt_ef, best_ref_ef)

    print("Computed R2 scores.")

    # compute avg MAE
    mae_avg_gen = np.mean(np.abs(gt_ef - gen_ef)) # type: ignore
    mae_avg_ref = np.mean(np.abs(gt_ef - ref_ef)) # type: ignore

    # compute best MAE
    mae_best_gen = np.mean(np.abs(unique_gt_ef - best_gen_ef))
    mae_best_ref = np.mean(np.abs(unique_gt_ef - best_ref_ef))

    print("Computed MAE scores.")

    # compute avg RMSE
    rmse_avg_gen = np.sqrt(np.mean((gt_ef - gen_ef)**2)) # type: ignore
    rmse_avg_ref = np.sqrt(np.mean((gt_ef - ref_ef)**2)) # type: ignore

    # compute best RMSE
    rmse_best_gen = np.sqrt(np.mean((unique_gt_ef - best_gen_ef)**2))
    rmse_best_ref = np.sqrt(np.mean((unique_gt_ef - best_ref_ef)**2))

    print("Computed RMSE scores.")

    # scores between ref and gen
    r2_gen_ref = r2_score(ref_ef, gen_ef)

    mae_gen_ref = np.mean(np.abs(ref_ef - gen_ef)) # type: ignore

    rmse_gen_ref = np.sqrt(np.mean((ref_ef - gen_ef)**2)) # type: ignore

    print("Computed scores between ref and gen.")

    # compute avg SSIM
    ssim_avg = np.mean(ssim) # type: ignore

    # compute best SSIM
    ssim_best = np.mean(best_ssim)

    print("Computed SSIM scores.")

    # compute avg LPIPS
    lpips_avg = np.mean(lpips) # type: ignore

    # compute best LPIPS
    lpips_best = np.mean(best_lpips)

    print("Computed LPIPS scores.")

    # Print all results in a formated table
    print(f"{'R2':<10}| {'GT/GEN':<10}{'GT/REF':<10}{'GEN/REF':<10}| {'BEST GEN':<10}{'BEST REF':<10}")
    print(f"{'- XSCM':<10}| {r2_avg_gen:<10.4f}{r2_avg_ref:<10.4f}{r2_gen_ref:<10.4f}| {r2_best_gen:<10.4f}{r2_best_ref:<10.4f}")
    
    print()

    print(f"{'MAE':<10}| {'GT/GEN':<10}{'GT/REF':<10}{'GEN/REF':<10}| {'BEST GEN':<10}{'BEST REF':<10}")
    print(f"{'- XSCM':<10}| {mae_avg_gen:<10.4f}{mae_avg_ref:<10.4f}{mae_gen_ref:<10.4f}| {mae_best_gen:<10.4f}{mae_best_ref:<10.4f}")
    
    print()
    
    print(f"{'RMSE':<10}| {'GT/GEN':<10}{'GT/REF':<10}{'GEN/REF':<10}| {'BEST GEN':<10}{'BEST REF':<10}")
    print(f"{'- XSCM':<10}| {rmse_avg_gen:<10.4f}{rmse_avg_ref:<10.4f}{rmse_gen_ref:<10.4f}| {rmse_best_gen:<10.4f}{rmse_best_ref:<10.4f}")
    
    print()
    
    print(f"{'SSIM':<10}| {'GT/GEN':<10}| {'BEST GEN':<10}")
    print(f"{'- XSCM':<10}| {ssim_avg:<10.4f}| {ssim_best:<10.4f}")

    print(f"{'LPIPS':<10}| {'GT/GEN':<10}| {'BEST GEN':<10}")
    print(f"{'- XSCM':<10}| {lpips_avg:<10.4f}| {lpips_best:<10.4f}")

    print("Done.")










