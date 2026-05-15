import subprocess
import os

def run_demo():
    # Settings derived from tsp_exp1.sbatch
    method = "dps"
    config = "config/inference.json"
    testing_data_folder = "/nfs/stak/users/xul2/hpc-share/sc-diffuse/TSP/settings/exp1_mr"
    save_folder = "./experiments/demo_exp1"
    
    # We will take SR as 1% and experiment ID as 0
    exp_id = 1
    sr = 1
    data_name = f"sr{sr}-exp{exp_id:03d}.npz"
    
    os.makedirs(save_folder, exist_ok=True)
    
    cmd = [
        "python", "run_method.py",
        "-c", config,
        "-method", method,
        "-testing_data_folder", testing_data_folder,
        "-testing_data_name", data_name,
        "-emitter_number", "1",
        "-band_number", "1",
        "-sample_rate", str(sr),
        "-quantization_bits", "0",
        "-dithering_sigma", "0",
        "-hole_ratio", "0",
        "-exp_id", str(exp_id),
        "-save_folder", save_folder,
        "-random_seed", "100"
        # "-enable_wandb" # assuming we disable it or it's handled; the user said wandb is commented out
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_demo()
