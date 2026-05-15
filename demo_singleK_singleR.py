import subprocess
import os

def run_demo():
    # Settings derived from tsp_exp1.sbatch
    method = "dps"
    config = "config/inference.json"
    save_folder = "./experiments/demo_K1_R1"
    
    # Set the data root directory for RadioMapSeer dataset
    data_root_dir = "/nfs/stak/users/xul2/hpc-share/datasets/SC/RadioMapSeer" 
    
    sr = 1 # sampling rate 1%
    
    os.makedirs(save_folder, exist_ok=True)
    
    cmd = [
        "python", "run_method.py",
        "-c", config,
        "-method", method,
        "-emitter_number", "1",
        "-band_number", "1",
        "-sample_rate", str(sr),
        "-quantization_bits", "0",
        "-hole_ratio", "0",
        "-data_root_dir", data_root_dir,
        "-save_folder", save_folder,
        "-random_seed", "0"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_demo()
