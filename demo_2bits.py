import subprocess
import os

def run_demo():
    # Settings derived from tsp_exp8.sbatch
    method = "dps"
    config = "config/inference.json"
    save_folder = "./experiments/demo_2bits"
    
    # Set the data root directory for RadioMapSeer dataset
    data_root_dir = "/nfs/stak/users/xul2/hpc-share/datasets/SC/RadioMapSeer" 
    
    sr = 5 # sampling rate 1%
    bits = 2 # 2-bit quantization
    
    os.makedirs(save_folder, exist_ok=True)
    
    cmd = [
        "python", "run_method.py",
        "-c", config,
        "-method", method,
        "-emitter_number", "3",
        "-band_number", "1",
        "-sample_rate", str(sr),
        "-quantization_bits", str(bits),
        "-hole_ratio", "0",
        "-data_root_dir", data_root_dir,
        "-save_folder", save_folder,
        "-random_seed", "200"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_demo()