import subprocess
import os

def run_demo():
    # Settings derived from tsp_exp8.sbatch
    method = "dps"
    config = "config/inference.json"
    save_folder = "./experiments/demo_hole_missing"
    
    sr = 5 # sampling rate 5%
    hole_ratio = 0.4 # 40% hole ratio
    
    os.makedirs(save_folder, exist_ok=True)
    
    cmd = [
        "python", "run_method.py",
        "-c", config,
        "-method", method,
        "-emitter_number", "3",
        "-band_number", "1",
        "-sample_rate", str(sr),
        "-quantization_bits", "0",
        "-hole_ratio", str(hole_ratio),
        "-save_folder", save_folder,
        "-random_seed", "0"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_demo()