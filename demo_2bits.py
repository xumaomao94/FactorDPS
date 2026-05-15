import subprocess
import os

def run_demo():
    # Settings derived from tsp_exp8.sbatch
    method = "dps"
    config = "config/inference.json"
    save_folder = "./experiments/demo_2bits"
    
    sr = 1 # sampling rate 1%
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
        "-save_folder", save_folder,
        "-random_seed", "0"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_demo()