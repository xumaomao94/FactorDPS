import numpy as np
from scipy.special import sinc
import os
from scipy.io import savemat
import sys
from tqdm import tqdm
import time

# Import GPU shadowing function
try:
    try:
        from .shadowing_pytorch import shadowing_pytorch
    except ImportError:
        from datagenerator.shadowing_pytorch import shadowing_pytorch
    import torch
    
    if torch.cuda.is_available():
        PYTORCH_GPU_AVAILABLE = True
        print(f"PyTorch GPU shadowing available - {torch.cuda.device_count()} GPU(s) found")
    else:
        PYTORCH_GPU_AVAILABLE = False
        print("PyTorch imported but CUDA not available - using CPU version")
except ImportError:
    PYTORCH_GPU_AVAILABLE = False
    print("PyTorch GPU shadowing not available - using CPU version")

def smap_gen(map_size, R, shadow_sigma, d_corr, basis="sinc", num_peaks_per_psd=3, strictly_separable=True, seed=None, use_gpu=False):
    if seed is None:
        seed = int(np.sum(100 * np.random.rand()))
    np.random.seed(seed)

    K = map_size[2]

    indK = np.arange(1, K + 1)
    if basis == "gaussian":
        psd = lambda f0, sigma: np.exp(-(indK - f0) ** 2 / (2 * sigma ** 2))
    elif basis == "sinc":
        psd = lambda f0, psd_span: sinc((indK - f0) / psd_span) ** 2 * (np.abs((indK - f0) / psd_span) <= 1)

    if strictly_separable:
        ind_psd = np.arange(R * 3 + 2, K, 2)
    else:
        ind_psd = np.arange(2, K)

    Ctrue = np.zeros((K, R))
    if K != 1:
        if not strictly_separable:
            for rr in range(R):
                psd_peaks = np.random.choice(ind_psd, num_peaks_per_psd, replace=False)
                am = 0.5 + 1.5 * np.random.rand(num_peaks_per_psd)

                cr = 0
                for q in range(num_peaks_per_psd):
                    cr += am[q] * psd(psd_peaks[q], 2 + 2.5 * np.random.rand())
                Ctrue[:, rr] = cr / np.linalg.norm(cr)
        else:
            for rr in range(R):
                psd_peaks = np.random.choice(ind_psd, num_peaks_per_psd, replace=False)
                am = 0.5 + 1.5 * np.random.rand(num_peaks_per_psd)

                cr = am[0] * psd( 1+rr*3 , 1.5 + 1.5 * np.random.rand())
                for q in range(num_peaks_per_psd-1):
                    cr += am[q+1] * psd(psd_peaks[q], 2 + 2.5 * np.random.rand())
                Ctrue[:, rr] = cr / np.linalg.norm(cr)
    else:
        for rr in range(R):
            Ctrue[:, rr] = 0.5 + 1.5 * np.random.rand()

    def loss_f(x, d, alpha):
        return np.minimum(1, (x / d) ** (-alpha))

    d0 = 2

    Xmesh_grid, Ymesh_grid = np.meshgrid(np.arange(map_size[0]), np.arange(map_size[1]))
    Xgrid = Xmesh_grid + 1j * Ymesh_grid

    Sc = np.zeros(map_size[:2] + [R])
    for rr in range(R):
        location = map_size[0] * np.random.rand() + 1j * map_size[0] * np.random.rand()
        loss_mat = np.abs(Xgrid - location)
        alpha = 2 + 0.5 * np.random.rand()
        p = np.exp(-1 / d_corr)
        
        # Use GPU or CPU shadowing based on availability and user preference
        if use_gpu and PYTORCH_GPU_AVAILABLE:
            try:
                shadow = shadowing_pytorch(Xgrid, shadow_sigma, p, device='cuda')
            except Exception as e:
                print(f"GPU shadowing failed: {e}, falling back to CPU")
                shadow = shadowing(Xgrid, shadow_sigma, p)
        else:
            shadow = shadowing(Xgrid, shadow_sigma, p)
            
        shadow_linear = 10 ** (shadow / 10)

        Sc_r = loss_f(loss_mat, d0, alpha) * shadow_linear
        Sc_r = Sc_r / np.linalg.norm(Sc_r, 'fro')
        Sc[:, :, rr] = Sc_r

    Sc_vec = Sc.reshape(-1, R)
    X = Sc_vec @ Ctrue.T
    X = X.reshape(map_size)

    return {'SLF': Sc, 'PSD': Ctrue, 'RM': X, 'mask': None, 'alpha': alpha, 'location': location}


def shadowing(Cloc, var, p=None):
    if p is None:
        p = np.exp(-1 / 50)
    
    if var == 0:
        return np.zeros_like(Cloc)
    else:
        m, n = Cloc.shape
        vec_Cloc = Cloc.flatten()
        shadowing_iid = var * np.random.randn(m, n)
        vec_shadowing_iid = shadowing_iid.flatten()
        R = lambda d: p ** d
        distance_corr = np.abs(vec_Cloc[:, None] - vec_Cloc[None, :])
        S = np.linalg.cholesky(R(distance_corr))
        vec_shadowing_correlation = S @ vec_shadowing_iid
        shadowing_correlation = vec_shadowing_correlation.reshape((m, n))
        return shadowing_correlation
    
    
if __name__ == "__main__":

    slf_DIR = "/nfs/stak/users/xul2/hpc-share/datasets/SC/Data_128_rand/"
    os.makedirs(slf_DIR, exist_ok=True)
    if len(sys.argv) >= 3:
        start_idx = int(sys.argv[1])
        num_files = int(sys.argv[2])
    else:
        print("Usage: python slfgenerator.py <start_idx> <num_files>")
        sys.exit(1)
        
    DETAILS_FILE = '/nfs/stak/users/xul2/ADMMPnP/ComparingAlgos/Deep-SC-main/deep_prior/dataset/details_128.csv'
    
    map_size = [128, 128, 1]
    R = 1
    basis = "sinc"
    num_peaks_per_psd = 1
    strictly_separable = True
    use_gpu = True
    
    f = open(DETAILS_FILE, 'a')
    if os.path.getsize(DETAILS_FILE) == 0:
        f.write("filename,sigma,d_corr,alpha,location_real,location_imag\n")

    for idx in tqdm(range(start_idx, start_idx + num_files), desc="Generating SLF files"):
        # Set random seed based on current time and index for better randomness
        # random_seed = int(time.time() * 1000000) % 2**32 + idx
        random_seed = hash((time.time(), idx)) % 2**32
        np.random.seed(random_seed)
        
        shadow_sigma = np.random.uniform(3, 9)
        d_corr = np.random.uniform(30, 100)
        sample = smap_gen(
            map_size=map_size,
            R=R,
            shadow_sigma=shadow_sigma,
            d_corr=d_corr,
            basis=basis,
            num_peaks_per_psd=num_peaks_per_psd,
            strictly_separable=strictly_separable,
            seed=random_seed,
            use_gpu=use_gpu
        )
        Sc = np.squeeze(sample['SLF'])
        alpha = sample['alpha']
        location = sample['location']
        
        filename = os.path.join(slf_DIR, f"slf_{idx:06d}.mat")
        savemat(filename, {'Sc':Sc})
        
        with open(DETAILS_FILE, 'a') as f:
            f.write(f"{filename},{shadow_sigma},{d_corr},{alpha},{location.real},{location.imag}\n")
        
        if (idx - start_idx + 1) % 1000 == 0:
            print(f"Generated {idx - start_idx + 1} / {num_files} files")
            
    f.close()
            