import torch
import numpy as np

def shadowing_pytorch(Cloc, var, p=None, device='cuda'):
    """
    PyTorch GPU implementation of shadowing function
    
    Args:
        Cloc: Complex location grid (numpy array)
        var: Variance parameter for shadowing
        p: Correlation parameter (default: exp(-1/50))
        device: 'cuda' for GPU or 'cpu' for CPU
    
    Returns:
        Shadowing correlation matrix (numpy array)
    """
    if p is None:
        p = np.exp(-1 / 50)
    
    if var == 0:
        return np.zeros_like(Cloc, dtype=np.float64)
    
    # Check if CUDA is available when requested
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Convert input to appropriate dtype and device
    # Handle complex numbers by converting to float64 for distance calculation
    if np.iscomplexobj(Cloc):
        # For complex input, we'll work with the absolute values for distance
        Cloc_real = torch.from_numpy(np.real(Cloc)).to(device, dtype=torch.float64)
        Cloc_imag = torch.from_numpy(np.imag(Cloc)).to(device, dtype=torch.float64)
        Cloc_torch = torch.complex(Cloc_real, Cloc_imag)
    else:
        Cloc_torch = torch.from_numpy(Cloc).to(device, dtype=torch.float64)
    
    m, n = Cloc_torch.shape
    
    # Flatten the location grid
    vec_Cloc = Cloc_torch.flatten()
    
    # Generate IID shadowing on the specified device
    torch.manual_seed(42)  # For reproducibility
    shadowing_iid = var * torch.randn(m * n, device=device, dtype=torch.float64)
    
    # Compute pairwise distances
    # For complex numbers, use absolute difference
    if torch.is_complex(vec_Cloc):
        distance_corr = torch.abs(vec_Cloc[:, None] - vec_Cloc[None, :])
    else:
        distance_corr = torch.abs(vec_Cloc[:, None] - vec_Cloc[None, :])
    
    # Apply correlation function R(d) = p^d
    R_matrix = torch.pow(p, distance_corr)
    
    # Add small diagonal regularization for numerical stability
    R_matrix = R_matrix + 1e-10 * torch.eye(R_matrix.shape[0], device=device, dtype=torch.float64)
    
    # Cholesky decomposition
    try:
        S = torch.linalg.cholesky(R_matrix)
        
        # Apply correlation transformation
        vec_shadowing_correlation = S @ shadowing_iid
        
        # Reshape back to original grid shape
        shadowing_correlation = vec_shadowing_correlation.reshape((m, n))
        
        # Convert back to numpy array
        result = shadowing_correlation.cpu().numpy()
        
        return result
        
    except torch.linalg.LinAlgError as e:
        print(f"Cholesky decomposition failed on {device}: {e}")
        if device == 'cuda':
            print("Falling back to CPU...")
            return shadowing_pytorch(Cloc, var, p, device='cpu')
        else:
            print("Using fallback method...")
            return _shadowing_fallback(Cloc, var, p)


def _shadowing_fallback(Cloc, var, p):
    """
    Fallback CPU implementation using numpy when Cholesky fails
    """
    if p is None:
        p = np.exp(-1 / 50)
    
    m, n = Cloc.shape
    vec_Cloc = Cloc.flatten()
    
    # Generate IID shadowing
    np.random.seed(42)  # For reproducibility
    shadowing_iid = var * np.random.randn(m * n)
    
    # Compute distance correlation
    distance_corr = np.abs(vec_Cloc[:, None] - vec_Cloc[None, :])
    R_matrix = p ** distance_corr
    
    # Add regularization
    R_matrix = R_matrix + 1e-10 * np.eye(R_matrix.shape[0])
    
    try:
        S = np.linalg.cholesky(R_matrix)
        vec_shadowing_correlation = S @ shadowing_iid
        return vec_shadowing_correlation.reshape((m, n))
    except np.linalg.LinAlgError:
        print("Fallback Cholesky also failed, using simplified correlation")
        # Very simple fallback - just return scaled noise
        return (var * np.random.randn(m, n)).astype(np.float64)


def check_gpu_availability():
    """
    Check if GPU acceleration is available
    
    Returns:
        bool: True if GPU is available and working
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Test a simple operation on GPU
        test_tensor = torch.randn(10, 10, device='cuda')
        _ = torch.matmul(test_tensor, test_tensor.T)
        return True
    except Exception as e:
        print(f"GPU test failed: {e}")
        return False


def benchmark_shadowing(size=64, var=4.0, device='cuda'):
    """
    Benchmark GPU vs CPU performance
    
    Args:
        size: Grid size (size x size)
        var: Variance parameter
        device: 'cuda' or 'cpu'
    
    Returns:
        dict: Performance metrics
    """
    import time
    
    # Create test data
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    Cloc = x + 1j * y
    p = np.exp(-1/50)
    
    print(f"Benchmarking shadowing with {size}x{size} grid on {device.upper()}...")
    
    # Warm up (for GPU)
    if device == 'cuda' and torch.cuda.is_available():
        _ = shadowing_pytorch(Cloc[:16, :16], var, p, device=device)
        torch.cuda.synchronize()
    
    # Actual benchmark
    start_time = time.time()
    
    if device == 'cuda' and torch.cuda.is_available():
        result = shadowing_pytorch(Cloc, var, p, device=device)
        torch.cuda.synchronize()  # Wait for GPU to finish
    else:
        result = shadowing_pytorch(Cloc, var, p, device='cpu')
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Memory usage (approximate)
    total_elements = size * size
    distance_matrix_size = total_elements ** 2 * 8 / (1024**3)  # GB for float64
    
    print(f"{device.upper()} execution time: {execution_time:.4f} seconds")
    print(f"Distance matrix size: {distance_matrix_size:.2f} GB")
    print(f"Result shape: {result.shape}")
    
    return {
        'device': device,
        'execution_time': execution_time,
        'matrix_size_gb': distance_matrix_size,
        'result_shape': result.shape
    }


if __name__ == "__main__":
    # Test and benchmark the function
    print("=== GPU Availability Check ===")
    gpu_available = check_gpu_availability()
    print(f"GPU available: {gpu_available}")
    
    if gpu_available:
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n=== Performance Benchmark ===")
    
    # Test with small grid first
    test_size = 32
    cpu_results = benchmark_shadowing(test_size, device='cpu')
    
    if gpu_available:
        gpu_results = benchmark_shadowing(test_size, device='cuda')
        speedup = cpu_results['execution_time'] / gpu_results['execution_time']
        print(f"GPU speedup: {speedup:.2f}x")
    
    print("\nBenchmark complete!")
