# Hill Cipher Cracking System - Windows Setup Guide

This guide explains how to set up and run the Hill cipher cracking system on Windows with both GPU (CUDA) and CPU-only options.

## System Requirements

### Minimum Requirements (CPU-Only)
- Windows 10/11 (64-bit)
- Python 3.8 or higher
- 8 GB RAM
- 2 GB free disk space

### Recommended Requirements (GPU Acceleration)
- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA 11.0 or higher
- Python 3.8 or higher
- 16 GB RAM
- 5 GB free disk space

### Check Your GPU
To check if you have a compatible NVIDIA GPU:
1. Open Command Prompt
2. Run: `nvidia-smi`
3. If you see GPU information, you have NVIDIA GPU support

## Installation Options

### Option 1: Quick CPU-Only Setup (Recommended for beginners)

1. **Install Python**
   - Download Python from https://python.org
   - Make sure to check "Add Python to PATH" during installation

2. **Install Dependencies**
   ```cmd
   pip install numpy scipy matplotlib pandas scikit-learn
   ```

3. **Test Installation**
   ```cmd
   python -c "import numpy; print('CPU setup ready!')"
   ```

### Option 2: GPU-Accelerated Setup (Advanced users)

1. **Install NVIDIA CUDA Toolkit**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Choose Windows x86_64
   - Install with default settings
   - Restart your computer

2. **Install Python Dependencies**
   ```cmd
   pip install numpy scipy matplotlib pandas
   ```

3. **Install CUDA Python Libraries**
   ```cmd
   # For CUDA 11.x
   pip install cupy-cuda11x
   
   # For CUDA 12.x
   pip install cupy-cuda12x
   ```

4. **Verify GPU Setup**
   ```cmd
   python -c "import cupy; print('GPU setup ready!')"
   ```

## Running the Code

### Basic Usage

1. **Navigate to Project Directory**
   ```cmd
   cd path\to\hill-cipher-cracking
   ```

2. **Run 2x2 Cipher Cracking (CPU)**
   ```cmd
   python run_all_hill_tests.py
   ```

3. **Run 3x3 Cipher with GPU**
   ```cmd
   python hill_cipher\breakers\cuda_breaker.py --ciphertext "your_cipher_here" --key-size 3
   ```

4. **Run 3x3 Cipher CPU-Only**
   ```cmd
   python hill_cipher\breakers\cuda_breaker.py --ciphertext "your_cipher_here" --key-size 3 --no-gpu
   ```

### Test Cases

**2x2 Ciphers (Known to work):**
```cmd
# Test 2x2 cipher cracking
python hill_cipher\breakers\enhanced_breaker.py --key-size 2 --method exhaustive
```

**3x3 Ciphers:**
```cmd
# Known 3x3 cipher
python hill_cipher\breakers\cuda_breaker.py --ciphertext "ysigztwrqxoegwfwveyjlcjlkpqbcggpqkdymglsavyacolzewfoxglvalewktqczasmtihavacolzewfstaocaxqvopiwkaxiwyawcjljaalrgpgqvgezmn" --key-size 3

# Unknown 3x3 cipher
python hill_cipher\breakers\cuda_breaker.py --ciphertext "aoaldaebgaoilwiuhmrhtwoagignwihpnfoommsmwmsllgwatayqcamooarehvtgjgucsmqqqntypvyzzgmelzzjjzavalkazbmnammxxlzdypazttxooshn" --key-size 3
```

## Performance Comparison

| Key Size | CPU-Only (keys/sec) | GPU Accelerated (keys/sec) | Speedup |
|----------|--------------------|-----------------------------|---------|
| 2x2      | 200-300            | 500-1000                   | 2-3x    |
| 3x3      | 100-200            | 1000-5000                  | 5-25x   |
| 4x4      | 50-100             | 2000-10000                 | 20-100x |

## Troubleshooting

**"CUDA not found" Error:**
```cmd
nvcc --version
nvidia-smi
```

**"CuPy installation failed":**
```cmd
pip install cupy-cuda12x  # Try different CUDA version
```

**"Module not found" Errors:**
```cmd
pip install numpy scipy matplotlib pandas
```

**GPU Out of Memory:**
```cmd
python cuda_breaker.py --batch-size 256  # Reduce batch size
```

## Quick Start Commands

**Test GPU setup:**
```cmd
python hill_cipher\breakers\cuda_breaker.py --gpu-info
```

**Run all tests:**
```cmd
python run_all_hill_tests.py
```

**GPU-accelerated 3x3 cracking:**
```cmd
python hill_cipher\breakers\cuda_breaker.py --ciphertext "your_cipher" --key-size 3
```

**CPU-only mode:**
```cmd
python hill_cipher\breakers\cuda_breaker.py --ciphertext "your_cipher" --key-size 3 --no-gpu
```

The system automatically falls back to CPU processing if GPU acceleration is not available.
