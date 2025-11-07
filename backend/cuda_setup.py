#!/usr/bin/env python3
"""
CUDA Setup Utility for AI Video Editor Style Filters
This script helps configure and test CUDA support for the style filters.
"""

import torch
import sys
import subprocess
import os

def check_cuda_installation():
    """Check if CUDA is properly installed and accessible"""
    print("🔍 Checking CUDA Installation...")
    print("=" * 50)
    
    # Check PyTorch CUDA support
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("❌ CUDA not available in PyTorch")
        return False
    
    return True

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print("\n🔍 Checking NVIDIA Driver...")
    print("=" * 50)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver is installed")
            print("GPU Information:")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA driver not found or not working")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found. NVIDIA driver may not be installed.")
        return False

def test_cuda_performance():
    """Test CUDA performance with a simple benchmark"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping performance test")
        return
    
    print("\n🚀 Testing CUDA Performance...")
    print("=" * 50)
    
    device = torch.device('cuda:0')
    
    # Test tensor operations
    print("Testing tensor operations...")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    import time
    start_time = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Matrix multiplication (1000x1000): {end_time - start_time:.4f} seconds")
    
    # Test memory allocation
    print("Testing memory allocation...")
    try:
        # Try to allocate a large tensor
        large_tensor = torch.randn(2000, 2000, 3).to(device)
        print(f"✅ Successfully allocated tensor of size: {large_tensor.numel() * 4 / 1e6:.1f} MB")
        del large_tensor
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"❌ Memory allocation failed: {e}")

def optimize_cuda_settings():
    """Optimize CUDA settings for better performance"""
    print("\n⚙️ Optimizing CUDA Settings...")
    print("=" * 50)
    
    if torch.cuda.is_available():
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        print("✅ Enabled cuDNN benchmarking")
        
        # Set memory fraction to avoid OOM errors
        torch.cuda.set_per_process_memory_fraction(0.8)
        print("✅ Set GPU memory fraction to 80%")
        
        # Clear cache
        torch.cuda.empty_cache()
        print("✅ Cleared GPU cache")
    else:
        print("❌ CUDA not available, skipping optimization")

def create_cuda_config():
    """Create a CUDA configuration file"""
    config_content = """# CUDA Configuration for AI Video Editor
# This file contains CUDA settings for optimal performance

# GPU Memory Settings
GPU_MEMORY_FRACTION = 0.8
BATCH_SIZE = 4

# Performance Settings
CUDNN_BENCHMARK = True
CUDNN_DETERMINISTIC = False

# Style Filter Settings
NEURAL_STYLE_ENABLED = True
TRADITIONAL_FALLBACK = True

# Memory Management
AUTO_CLEAR_CACHE = True
"""
    
    config_path = "backend/cuda_config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created CUDA configuration file: {config_path}")

def main():
    """Main function to run all CUDA checks and setup"""
    print("🎨 AI Video Editor - CUDA Setup Utility")
    print("=" * 60)
    
    # Check installations
    cuda_available = check_cuda_installation()
    driver_available = check_nvidia_driver()
    
    if cuda_available and driver_available:
        print("\n✅ CUDA is properly configured!")
        
        # Test performance
        test_cuda_performance()
        
        # Optimize settings
        optimize_cuda_settings()
        
        # Create config
        create_cuda_config()
        
        print("\n🎉 CUDA setup completed successfully!")
        print("Your AI Style Filters will now use GPU acceleration.")
        
    else:
        print("\n❌ CUDA setup incomplete.")
        print("\nTo enable CUDA support:")
        print("1. Install NVIDIA GPU drivers")
        print("2. Install CUDA Toolkit")
        print("3. Install PyTorch with CUDA support")
        print("4. Run this script again")
        
        print("\nFor now, the style filters will use CPU processing.")

if __name__ == "__main__":
    main()
