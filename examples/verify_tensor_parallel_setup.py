"""
Verify that your environment supports tensor parallelism.

Run this script to check if you have the right PyTorch version and CUDA setup.
"""

import sys

def check_pytorch():
    """Check PyTorch installation and version."""
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        
        # Check for tensor parallel support
        version_parts = torch.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1].split('+')[0])  # Handle versions like "2.3.0+cu121"
        
        if major > 2 or (major == 2 and minor >= 3):
            print(f"✓ PyTorch version supports tensor parallelism (>= 2.3.0)")
        else:
            print(f"✗ PyTorch version too old. Need >= 2.3.0, have {torch.__version__}")
            return False
        
        # Check for required modules
        try:
            from torch.distributed.tensor.parallel import parallelize_module
            print("✓ torch.distributed.tensor.parallel module available")
        except ImportError as e:
            print(f"✗ torch.distributed.tensor.parallel not available: {e}")
            return False
        
        try:
            from torch.distributed.device_mesh import init_device_mesh
            print("✓ torch.distributed.device_mesh module available")
        except ImportError as e:
            print(f"✗ torch.distributed.device_mesh not available: {e}")
            return False
        
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  - GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
            return True
        else:
            print("✗ CUDA not available")
            print("  Note: Tensor parallelism requires CUDA/GPUs")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def check_transformers():
    """Check transformers library."""
    try:
        import transformers
        print(f"✓ Transformers installed: {transformers.__version__}")
        return True
    except ImportError:
        print("✗ Transformers not installed")
        print("  Install with: pip install transformers")
        return False


def check_nccl():
    """Check NCCL backend for distributed training."""
    try:
        import torch
        if torch.cuda.is_available():
            # NCCL is typically included with CUDA-enabled PyTorch
            print("✓ NCCL backend should be available (included with CUDA PyTorch)")
            return True
        else:
            print("✗ NCCL requires CUDA")
            return False
    except Exception as e:
        print(f"✗ Error checking NCCL: {e}")
        return False


def main():
    print("="*70)
    print("Tensor Parallelism Environment Check")
    print("="*70)
    print()
    
    all_good = True
    
    print("Checking PyTorch...")
    if not check_pytorch():
        all_good = False
    print()
    
    print("Checking CUDA...")
    if not check_cuda():
        all_good = False
    print()
    
    print("Checking Transformers...")
    if not check_transformers():
        all_good = False
    print()
    
    print("Checking NCCL...")
    if not check_nccl():
        all_good = False
    print()
    
    print("="*70)
    if all_good:
        print("✓ All checks passed! You're ready to run tensor parallelism.")
        print()
        print("To run the example:")
        print("  torchrun --nproc_per_node=2 test_tensor_parallelism.py")
        print()
        print("Or with specific GPUs:")
        print("  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_tensor_parallelism.py")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print()
        print("To install/upgrade PyTorch with CUDA:")
        print("  pip install torch>=2.3.0 --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("To install transformers:")
        print("  pip install transformers")
    print("="*70)
    
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())

