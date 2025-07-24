#!/usr/bin/env python3
"""
ChemiAccel GPU Kernel Validator
Validates CUDA/ROCm kernel syntax and common patterns.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

def validate_cuda_kernel(file_path: Path) -> List[str]:
    """Validate CUDA kernel file."""
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return [f"Could not read file: {e}"]
    
    # Check for common CUDA patterns
    checks = [
        (r'__global__\s+void\s+\w+\s*\([^)]*\)\s*{', "CUDA kernel function"),
        (r'threadIdx\.[xyz]', "Thread indexing"),
        (r'blockIdx\.[xyz]', "Block indexing"),
        (r'blockDim\.[xyz]', "Block dimension"),
        (r'gridDim\.[xyz]', "Grid dimension"),
    ]
    
    found_patterns = []
    for pattern, description in checks:
        if re.search(pattern, content):
            found_patterns.append(description)
    
    # Basic syntax checks
    if '__global__' in content:
        # Check for proper kernel launch syntax in comments or examples
        if not re.search(r'<<<.*>>>', content) and '<<<' not in content:
            errors.append("CUDA kernel defined but no launch syntax found in comments")
    
    # Check for common memory access patterns
    if '__shared__' in content:
        if not re.search(r'__syncthreads\(\)', content):
            errors.append("Shared memory used but no __syncthreads() found")
    
    # Check for proper error handling patterns
    cuda_api_calls = re.findall(r'cuda\w+\s*\(', content)
    if cuda_api_calls and 'cudaGetLastError' not in content:
        errors.append("CUDA API calls found but no error checking detected")
    
    if found_patterns:
        print(f"PASS: {file_path}: Found CUDA patterns: {', '.join(found_patterns)}")
    
    return errors

def validate_hip_kernel(file_path: Path) -> List[str]:
    """Validate ROCm/HIP kernel file."""
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return [f"Could not read file: {e}"]
    
    # Check for HIP patterns
    if '__global__' in content and 'hipLaunchKernelGGL' not in content:
        if not re.search(r'<<<.*>>>', content):
            errors.append("HIP kernel defined but no launch syntax found")
    
    return errors

def main():
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: validate_gpu_kernels.py <file1> [file2] ...")
        return 0
    
    files = sys.argv[1:]
    all_errors = []
    
    for file_path_str in files:
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            continue
            
        if file_path.suffix == '.cu':
            errors = validate_cuda_kernel(file_path)
        elif file_path.suffix == '.hip':
            errors = validate_hip_kernel(file_path)
        else:
            continue
            
        if errors:
            all_errors.extend([f"{file_path}: {error}" for error in errors])
    
    if all_errors:
        print("GPU kernel validation issues found:")
        for error in all_errors:
            print(f"  ERROR: {error}")
        return 1
    else:
        if files:
            print("PASS: GPU kernel validation passed")
        return 0

if __name__ == "__main__":
    sys.exit(main())