#!/usr/bin/env bash
#SBATCH --job-name=diag-jax-cuda
#SBATCH --account=def-pviswana_gpu
#SBATCH --partition=gpubase_bygpu_b4
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/diag_jax_cuda_%j.out

set -uo pipefail
cd /scratch/lblommes/diamondworld
source scripts/use_nibi_env.sh

echo "=== nvidia-smi ==="
nvidia-smi
echo ""
echo "=== Driver version ==="
nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo ""
echo "=== CUDA_HOME / module env ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "EBROOTCUDA=${EBROOTCUDA:-unset}"
echo "EBROOTCUDNN=${EBROOTCUDNN:-unset}"
echo ""
echo "=== LD_LIBRARY_PATH (cuda/cudnn parts) ==="
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -iE 'cuda|cudnn' || echo "(no cuda/cudnn entries)"
echo ""
echo "=== libcuda.so / libcudart.so visibility ==="
ldconfig -p 2>/dev/null | grep -E "libcuda\.so|libcudart\.so|libcublas\.so" | head -10
echo ""
echo "=== JAX import + device probe ==="
python <<'PY'
import os
print("XLA_PYTHON_CLIENT_PREALLOCATE =", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
import jax
print("jax version:", jax.__version__)
print("jax.default_backend():", jax.default_backend())
try:
    print("jax.devices():", jax.devices())
except Exception as e:
    print("jax.devices() raised:", repr(e))
PY
echo ""
echo "=== retry with XLA verbose logging ==="
TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=stream_executor=2,gpu_executor=2 python -c "import jax; print(jax.devices())" 2>&1 | head -40
