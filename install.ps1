# AEROMamba Installation Script for PyTorch 2.x & CUDA 12.x
# This script automates the installation process

Write-Host "================================" -ForegroundColor Cyan
Write-Host "AEROMamba Installation Script" -ForegroundColor Cyan
Write-Host "PyTorch 2.x + CUDA 12.x" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Check Python version (should be 3.10 or later)
$version = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$majorVersion = [int]$version.Split('.')[0]
$minorVersion = [int]$version.Split('.')[1]

if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 10)) {
    Write-Host "Error: Python 3.10 or later is required. Found: $version" -ForegroundColor Red
    exit 1
}
Write-Host "Python version $version is compatible" -ForegroundColor Green
Write-Host ""

# Check CUDA availability
Write-Host "Checking NVIDIA GPU and CUDA..." -ForegroundColor Yellow
$nvidiaSmi = nvidia-smi 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "NVIDIA GPU detected" -ForegroundColor Green
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} else {
    Write-Host "Warning: No NVIDIA GPU detected or nvidia-smi not found" -ForegroundColor Yellow
    Write-Host "CPU-only installation will proceed" -ForegroundColor Yellow
}
Write-Host ""

# Ask user for installation method
Write-Host "Choose installation method:" -ForegroundColor Cyan
Write-Host "1. Quick install (pip install -r requirements.txt)" -ForegroundColor White
Write-Host "2. Manual install (step-by-step with verification)" -ForegroundColor White
Write-Host "3. Install with CUDA 12.1 (recommended for RTX 30xx/40xx)" -ForegroundColor White
Write-Host "4. CPU-only installation" -ForegroundColor White
$choice = Read-Host "Enter choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host "`nInstalling dependencies from requirements.txt..." -ForegroundColor Yellow
        pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error during installation. Try option 2 for step-by-step installation." -ForegroundColor Red
            exit 1
        }
    }
    "2" {
        Write-Host "`nStep 1: Installing PyTorch with CUDA 12.1..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error installing PyTorch" -ForegroundColor Red
            exit 1
        }
        Write-Host "PyTorch installed successfully" -ForegroundColor Green

        Write-Host "`nStep 2: Installing Mamba dependencies..." -ForegroundColor Yellow
        pip install causal-conv1d>=1.1.0
        pip install mamba-ssm>=1.1.0
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: Mamba installation may require compilation" -ForegroundColor Yellow
            Write-Host "Trying with --no-build-isolation..." -ForegroundColor Yellow
            pip install causal-conv1d --no-build-isolation
            pip install mamba-ssm --no-build-isolation
        }
        Write-Host "Mamba dependencies installed" -ForegroundColor Green

        Write-Host "`nStep 3: Installing other dependencies..." -ForegroundColor Yellow
        pip install colorlog==6.8.0 hydra-colorlog==1.2.0 hydra-core==1.3.2
        pip install hyperlink==21.0.0 HyperPyYAML==1.2.2
        pip install "matplotlib>=3.7.0" "numpy>=1.24.0,<2.0.0" "opencv-python>=4.9.0.80"
        pip install "soundfile>=0.12.1" "tqdm>=4.66.0" "wandb>=0.16.0"
        Write-Host "All dependencies installed" -ForegroundColor Green
    }
    "3" {
        Write-Host "`nInstalling PyTorch with CUDA 12.1..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install causal-conv1d>=1.1.0 mamba-ssm>=1.1.0
        pip install colorlog hydra-core hydra-colorlog HyperPyYAML matplotlib numpy soundfile tqdm wandb opencv-python
    }
    "4" {
        Write-Host "`nInstalling CPU-only version..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install causal-conv1d>=1.1.0 mamba-ssm>=1.1.0
        pip install colorlog hydra-core hydra-colorlog HyperPyYAML matplotlib numpy soundfile tqdm wandb opencv-python
    }
    default {
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Verifying Installation" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Verify PyTorch installation
Write-Host "`nChecking PyTorch..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: PyTorch import failed" -ForegroundColor Red
    exit 1
}

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('CPU only')"

# Check if Mamba is installed
Write-Host "`nChecking Mamba..." -ForegroundColor Yellow
python -c "import mamba_ssm; print('Mamba-SSM installed successfully')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Mamba-SSM not installed correctly" -ForegroundColor Yellow
    Write-Host "You may need to install it manually" -ForegroundColor Yellow
}

# Check other key dependencies
Write-Host "`nChecking other dependencies..." -ForegroundColor Yellow
python -c "import hydra; import wandb; import numpy; import soundfile; print('All key dependencies found')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Some dependencies may be missing" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Prepare your dataset (see README.md)" -ForegroundColor White
Write-Host "2. Run training: python train.py dset=<dset-name> experiment=<experiment-name>" -ForegroundColor White
Write-Host "3. For testing: python test.py dset=<dset-name> experiment=<experiment-name>" -ForegroundColor White
Write-Host ""
Write-Host "For detailed migration information, see MIGRATION_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
