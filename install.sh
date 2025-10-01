#!/bin/bash

# AEROMamba Installation Script for PyTorch 2.x & CUDA 12.x
# This script automates the installation process for Linux/Mac

echo "================================"
echo "AEROMamba Installation Script"
echo "PyTorch 2.x + CUDA 12.x"
echo "================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Python is installed
echo -e "${YELLOW}Checking Python installation...${NC}"
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PIP_CMD=pip3
else
    PYTHON_CMD=python
    PIP_CMD=pip
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}Found: $PYTHON_VERSION${NC}"

# Check Python version (should be 3.10 or later)
VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
MAJOR=$(echo $VERSION | cut -d. -f1)
MINOR=$(echo $VERSION | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or later is required. Found: $VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}Python version $VERSION is compatible${NC}"
echo ""

# Check CUDA availability
echo -e "${YELLOW}Checking NVIDIA GPU and CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}Warning: No NVIDIA GPU detected or nvidia-smi not found${NC}"
    echo -e "${YELLOW}CPU-only installation will proceed${NC}"
fi
echo ""

# Ask user for installation method
echo -e "${CYAN}Choose installation method:${NC}"
echo "1. Quick install (pip install -r requirements.txt)"
echo "2. Manual install (step-by-step with verification)"
echo "3. Install with CUDA 12.1 (recommended for RTX 30xx/40xx)"
echo "4. CPU-only installation"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo -e "\n${YELLOW}Installing dependencies from requirements.txt...${NC}"
        $PIP_CMD install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo -e "${RED}Error during installation. Try option 2 for step-by-step installation.${NC}"
            exit 1
        fi
        ;;
    2)
        echo -e "\n${YELLOW}Step 1: Installing PyTorch with CUDA 12.1...${NC}"
        $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        if [ $? -ne 0 ]; then
            echo -e "${RED}Error installing PyTorch${NC}"
            exit 1
        fi
        echo -e "${GREEN}PyTorch installed successfully${NC}"

        echo -e "\n${YELLOW}Step 2: Installing Mamba dependencies...${NC}"
        $PIP_CMD install "causal-conv1d>=1.1.0"
        $PIP_CMD install "mamba-ssm>=1.1.0"
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Warning: Mamba installation may require compilation${NC}"
            echo -e "${YELLOW}Trying with --no-build-isolation...${NC}"
            $PIP_CMD install causal-conv1d --no-build-isolation
            $PIP_CMD install mamba-ssm --no-build-isolation
        fi
        echo -e "${GREEN}Mamba dependencies installed${NC}"

        echo -e "\n${YELLOW}Step 3: Installing other dependencies...${NC}"
        $PIP_CMD install colorlog==6.8.0 hydra-colorlog==1.2.0 hydra-core==1.3.2
        $PIP_CMD install hyperlink==21.0.0 HyperPyYAML==1.2.2
        $PIP_CMD install "matplotlib>=3.7.0" "numpy>=1.24.0,<2.0.0" "opencv-python>=4.9.0.80"
        $PIP_CMD install "soundfile>=0.12.1" "tqdm>=4.66.0" "wandb>=0.16.0"
        echo -e "${GREEN}All dependencies installed${NC}"
        ;;
    3)
        echo -e "\n${YELLOW}Installing PyTorch with CUDA 12.1...${NC}"
        $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        $PIP_CMD install "causal-conv1d>=1.1.0" "mamba-ssm>=1.1.0"
        $PIP_CMD install colorlog hydra-core hydra-colorlog HyperPyYAML matplotlib numpy soundfile tqdm wandb opencv-python
        ;;
    4)
        echo -e "\n${YELLOW}Installing CPU-only version...${NC}"
        $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        $PIP_CMD install "causal-conv1d>=1.1.0" "mamba-ssm>=1.1.0"
        $PIP_CMD install colorlog hydra-core hydra-colorlog HyperPyYAML matplotlib numpy soundfile tqdm wandb opencv-python
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${CYAN}================================${NC}"
echo -e "${CYAN}Verifying Installation${NC}"
echo -e "${CYAN}================================${NC}"

# Verify PyTorch installation
echo -e "\n${YELLOW}Checking PyTorch...${NC}"
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}')"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: PyTorch import failed${NC}"
    exit 1
fi

# Check CUDA availability
$PYTHON_CMD -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
$PYTHON_CMD -c "import torch; print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('CPU only')"

# Check if Mamba is installed
echo -e "\n${YELLOW}Checking Mamba...${NC}"
$PYTHON_CMD -c "import mamba_ssm; print('Mamba-SSM installed successfully')" 2>&1
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Mamba-SSM not installed correctly${NC}"
    echo -e "${YELLOW}You may need to install it manually${NC}"
fi

# Check other key dependencies
echo -e "\n${YELLOW}Checking other dependencies...${NC}"
$PYTHON_CMD -c "import hydra; import wandb; import numpy; import soundfile; print('All key dependencies found')"
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Some dependencies may be missing${NC}"
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "1. Prepare your dataset (see README.md)"
echo "2. Run training: python train.py dset=<dset-name> experiment=<experiment-name>"
echo "3. For testing: python test.py dset=<dset-name> experiment=<experiment-name>"
echo ""
echo -e "${CYAN}For detailed migration information, see MIGRATION_GUIDE.md${NC}"
echo ""
