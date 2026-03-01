#!/usr/bin/env bash
set -e

echo
echo "=================================================="
echo "   AI Speech Detector - Local Launcher"
echo "=================================================="
echo

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$PROJECT_DIR/venv"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
APP_FILE="$PROJECT_DIR/app.py"

# --------------------------------------------------
# Check Python
# --------------------------------------------------
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed."
    exit 1
fi

echo "[INFO] Python detected."
echo

# --------------------------------------------------
# Create virtual environment if missing
# --------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# --------------------------------------------------
# Activate environment
# --------------------------------------------------
source "$VENV_DIR/bin/activate"
echo "[INFO] Virtual environment activated."
echo

# --------------------------------------------------
# Install Python dependencies
# --------------------------------------------------
echo "[INFO] Installing Python dependencies..."
pip install --upgrade pip > /dev/null
pip install -r "$REQUIREMENTS"

echo

# --------------------------------------------------
# Install FFmpeg (Linux/macOS)
# --------------------------------------------------
if ! command -v ffmpeg &> /dev/null; then
    echo "[INFO] FFmpeg not found. Attempting installation..."

    OS="$(uname)"

    if [ "$OS" = "Linux" ]; then
        if command -v apt &> /dev/null; then
            sudo apt update && sudo apt install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm ffmpeg
        else
            echo "[WARNING] Could not detect package manager. Please install ffmpeg manually."
        fi

    elif [ "$OS" = "Darwin" ]; then
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "[WARNING] Homebrew not found. Install Homebrew and run:"
            echo "brew install ffmpeg"
        fi
    fi

else
    echo "[INFO] FFmpeg detected."
fi

echo
echo "=================================================="
echo "     Starting Streamlit Application"
echo "=================================================="
echo

streamlit run "$APP_FILE"
