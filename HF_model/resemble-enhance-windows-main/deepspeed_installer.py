import os
import subprocess
import sys

import importlib.metadata as metadata  # Use importlib.metadata
from packaging import version
from loguru import logger

def install_package(package_link):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_link])

def is_package_installed(package_name):
    try:
        metadata.version(package_name)
        return True
    except metadata.PackageNotFoundError:
        return False

def check_and_install_torch():
    required_torch_version = 'torch==2.1.1+cu118 torchaudio==2.1.1+cu118'

    # Check if torch with CUDA 11.8 is installed.
    if not any(required_torch_version in pkg for pkg in metadata.distributions()):
        logger.info(f"'{required_torch_version}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", required_torch_version,"--index-url https://download.pytorch.org/whl/cu118"])
    else:
        logger.info(f"'{required_torch_version}' already installed.")

def install_deepspeed_based_on_python_version():
    # check_and_install_torch()
    if not is_package_installed('deepspeed'):
        python_version = sys.version_info

        logger.info(f"Python version: {python_version.major}.{python_version.minor}")

        # Define your package links here
        py310_win = "https://github.com/daswer123/xtts-webui/releases/download/deepspeed/deepspeed-0.11.2+cuda118-cp310-cp310-win_amd64.whl"
        py311_win = "https://github.com/daswer123/xtts-webui/releases/download/deepspeed/deepspeed-0.11.2+cuda118-cp311-cp311-win_amd64.whl"

        # Use generic pip install deepspeed for Linux or custom wheels for Windows.
        deepspeed_link = None

        if sys.platform == 'win32':
            if python_version.major == 3 and python_version.minor == 10:
                deepspeed_link = py310_win

            elif python_version.major == 3 and python_version.minor == 11:
                deepspeed_link = py311_win

            else:
                logger.error("Unsupported Python version on Windows.")

        else: # Assuming Linux/MacOS otherwise (add specific checks if necessary)
             deepspeed_link = 'deepspeed==0.11.2'

        if deepspeed_link:
             logger.info("Installing DeepSpeed...")
             install_package(deepspeed_link)

    # else:
        #  logger.info("'deepspeed' already installed.")


if __name__ == "__main__":
    install_deepspeed_based_on_python_version()