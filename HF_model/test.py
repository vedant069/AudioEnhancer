import os
import sys
import torch
import numpy as np
import soundfile as sf

# Version check
print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

try:
    from transformers import AutoConfig
    from xcodec2.modeling_xcodec2 import XCodec2Model
    
    # Check CUDA availability and version
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        device = torch.device('cuda')
    else:
        print("CUDA not available, using CPU")
        device = torch.device('cpu')

    model_path = "HKUSTAudio/xcodec2"  
    model = XCodec2Model.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Check if input file exists
    input_file = "recorded_audio.wav"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")

    wav, sr = sf.read(input_file)   
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)
    wav_tensor = wav_tensor.to(device)

    with torch.no_grad():
        vq_code = model.encode_code(input_waveform=wav_tensor)
        print(f"Code shape: {vq_code.shape}")  
        recon_wav = model.decode_code(vq_code).cpu()

    sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), sr)
    print(f"Done! Check reconstructed.wav (Using device: {device})")

except ImportError as e:
    print(f"Import Error: {str(e)}")
    print("Please check if all required packages are installed correctly")
except Exception as e:
    print(f"Error: {str(e)}")