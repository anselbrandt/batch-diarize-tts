import argparse
import os
import logging

import torch

from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import create_config

logging.getLogger("nemo_logger").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)
args = parser.parse_args()

audio_path = args.audio

ROOT = os.getcwd()

temp_path = os.path.join(ROOT, "temp")

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path, audio_path)).to(args.device)
msdd_model.diarize()
