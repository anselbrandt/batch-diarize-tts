import argparse
import logging
import os
import re
import subprocess
import time

import faster_whisper
import torch

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel

from helpers import (
    cleanup,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    write_srt,
)

from utils import saveToPkl

start_time = time.time()

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("logs.txt"), stream_handler],
)

mtypes = {"cpu": "int8", "cuda": "float16"}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)

args = parser.parse_args()

ROOT = os.getcwd()

batch_size = 4
model_name = "large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"

language = "en"
audio_filepath = args.audio
audio_filename = os.path.basename(audio_filepath)
audio_subdir = os.path.split(os.path.dirname(audio_filepath))[1]
temp_path = os.path.join(ROOT, "temp")
generated_rttm_filepath = os.path.join(
    temp_path, "pred_rttms", audio_filename.replace(".wav", ".rttm")
)
out_dir = os.path.join(ROOT, "output", audio_subdir)
os.makedirs(out_dir, exist_ok=True)
txt_out_path = os.path.join(out_dir, audio_filename.replace(".wav", ".txt"))
srt_out_path = os.path.join(out_dir, audio_filename.replace(".wav", ".srt"))
pkl_out_path = os.path.join(out_dir, audio_filename.replace(".wav", ".pkl"))


nemo_process = subprocess.Popen(
    ["python3", "nemo_process.py", "-a", audio_filepath, "--device", device],
    stderr=subprocess.PIPE,
)
# Transcribe the audio file

whisper_model = faster_whisper.WhisperModel(
    model_name, device=device, compute_type=mtypes[device]
)
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
audio_waveform = faster_whisper.decode_audio(audio_filepath)
suppress_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)

transcript_segments, info = whisper_pipeline.transcribe(
    audio_waveform,
    language,
    suppress_tokens=suppress_tokens,
    batch_size=batch_size,
)

full_transcript = "".join(segment.text for segment in transcript_segments)

# clear gpu vram
del whisper_model, whisper_pipeline
torch.cuda.empty_cache()


# Forced Alignment
alignment_model, alignment_tokenizer = load_alignment_model(
    device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)

emissions, stride = generate_emissions(
    alignment_model,
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device),
    batch_size=batch_size,
)

del alignment_model
torch.cuda.empty_cache()

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language="en",
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)

spans = get_spans(tokens_starred, segments, blank_token)

word_timestamps = postprocess_results(text_starred, spans, stride, scores)


# Reading timestamps <> Speaker Labels mapping

nemo_return_code = nemo_process.wait()
nemo_error_trace = nemo_process.stderr.read()
assert nemo_return_code == 0, (
    "Diarization failed with the following error:"
    f"\n{nemo_error_trace.decode('utf-8')}"
)

speaker_ts = []
with open(generated_rttm_filepath, "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

# restoring punctuation in the transcript to help realign the sentences
punct_model = PunctuationModel(model="kredor/punctuate-all")

words_list = list(map(lambda x: x["word"], wsm))

labeled_words = None
chunk_size = 230
while labeled_words is None:
    try:
        labeled_words = punct_model.predict(words_list, chunk_size)
    except:
        chunk_size = chunk_size - 10
        logging.info(
            f"{audio_subdir}/{audio_filename}|Retrying with chunk size {chunk_size}"
        )
        pass

ending_puncts = ".?!"
model_puncts = ".,;:!?"

# We don't want to punctuate U.S.A. with a period. Right?
is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

for word_dict, labeled_tuple in zip(wsm, labeled_words):
    word = word_dict["word"]
    if (
        word
        and labeled_tuple[1] in ending_puncts
        and (word[-1] not in model_puncts or is_acronym(word))
    ):
        word += labeled_tuple[1]
        if word.endswith(".."):
            word = word.rstrip(".")
        word_dict["word"] = word

wsm = get_realigned_ws_mapping_with_punctuation(wsm)

saveToPkl(pkl_out_path, wsm)

ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

with open(txt_out_path, "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(srt_out_path, "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

execution_time = time.time() - start_time

logging.info(f"{audio_subdir}/{audio_filename}|{execution_time}")

cleanup(temp_path)
