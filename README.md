# Podcast Diarization with Faster Whisper - For TTS Training Data (using clean audio with overlapped speech removed)

### Installation

```
pip install -c constraints.txt -r requirements.txt
```

To fix `"resume_download" is deprecated` error:
```
pip install --upgrade transformers
```

### Usage

```
python batch.py
```

##### Input files folder structure
```
audio
├──<showname>
    ├──001 - <episode name>.mp3
    ├──002 - <episode name>.mp3
├──<showname>
    ├──001 - <episode name>.mp3
    ├──002 - <episode name>.mp3
```

Make sure the following is in path in `.bashrc`:

```
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:$HOME/.pyenv/versions/3.12.7/envs/<python virtual env name>/lib64/python3.12/site-packages/nvidia/cudnn/lib:$HOME/.pyenv/versions/3.12.7/envs/<python virtual env name>/lib64/python3.12/site-packages/nvidia/cublas/lib
```

Path variables will be different if not using [pyenv](https://github.com/pyenv/pyenv)

To disable console warning from `transformers`, add the following env variables:

```
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
```