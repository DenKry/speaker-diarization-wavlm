---
license: mit
language:
- en
library_name: transformers
pipeline_tag: voice-activity-detection
base_model: microsoft/wavlm-base-plus
datasets:
- diarizers-community/ami
tags:
- speaker-diarization
- speaker-segmentation
- wavlm
- pyannote
- audio
metrics:
- der
---

# WavLM speaker diarization (AMI)

Who spoke when? This is a speaker segmentation model: [WavLM Base+](https://huggingface.co/microsoft/wavlm-base-plus)
fine-tuned on the AMI meeting corpus, with a BiLSTM head. Its `diarize()` method runs it inside
the [pyannote.audio](https://github.com/pyannote/pyannote-audio) pipeline and returns the
speech turns of a complete recording.

On the AMI test set it reaches a diarization error rate of **14.1 %** (no collar, overlap
included), against 17.7 % for the pyannote 3.1 pipeline evaluated in the same way.

Training and evaluation code: [github.com/DenKry/speaker-diarization-wavlm](https://github.com/DenKry/speaker-diarization-wavlm).

## Usage

```bash
pip install torch transformers pyannote.audio soundfile
```

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("denkry/wavlm-diarization-ami", trust_remote_code=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")

diarization = model.diarize("meeting.wav")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:7.1f}s - {turn.end:7.1f}s  {speaker}")
```

```
    0.5s -     4.2s  SPEAKER_00
    4.0s -     9.8s  SPEAKER_01
    ...
```

- `trust_remote_code=True` is needed because the model class is stored in this repository
  (`modeling_wavlm_diarization.py`); no other package has to be installed.
- The audio can have any length and sampling rate. A waveform tensor also works:
  `model.diarize(waveform, sample_rate=16000)`.
- The number of speakers is found automatically. If you know it, pass `num_speakers=4`
  (or `min_speakers` / `max_speakers`).
- Save the result in the standard RTTM format:
  `with open("meeting.rttm", "w") as f: diarization.write_rttm(f)`.
- The first call downloads the speaker embedding model
  [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM).

A GPU is recommended: about 6 to 7 minutes per hour of audio on a laptop RTX 3050, much slower
on a CPU.

### Frame-level output only

Without pyannote.audio the model can still be used as a plain segmentation model:

```python
logits = model(waveform_16khz).logits      # (batch, frames, 7), one frame every 20 ms
```

The 7 classes are: nobody, speaker 1, speaker 2, speaker 3, speakers 1+2, 1+3 and 2+3. The
speaker order is arbitrary inside each 10-second chunk.

## How it works

1. **Segmentation (this model).** 10 s of audio go through WavLM; a learned weighted sum of
   its 13 layer outputs feeds a 2-layer BiLSTM, which predicts one of 7 "powerset" classes for
   every 20 ms frame (up to 3 speakers per chunk, up to 2 at the same time).
2. **Whole recording (pyannote.audio).** The model slides over the recording with a 1 s step,
   a voice embedding is computed for every local speaker, and agglomerative clustering joins
   them into global speakers. The settings are those of `pyannote/speaker-diarization-3.1`.

## Results

Complete recordings, number of speakers not given, in %. DER with no collar and overlapping
speech included; "collar" leaves out 0.25 s around every reference boundary; JER is the
Jaccard error rate.

| Test data | DER | Missed | False alarm | Confusion | DER (collar) | JER |
|---|---|---|---|---|---|---|
| AMI test, headset mix (16 meetings) | 14.14 | 8.30 | 2.59 | 3.25 | 9.09 | 20.43 |
| VoxConverse test (29 recordings) | 12.67 | 4.27 | 6.32 | 2.08 | 8.40 | 33.89 |

The pyannote 3.1 pipeline, evaluated in the same way, gets 17.71 on AMI and 9.08 on these
VoxConverse recordings.

## Limitations

- **Trained on AMI only.** The model is specialised on meetings recorded with headset
  microphones. On other kinds of audio (broadcast, phone calls, noisy recordings),
  general-purpose systems trained on many corpora are better, as the VoxConverse row shows.
- At most two speakers can be detected at the same moment, and three per 10-second window.
- English meeting speech only was seen in fine-tuning.

## Training

- Data: AMI train (136 meetings, 81 h), version
  [diarizers-community/ami](https://huggingface.co/datasets/diarizers-community/ami), `ihm`.
- Two steps: first the head on top of a frozen WavLM (10 epochs), then WavLM and the head
  together (6 epochs, learning rate 1e-5 for WavLM and 1e-4 for the head; the convolutional
  feature encoder stays frozen).
- Random 10 s chunks, random gain and added noise, permutation-invariant powerset
  cross-entropy, AdamW, mixed precision, batch size 4 on a 6 GB GPU.

## Acknowledgements

- WavLM: Chen et al., 2022, [arXiv:2110.13900](https://arxiv.org/abs/2110.13900).
- pyannote.audio (Bredin, 2023) and the powerset loss (Plaquet & Bredin, 2023,
  [arXiv:2310.13025](https://arxiv.org/abs/2310.13025)).
- WeSpeaker speaker embeddings: Wang et al., 2023, [arXiv:2210.17016](https://arxiv.org/abs/2210.17016).
- AMI Meeting Corpus (CC BY 4.0): Carletta et al., 2006.
