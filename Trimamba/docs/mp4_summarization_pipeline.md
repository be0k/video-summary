# MP4 Summarization Pipeline

This project trains and evaluates TriMamba on pre-extracted timestep-level
features, not raw videos. The single-video pipeline is therefore:

```text
input.mp4
  -> 1 Hz visual/text/audio feature extraction
  -> TriMamba timestep importance prediction
  -> shot selection under 15% budget
  -> output_summary.mp4
```

## What The BMVC PDF Specifies

The PDF's "Feature Preparation" section states that all methods use
timestep-level pre-extracted features so the comparison focuses on the
summarization architecture. For MoSu, the released tri-modal features are:

- Visual: CLIP features
- Text: RoBERTa representations of time-stamped transcripts
- Audio: AST features

The local dataset README adds the missing operational detail used by the HDF5
files in this repository: every modality has shape `(N, 768)`, where `N`
corresponds to the video duration in seconds. In other words, the MoSu-style
input is aligned at roughly 1 feature vector per second.

The PDF also states that AST audio features are extracted from a local temporal
window `[t - 5, t + 5]` around the current timestep `t`.

For datasets without transcripts, the PDF says captions are generated with
Qwen-2.5-VL-7B-Instruct and encoded with the same RoBERTa text encoder. For an
arbitrary mp4, this script instead supports:

- a timestamped transcript JSON,
- Whisper ASR,
- or zero text vectors for missing-modality inference.

## Implemented Script

Use:

```bash
python scripts/summarize_mp4.py \
  --input path/to/input.mp4 \
  --output path/to/summary.mp4 \
  --config configs/mosu.yaml \
  --ckpt checkpoints/best_model_ckpt_mosu.pth \
  --feature-npz outputs/input_features_and_scores.npz
```

By default, the script uses zero text vectors so it can run without Whisper or
timestamped transcripts. This is a missing-modality setting, not the most
paper-faithful text preprocessing.

Useful options:

```bash
# Use Whisper ASR when openai-whisper is installed.
python scripts/summarize_mp4.py \
  --input input.mp4 \
  --output summary.mp4 \
  --text-source whisper

# Use a known transcript instead of ASR.
python scripts/summarize_mp4.py \
  --input input.mp4 \
  --output summary.mp4 \
  --text-source transcript \
  --transcript transcript.json

# Run without text if transcripts/Whisper are unavailable.
python scripts/summarize_mp4.py \
  --input input.mp4 \
  --output summary.mp4 \
  --text-source zero

# Fall back to zero vectors if audio or ASR fails.
python scripts/summarize_mp4.py \
  --input input.mp4 \
  --output summary.mp4 \
  --allow-missing-text \
  --allow-missing-audio
```

Transcript JSON format:

```json
[
  {"start": 0.0, "end": 3.4, "text": "spoken sentence"},
  {"start": 3.4, "end": 6.1, "text": "next sentence"}
]
```

## Model Choices For 768-D Features

The script uses these defaults:

- Visual: `openai/clip-vit-large-patch14`
  - `CLIPModel.get_image_features()` returns 768-d projected image features.
- Text: `roberta-base`
  - `pooler_output` is 768-d.
- Audio: `MIT/ast-finetuned-audioset-10-10-0.4593`
  - AST hidden/pooler output is 768-d.

These are practical matches for the paper's stated CLIP/RoBERTa/AST protocol.
The exact released MoSu feature extraction code is not included in this repo or
the PDF, so perfect bitwise reproduction is not guaranteed.

## Shot Selection And MP4 Export

TriMamba outputs a score for each 1 Hz timestep. The script computes shot
boundaries with a lightweight HSV histogram difference method over sampled
frames, then selects shots using the existing knapsack solver under the standard
15% summary budget. The selected time spans are cut from the original video with
ffmpeg and concatenated into the output mp4.

For more stable research-grade shot boundaries, replace the built-in histogram
detector with PySceneDetect/KTS and feed those boundaries into the same
selection/export stage.

## Environment Notes

Required for end-to-end inference:

- `mamba_ssm`, because `enc_dec/blocks.py` imports `Mamba`.
- `transformers`
- `opencv-python`
- `pillow`
- `PyYAML`
- `ffmpeg`
- optional: `openai-whisper` for `--text-source whisper`

This current workspace has `transformers`, `torch`, `opencv`, and `ffmpeg`, but
`mamba_ssm` is not installed, so model inference will fail until that package is
available in the active environment.
