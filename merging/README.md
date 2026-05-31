# Drone Map Merge

Local-only web app for:

- uploading and previewing a drone video
- selecting the start time and trimming the earlier part
- running LingBot-MAP on the trimmed video
- viewing raw LingBot output with the demo-style Viser viewer
- viewing the `translation_only_damped.pcd` postprocess output with `visualize_predictions.py`
- running TriMamba video summarization and previewing the summary mp4
- searching the trimmed video with Lighthouse `clip_slowfast + qd_detr`
- inspecting summary score/segment analysis when `trimamba_features.npz` is available
- selecting, stopping, and deleting previous local jobs

## Run

```bash
cd merging
python server.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

Outputs are saved under:

```text
merging/data/jobs/<job_id>/
```

## Configuration

The server derives paths from the repository layout. Override them with environment variables when your local setup differs:

```bash
export DRONE_PROJECT_ROOT=/path/to/graduate_project
export CONDA_ROOT=/path/to/miniconda3
export LINGBOT_MODEL_PATH=/path/to/lingbot-map-long.pt
export TRIMAMBA_CKPT=/path/to/best_model_ckpt.pth
```

Pipeline steps run in separate conda environments:

```text
LingBot-MAP       lingbot-map
TriMamba          triple
Video Retrieval   retrieval
```

You can also override individual interpreters:

```bash
export LINGBOT_PYTHON=/path/to/envs/lingbot-map/bin/python
export TRIMAMBA_PYTHON=/path/to/envs/triple/bin/python
export RETRIEVAL_PYTHON=/path/to/envs/retrieval/bin/python
```

TriMamba mp4 summarization uses `transformers` model encoders. If those dependencies or cached encoder weights are missing, the web app keeps the LingBot outputs and creates a lightweight OpenCV fallback summary so the summary player is still usable. The log will still show that the real TriMamba pass failed.
