# Drone Mapping and Video Understanding

Local web application for turning drone videos into a compact analysis workspace:

- trim uploaded video clips,
- reconstruct raw 3D maps with LingBot-MAP,
- view postprocessed point clouds,
- summarize video with TriMamba,
- retrieve query-relevant moments with Lighthouse `clip_slowfast + qd_detr`.

The project is intended for local research/demo use. Model weights, uploaded videos, generated point clouds, retrieval indexes, and summary artifacts are not committed to Git.

## Repository Layout

```text
.
├── lingbot-map/          # LingBot-MAP inference and point cloud viewers
├── Trimamba/             # TriMamba inference wrapper and model code
├── video_retrieval/      # Lighthouse retrieval wrapper
├── merging/              # Local web UI and API server
├── docs/wiki/            # GitHub Wiki-ready documentation
├── .env.example          # Optional local path overrides
└── .gitignore
```

## Required Conda Environments

The web app runs each model family in its own environment:

```text
lingbot-map   LingBot-MAP reconstruction and viewers
triple        TriMamba video summarization
retrieval     Lighthouse moment retrieval
```

Set `CONDA_ROOT` or individual interpreter variables if your environment paths differ. See [.env.example](.env.example).

## Required Weights

Place runtime weights locally. Do not commit them.

```text
lingbot-map/lingbot-map-long.pt
Trimamba/checkpoints/best_model_ckpt.pth
video_retrieval/weights/SLOWFAST_8x8_R50.pkl
video_retrieval/weights/clip_slowfast_qd_detr_qvhighlight.ckpt
```

## Quick Start

```bash
conda activate lingbot-map
cd merging
python server.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

## API Example

```bash
JOB_ID=$(curl -s -F "video=@sample.mp4" http://127.0.0.1:7860/api/upload \
  | python -c "import sys,json; print(json.load(sys.stdin)['id'])")

curl -X POST "http://127.0.0.1:7860/api/jobs/$JOB_ID/run" \
  -H "Content-Type: application/json" \
  -d '{"start_seconds": 0, "end_seconds": 120, "lingbot_fps": 5, "summary_ratio": 0.15}'
```


## Notes

- Generated job data is stored under `merging/data/jobs/` by default.
- Postprocessing currently uses `translation_only_damped`.
- TriMamba is published here as an inference-focused integration; training/evaluation scripts are ignored for the public release surface.
- Lighthouse source needed for retrieval is vendored under `video_retrieval/lighthouse`; required retrieval weights live in `video_retrieval/weights`.
