# Environment Setup

This project intentionally uses separate conda environments because the three model stacks have different dependency constraints.

| Environment | Used by | Main purpose |
| --- | --- | --- |
| `lingbot-map` | `merging/server.py`, `lingbot-map/` | Web server, LingBot-MAP inference, raw/postprocessed point cloud viewers |
| `triple` | `Trimamba/scripts/summarize_mp4.py` | TriMamba video summarization and score analysis |
| `retrieval` | `video_retrieval/retrieval.py` | Lighthouse `clip_slowfast + qd_detr` indexing and query retrieval |

The environment names above are the defaults expected by the local web app. If you use different names, set the explicit Python interpreter variables described in [Configuration](Configuration.md).

## 0. System Packages

Install these once on the host machine:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg build-essential
```

`ffmpeg` is required for trimming uploaded videos, writing summary videos, and loading videos for retrieval.

## 1. Repository Layout

The expected layout is:

```text
graduate_project/
├── lingbot-map/
├── Trimamba/
├── video_retrieval/
├── merging/
├── docs/
├── .env.example
└── README.md
```

Run the following commands from `graduate_project/` unless otherwise noted.

## 2. LingBot-MAP Environment

Create the environment:

```bash
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map
```

Install PyTorch for your CUDA version. Example for CUDA 12.8:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

Install LingBot-MAP and visualization/postprocessing dependencies:

```bash
pip install -e ./lingbot-map
pip install -e "./lingbot-map[vis]"
pip install open3d
```

Optional but recommended for LingBot streaming performance:

```bash
pip install --index-url https://pypi.org/simple flashinfer-python
```

Verify:

```bash
python -c "import torch, cv2, open3d; import lingbot_map; print('lingbot-map ok', torch.cuda.is_available())"
```

This environment also runs the local web server:

```bash
conda activate lingbot-map
python merging/server.py --host 127.0.0.1 --port 7860
```

## 3. TriMamba Environment

The web app expects this environment to be named `triple`.

```bash
conda create -n triple python=3.10 -y
conda activate triple
```

Install PyTorch. Example for CUDA 12.1, matching the upstream TriMamba test setup:

```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install runtime dependencies used by `scripts/summarize_mp4.py` and summary analysis:

```bash
pip install opencv-python pillow numpy pyyaml scipy scikit-learn tqdm h5py matplotlib thop wandb
pip install transformers accelerate
pip install mamba-ssm causal-conv1d
```

If `mamba-ssm` or `causal-conv1d` fails to build, install versions compatible with your CUDA/PyTorch pair. This is the most common TriMamba environment issue.

Verify:

```bash
python -c "import torch, cv2, yaml; from PIL import Image; from transformers import CLIPModel; from mamba_ssm import Mamba; print('triple ok', torch.cuda.is_available())"
```

Required checkpoint location:

```text
Trimamba/checkpoints/best_model_ckpt.pth
```

The web app calls TriMamba like this:

```bash
conda activate triple
cd Trimamba
python scripts/summarize_mp4.py \
  --input ../merging/data/jobs/<job_id>/trimmed.mp4 \
  --output ../merging/data/jobs/<job_id>/trimamba_summary.mp4 \
  --config configs/mosu.yaml \
  --ckpt checkpoints/best_model_ckpt.pth \
  --text-source zero \
  --allow-missing-text \
  --allow-missing-audio
```

## 4. Video Retrieval Environment

The web app expects this environment to be named `retrieval`.

```bash
conda create -n retrieval python=3.10 -y
conda activate retrieval
```

Install PyTorch. Use the CUDA build that matches your machine:

```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install Lighthouse and retrieval dependencies:

```bash
pip install -e ./video_retrieval/lighthouse
pip install opencv-python ffmpeg-python
```

If editable install cannot fetch OpenAI CLIP automatically, install it manually:

```bash
pip install git+https://github.com/openai/CLIP.git
```

The vendored Lighthouse package pins some dependencies, including `numpy<=1.23.5` and `transformers<=4.51.3`. Keep these constraints in the retrieval environment only; do not mix this environment with LingBot-MAP or TriMamba.

Verify:

```bash
python -c "import torch, ffmpeg, lighthouse; from lighthouse.models import QDDETRPredictor; print('retrieval ok', torch.cuda.is_available())"
```

Required weight locations:

```text
video_retrieval/weights/SLOWFAST_8x8_R50.pkl
video_retrieval/weights/clip_slowfast_qd_detr_qvhighlight.ckpt
```

## 5. Model Weights

Weights are intentionally ignored by Git. Place them locally:

```text
lingbot-map/lingbot-map-long.pt
Trimamba/checkpoints/best_model_ckpt.pth
video_retrieval/weights/SLOWFAST_8x8_R50.pkl
video_retrieval/weights/clip_slowfast_qd_detr_qvhighlight.ckpt
```

If you store them elsewhere, export overrides before starting the server:

```bash
export LINGBOT_MODEL_PATH=/path/to/lingbot-map-long.pt
export TRIMAMBA_CKPT=/path/to/best_model_ckpt.pth
```

## 6. Optional Path Overrides

Copy `.env.example` or export variables directly:

```bash
export DRONE_PROJECT_ROOT=/path/to/graduate_project
export CONDA_ROOT=/path/to/miniconda3
export LINGBOT_PYTHON=$CONDA_ROOT/envs/lingbot-map/bin/python
export TRIMAMBA_PYTHON=$CONDA_ROOT/envs/triple/bin/python
export RETRIEVAL_PYTHON=$CONDA_ROOT/envs/retrieval/bin/python
export MERGING_DATA_ROOT=/path/to/jobs
```

The server resolves interpreters in this order:

1. Explicit `LINGBOT_PYTHON`, `TRIMAMBA_PYTHON`, `RETRIEVAL_PYTHON`.
2. `CONDA_ROOT`.
3. `CONDA_EXE`.
4. `CONDA_PREFIX`.
5. `~/miniconda3`.

## 7. End-to-End Check

Start the web app:

```bash
conda activate lingbot-map
python merging/server.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

Upload a short video first. A short test clip is useful because it verifies all three environments without wasting GPU time.

## 8. Common Errors

| Error | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'PIL'` | `pillow` missing in `triple` | `conda activate triple && pip install pillow` |
| `ModuleNotFoundError: No module named 'transformers'` | TriMamba encoder dependencies missing | `conda activate triple && pip install transformers accelerate` |
| `ModuleNotFoundError: No module named 'mamba_ssm'` | TriMamba Mamba block dependency missing | `conda activate triple && pip install mamba-ssm causal-conv1d` |
| `No visual loop candidates found` | LingBot postprocess could not detect a reliable loop | The pipeline falls back to exporting without pose correction; use a better loop video or lower closure thresholds for experiments |
| Retrieval import/version conflict | Lighthouse pins older `numpy`/`transformers` | Keep retrieval in the isolated `retrieval` environment |
| `ffmpeg` not found | System video tool missing | Install `ffmpeg` with apt or conda-forge |

