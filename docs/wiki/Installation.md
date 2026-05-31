# Installation

## 1. Clone

```bash
git clone <your-repo-url>
cd <repo>
```

The expected layout is:

```text
lingbot-map/
Trimamba/
video_retrieval/
merging/
```

## 2. Create Conda Environments

Use the environment names expected by the web app:

```bash
conda create -n lingbot-map python=3.10
conda create -n triple python=3.12
conda create -n retrieval python=3.10
```

Install each component's dependencies inside its matching environment. The exact dependency set depends on your CUDA/PyTorch version.

## 3. Place Required Weights

The web app expects these files by default:

```text
lingbot-map/lingbot-map-long.pt
Trimamba/checkpoints/best_model_ckpt.pth
video_retrieval/weights/SLOWFAST_8x8_R50.pkl
video_retrieval/weights/clip_slowfast_qd_detr_qvhighlight.ckpt
```

Do not commit these weights to Git. Use release assets, Git LFS, or a separate download script if you publish the project.

## 4. Run the Local Web App

```bash
conda activate lingbot-map
cd merging
python server.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

## 5. Optional Path Overrides

Copy `.env.example` and export the variables you need:

```bash
export DRONE_PROJECT_ROOT=/path/to/repo
export CONDA_ROOT=/path/to/miniconda3
export LINGBOT_MODEL_PATH=/path/to/lingbot-map-long.pt
export TRIMAMBA_CKPT=/path/to/best_model_ckpt.pth
```
