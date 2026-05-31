# Artifacts and Weights

## Required Weights

These files are required at runtime but should not be committed:

```text
lingbot-map/lingbot-map-long.pt
Trimamba/checkpoints/best_model_ckpt.pth
video_retrieval/weights/SLOWFAST_8x8_R50.pkl
video_retrieval/weights/clip_slowfast_qd_detr_qvhighlight.ckpt
```

Recommended publishing options:

- GitHub Releases for model files.
- Git LFS if your hosting policy allows it.
- A separate download script with checksums.

## Generated Job Artifacts

Each job may create:

```text
input.mp4
trimmed.mp4
lingbot_raw.npz
lingbot_raw.pcd
translation_only_damped.pcd
trimamba_summary.mp4
trimamba_features.npz
lighthouse_qd_detr_index.pt
retrieval_highlights/
job.log
status.json
```

These files are ignored by Git.

## Why Artifacts Are Ignored

Generated videos, point clouds, feature files, and model checkpoints are large and machine-specific. Keeping them out of Git makes the repository cloneable and easier to review.
