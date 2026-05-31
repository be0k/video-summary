# Video Retrieval

Local Lighthouse-based moment retrieval for uploaded drone videos.

The web app calls this project from the Retrieval page:

1. `index` encodes the already-trimmed video with `clip_slowfast`.
2. `query` runs `qd_detr` for a text query.
3. The JSON result contains `moments`, `saliency`, and `highlights`.

Expected runtime environment:

```text
<conda-root>/envs/retrieval/bin/python
```

The Lighthouse source package is copied into `video_retrieval/lighthouse`, and the required model files are stored in `video_retrieval/weights`. This feature does not depend on the old top-level `lighthouse` folder.

Example:

```bash
<conda-root>/envs/retrieval/bin/python retrieval.py index \
  --video /path/to/trimmed.mp4 \
  --output /path/to/lighthouse_qd_detr_index.pt

<conda-root>/envs/retrieval/bin/python retrieval.py query \
  --index /path/to/lighthouse_qd_detr_index.pt \
  --query "person walking near the building"
```
