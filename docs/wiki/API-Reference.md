# API Reference

The local server is implemented in `merging/server.py`. It serves both the browser UI and REST endpoints.

Base URL:

```text
http://127.0.0.1:7860
```

## Jobs

### `GET /api/jobs`

Returns all known jobs.

```bash
curl http://127.0.0.1:7860/api/jobs
```

### `GET /api/jobs/{job_id}`

Returns one job status, including current artifacts.

```bash
curl http://127.0.0.1:7860/api/jobs/<job_id>
```

### `DELETE /api/jobs/{job_id}?force=1`

Deletes a job. Use `force=1` to stop and delete a running job.

## Upload and Run

### `POST /api/upload`

Upload a local video using multipart form data.

```bash
curl -F "video=@sample.mp4" http://127.0.0.1:7860/api/upload
```

### `POST /api/jobs/{job_id}/run`

Run trimming, LingBot-MAP, postprocessing, TriMamba, and artifact refresh.

```bash
curl -X POST http://127.0.0.1:7860/api/jobs/<job_id>/run \
  -H "Content-Type: application/json" \
  -d '{
    "start_seconds": 0,
    "end_seconds": 120,
    "lingbot_fps": 5,
    "summary_ratio": 0.15,
    "max_points": 800000
  }'
```

### `POST /api/jobs/{job_id}/stop`

Requests cancellation for a running job.

## Viewers

### `POST /api/jobs/{job_id}/view/raw`

Starts the LingBot demo-style Viser viewer from `lingbot_raw.npz`.

### `POST /api/jobs/{job_id}/view/post`

Starts the postprocessed point cloud viewer from `translation_only_damped.pcd`.

Both endpoints return:

```json
{
  "url": "http://127.0.0.1:<port>"
}
```

## Summary Analysis

### `GET /api/jobs/{job_id}/summary/analysis`

Returns TriMamba score and selected segment data from `trimamba_features.npz`.

## Video Retrieval

### `POST /api/jobs/{job_id}/retrieval/index`

Builds a Lighthouse `clip_slowfast + qd_detr` index from `trimmed.mp4`.

```bash
curl -X POST http://127.0.0.1:7860/api/jobs/<job_id>/retrieval/index \
  -H "Content-Type: application/json" \
  -d '{}'
```

### `POST /api/jobs/{job_id}/retrieval/query`

Runs query-based moment retrieval.

```bash
curl -X POST http://127.0.0.1:7860/api/jobs/<job_id>/retrieval/query \
  -H "Content-Type: application/json" \
  -d '{"query": "person walking near the building", "top_k": 5}'
```

Response fields:

| Field | Description |
| --- | --- |
| `moments` | QD-DETR retrieved time windows |
| `saliency` | per-clip saliency scores |
| `highlights` | top highlighted frames with thumbnails |

## Media

### `GET /media/{job_id}/{relative_path}`

Serves videos, frames, point clouds, and generated artifacts. The server supports HTTP range requests for video seeking.
