# Examples

## End-to-End Browser Workflow

1. Start the server.
2. Upload a video.
3. Select start and end time.
4. Run the pipeline.
5. Open the LingBot raw viewer.
6. Open the postprocessed point cloud viewer.
7. Inspect the TriMamba summary.
8. Build the retrieval index and run a text query.

## Upload and Run via API

```bash
JOB_ID=$(curl -s -F "video=@sample.mp4" http://127.0.0.1:7860/api/upload | python -c "import sys,json; print(json.load(sys.stdin)['id'])")

curl -X POST "http://127.0.0.1:7860/api/jobs/$JOB_ID/run" \
  -H "Content-Type: application/json" \
  -d '{"start_seconds": 0, "end_seconds": 120, "lingbot_fps": 5, "summary_ratio": 0.15}'
```

Poll status:

```bash
curl "http://127.0.0.1:7860/api/jobs/$JOB_ID"
```

## Retrieval Query

```bash
curl -X POST "http://127.0.0.1:7860/api/jobs/$JOB_ID/retrieval/index" \
  -H "Content-Type: application/json" \
  -d '{}'

curl -X POST "http://127.0.0.1:7860/api/jobs/$JOB_ID/retrieval/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "building entrance", "top_k": 5}'
```

## Direct Lighthouse CLI

```bash
conda activate retrieval
cd video_retrieval

python retrieval.py index \
  --video ../merging/data/jobs/<job_id>/trimmed.mp4 \
  --output ../merging/data/jobs/<job_id>/lighthouse_qd_detr_index.pt

python retrieval.py query \
  --index ../merging/data/jobs/<job_id>/lighthouse_qd_detr_index.pt \
  --query "person walking near the building"
```
