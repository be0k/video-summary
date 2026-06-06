# Drone Mapping and Video Understanding

This project provides a local web workflow for drone video analysis:

- Upload and trim a local drone video.
- Reconstruct a raw 3D map with LingBot-MAP.
- Apply point cloud postprocessing for drift-reduced visualization.
- Summarize the trimmed video with TriMamba.
- Retrieve query-relevant moments with Lighthouse `clip_slowfast + qd_detr`.

The application is designed for local research and demonstration. Uploaded videos, generated point clouds, intermediate features, and model weights are intentionally ignored by Git.

## Repository Layout

```text
.
├── lingbot-map/          # LingBot-MAP inference and point cloud visualization
├── Trimamba/             # TriMamba video summarization
├── video_retrieval/      # Lighthouse-based moment retrieval wrapper
├── merging/              # Local web app and REST API server
├── docs/wiki/            # Wiki-ready documentation
└── .env.example          # Optional local path overrides
```

## Main Components

| Component | Purpose | Conda env |
| --- | --- | --- |
| LingBot-MAP | Video-to-3D map reconstruction | `lingbot-map` |
| TriMamba | Video summarization | `triple` |
| Lighthouse Retrieval | Query-based moment retrieval | `retrieval` |
| Merging Web App | Local API and browser UI | usually `lingbot-map` |

## Next Pages

- [Installation](Installation.md)
- [Environment Setup](Environment-Setup.md)
- [Configuration](Configuration.md)
- [API Reference](API-Reference.md)
- [Examples](Examples.md)
- [Artifacts and Weights](Artifacts-and-Weights.md)
- [Development Notes](Development-Notes.md)
