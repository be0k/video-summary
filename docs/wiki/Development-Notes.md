# Development Notes

## Trimamba Notes

The current TriMamba integration supports local mp4 summarization through `Trimamba/scripts/summarize_mp4.py`.

For an inference-only public release, keep:

```text
Trimamba/scripts/summarize_mp4.py
Trimamba/configs/mosu.yaml
Trimamba/models/
Trimamba/utils/generate_summary.py
```

Training, evaluation, benchmark datasets, and analysis scripts are ignored by the project `.gitignore` files.

Potential issues to review before a formal release:

- `Trimamba/dataset.py` builds TVSum user-score lookup keys from resampled score length. If multiple videos share the same length, later entries can overwrite earlier entries. A video-id keyed lookup would be safer.
- Several training/evaluation scripts still assume local `./data` and checkpoint layouts. They are acceptable for research scripts, but release docs should state this clearly.
- `summe` evaluation in `solver.py` has custom correlation logic that diverges from the generic metric path. It should be validated against the original benchmark protocol before claiming benchmark numbers.
- `summarize_mp4.py` is the recommended API surface for user-uploaded videos. Training datasets are not required for web inference.

## Postprocessing Notes

The web app currently uses `translation_only_damped` as the postprocess output. This is conservative and avoids aggressive loop closure rotation, but map quality still depends heavily on camera motion, overlap, and LingBot prediction stability.

## Open Source Checklist

- Keep weights and generated artifacts out of Git.
- Add release/download instructions for required weights.
- Run a short smoke test for upload, run, viewer launch, summary, and retrieval.
- Verify all default paths are repository-relative or environment-variable driven.
- Include license notices for vendored or adapted third-party code.
