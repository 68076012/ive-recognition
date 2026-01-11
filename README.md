# IVE Face Recognition (InsightFace + Tracking)

Face recognition + simple tracking for IVE members using InsightFace. A YouTube interview is downloaded (or a local MP4 is used), faces are detected/identified against a reference gallery, and an annotated video is written to `outputs/`.

## Project layout
- `ive_face_recognition.ipynb` — end‑to‑end notebook (deps install, gallery build, recognition + tracking, video export)
- `ive_reference/` — per‑member reference images used to build the gallery
- `outputs/` — sample videos and generated results (ignored by git)

## Requirements
- Python 3.9–3.11 (tested with CPU fallback; CUDA used when available)
- Dependencies (same as notebook cell):
  - `insightface==0.7.3`
  - `onnxruntime-gpu==1.17.1` (or `onnxruntime==1.17.1` for CPU-only)
  - `opencv-python==4.8.1.78`
  - `scipy==1.10.1`
  - `filterpy==1.4.5`
  - `pytube==15.0.0`
  - `yt-dlp==2024.7.16`
  - `tqdm>=4.66.1`

## Quickstart (recommended flow)
```bash
# from project_recognition/
python -m venv .venv_ive
.venv_ive\Scripts\activate   # or source .venv_ive/bin/activate on mac/linux
pip install insightface==0.7.3 onnxruntime-gpu==1.17.1 opencv-python==4.8.1.78 scipy==1.10.1 filterpy==1.4.5 pytube==15.0.0 yt-dlp==2024.7.16 "tqdm>=4.66.1"

jupyter notebook ive_face_recognition.ipynb
```
- In the notebook, confirm `Reference dir` and `Output dir` point to this folder.
- Set `default_video_url` (YouTube) or drop a local MP4 and update `local_video_fallback`.
- Run cells: deps check → gallery build → recognition/tracking → export. Outputs land in `outputs/`.

## Notes on data
- Reference gallery uses folders under `ive_reference/<Member_Name>/`. Add or replace images to improve recognition; the notebook averages embeddings per member.
- The tracker blends IoU + embedding similarity for stability; thresholds are adjustable in `SimpleTracker`.

## Git setup (optional)
If you want this subproject in its own repository:
```bash
git init
git add README.md ive_face_recognition.ipynb ive_reference
git commit -m "Initial commit: IVE face recognition"
```
`outputs/`, virtualenvs, and checkpoints are ignored by the provided `.gitignore`.

## Troubleshooting
- If CUDA is present but failing, set `force_cpu=True` in `init_face_analysis`.
- For slow runs, reduce frame processing rate or resolution in the notebook before exporting.
