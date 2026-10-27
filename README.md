# Multimodal Attendance & Engagement Agent

A local college/demo prototype for manual attendance, consent-based face enrollment, optional LBPH recognition, webcam face-presence tracking, audio RMS sampling, session analytics, CSV exports, and rule-based instructor-support suggestions.

## Quick setup (Windows PowerShell)

Open PowerShell in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`.

If you already have a working virtual environment, do not recreate it; install requirements into that environment and run Streamlit from this folder.

## Files

- `app.py` — Streamlit prototype.
- `requirements.txt` — Python dependencies.
- `sample_data/synthetic_multimodal_signals.csv` — clearly synthetic example signals; no real student records or faces.
- `tests/test_validation.py` — offline validation checks for the demo CSV and app syntax.
- `SKILL_WALLET_MAPPING.md` — activity-by-activity mapping for the six milestones.
- `VALIDATION_REPORT.md` — scope and checks performed in this packaging environment.

## Database safety — read before running

The app creates missing SQLite tables using `CREATE TABLE IF NOT EXISTS`; it does not intentionally delete existing rows. The database is created at `attendance.db` beside `app.py`. **Back up your current `attendance.db` before replacing or moving files.** This ZIP does not include your database or personal `face_data/` images. If your database is currently in a different folder, keep it safe and do not delete either copy; the app may create a new database beside this `app.py`.

## Face data and privacy

Only enroll people who have given informed consent and where institutional policy allows it. Face samples are stored locally under `face_data/`. Protect them and delete them when no longer needed. Recognition is probabilistic: manually verify identity and attendance. Do not use recognition alone for official decisions.

## Audio and engagement limitations

The microphone feature stores RMS amplitude and a rough sound-level label, not raw audio. Amplitude cannot identify speech, speakers, emotion, attention, comprehension, or learning. Face counts and movement are not validated engagement measurements. Instructor suggestions are rule-based heuristics, not a trained or scientifically validated assessment.

## Demo data

The CSV under `sample_data/` is synthetic and for testing/visualization only. It is not real attendance, actual students, or evidence of engagement. Do not upload real student personal data to a public repository.

## Troubleshooting

- **Camera does not open:** close other camera apps and allow Windows camera permission.
- **Recognition unavailable:** confirm `opencv-contrib-python` is installed in the same Python environment that runs Streamlit.
- **Microphone unavailable:** check `sounddevice` installation, device selection, OS permission, and microphone access. The app can still run without the optional audio device.
- **Import errors:** run `python -m pip install -r requirements.txt` using the same interpreter used to run Streamlit.
