# Validation report

Checks performed while assembling this package:
- Python syntax parsed for the uploaded application.
- Synthetic demo CSV generated with explicit `SYNTHETIC_DEMO_ONLY` markers.
- Offline package validation tests included.

Not tested in this packaging environment:
- Webcam hardware access and face-recognition accuracy.
- Microphone device access or Windows audio permissions.
- Running Streamlit interactively on the user's laptop.
- Correctness of the user's existing database or migration between folders.

Run the included checks from the package folder:

```powershell
python -m unittest discover -s tests -v
```

These tests validate package structure, syntax and synthetic data labeling; they do not certify the hardware pipeline or validate recognition/engagement accuracy.
