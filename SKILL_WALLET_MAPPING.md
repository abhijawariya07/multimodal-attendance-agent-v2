# Skill Wallet mapping — Multimodal Attendance & Engagement Agent

Use this as an evidence checklist. An activity should only be marked complete in Skill Wallet after you have performed the corresponding check and captured evidence.

| Milestone / activity | Current project evidence | Remaining validation / suggested evidence |
|---|---|---|
| 1.1 Requirements specification and `requirements.txt` | `requirements.txt` lists Streamlit, OpenCV-contrib, Pandas, NumPy and SoundDevice. | Save screenshot of dependency file and successful dependency install. |
| 1.2 Environment setup | README includes Windows virtual-environment and launch commands. | Run import checks in the actual laptop environment and save terminal screenshot. |
| 2.1 Face detection and tracking | App uses OpenCV Haar cascade, temporary IDs and face-count logging during webcam sessions. | Test webcam/prerecorded video if available; capture sample output and note tracking limitations. |
| 2.2 Presence-based attendance | App records manual attendance; optional candidate recognition can record attendance when enabled. Temporary tracks are confirmed after 3 seconds. | Test short/intermittent appearances and duplicate same-day attendance. Recognition must be manually verified. |
| 3.1 Audio signal extraction | App records a short microphone window at 16 kHz mono and calculates RMS. | Test with microphone permission and capture RMS output; optional device may not exist in CI. |
| 3.2 Noise / activity analysis | App labels sound levels using RMS thresholds. | Test quiet and louder environments; describe this as sound activity, not speech recognition. |
| 4.1 Multimodal agent implementation | Streamlit app brings together vision, optional audio, attendance and reporting tabs. | Run the full app locally and capture a successful launch screenshot. |
| 4.2 Multimodal signal fusion | Current prototype displays separate vision and audio signal summaries; it does not implement a synchronized fusion model. | Further implementation is needed for timestamp-aligned fusion across modalities. Do not claim this is complete yet. |
| 4.3 Engagement scoring | No validated engagement score is implemented. The app provides signal-based heuristic suggestions only. | Further implementation needed for an interval-based, explainable experimental score; do not call sensor signals actual attention. |
| 5.1 Teaching suggestion logic | Rule-based camera/audio troubleshooting suggestions are shown. | Test rules with synthetic and live signal examples; keep wording as suggestions, not teaching-quality claims. |
| 5.2 Student session analytics | Attendance summary, face-count trend, audio RMS trend and session reports are available. | Validate summary calculations against known sample data; session-level per-student engagement trends are not currently implemented. |
| 6.1 Run core application | README provides run command. | Run from the project directory with webcam and optional microphone; capture evidence. |
| 6.2 Streamlit dashboard | Streamlit dashboard includes attendance, enrollment, tracking, audio, analytics and suggestions. | Capture screenshots of each relevant tab after successful run. |
| 6.3 Validate outputs / edge cases | Test checklist and synthetic demo CSV included. | Run `python -m unittest discover -s tests -v`, then test lighting, occlusion, no camera, no mic, duplicate attendance and intermittent presence locally. |

## Important distinction

This repository is a working prototype, not a completed implementation of every aspirational milestone. In particular, timestamp-synchronized multimodal fusion and an explainable interval engagement score are not currently present. Mark Skill Wallet items according to evidence from actual execution, not this checklist alone.
