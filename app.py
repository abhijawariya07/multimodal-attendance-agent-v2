import math
import sqlite3
import time
from datetime import datetime, date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    sd = None
    AUDIO_AVAILABLE = False

try:
    # Requires opencv-contrib-python. The app still runs without it.
    LBPH_AVAILABLE = hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")
except Exception:
    LBPH_AVAILABLE = False

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "attendance.db"
FACE_DIR = APP_DIR / "face_data"
FACE_DIR.mkdir(exist_ok=True)
CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", page_icon="🎓", layout="wide")


# ----------------------------- Database -----------------------------
def db():
    return sqlite3.connect(DB_PATH, timeout=15)


def init_db():
    """Create missing tables without deleting or recreating existing data."""
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            student_id TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS engagement_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_time TEXT NOT NULL,
            faces_detected INTEGER NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS audio_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_time TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            rms_level REAL NOT NULL,
            activity_label TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS session_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_time TEXT NOT NULL,
            unique_tracks INTEGER NOT NULL,
            confirmed_tracks INTEGER NOT NULL,
            face_samples INTEGER NOT NULL,
            session_duration REAL NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            student_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS recognition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_time TEXT NOT NULL,
            student_id TEXT,
            predicted_name TEXT NOT NULL,
            confidence REAL,
            result TEXT NOT NULL
        )""")


def read_df(query, params=()):
    with db() as con:
        return pd.read_sql_query(query, con, params=params)


def save_attendance(name, sid, source="Manual"):
    name, sid = name.strip(), sid.strip()
    if not name or not sid:
        return False, "Enter both student name and student ID."
    today = date.today().isoformat()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db() as con:
        exists = con.execute(
            "SELECT 1 FROM attendance WHERE student_id=? AND timestamp LIKE ? LIMIT 1",
            (sid, today + "%")
        ).fetchone()
        if exists:
            return False, "Attendance for this student is already recorded today."
        con.execute(
            "INSERT INTO attendance(student_name, student_id, timestamp) VALUES(?,?,?)",
            (name, sid, now)
        )
    return True, "Attendance saved."


def save_face_sample(count):
    with db() as con:
        con.execute("INSERT INTO engagement_logs(session_time, faces_detected) VALUES(?,?)",
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), int(count)))


def save_audio_sample(seconds, rms, label):
    with db() as con:
        con.execute(
            "INSERT INTO audio_logs(session_time,duration_seconds,rms_level,activity_label) VALUES(?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), float(seconds), float(rms), label)
        )


def save_session_report(unique, confirmed, samples, duration):
    with db() as con:
        con.execute(
            "INSERT INTO session_reports(session_time,unique_tracks,confirmed_tracks,face_samples,session_duration) VALUES(?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), int(unique), int(confirmed), int(samples), float(duration))
        )


def log_recognition(sid, name, confidence, result):
    with db() as con:
        con.execute(
            "INSERT INTO recognition_logs(session_time,student_id,predicted_name,confidence,result) VALUES(?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), sid, name, confidence, result)
        )


def load_attendance():
    return read_df("""SELECT student_name AS 'Student Name', student_id AS 'Student ID',
                      timestamp AS 'Timestamp' FROM attendance ORDER BY id DESC""")


def load_students():
    return read_df("SELECT student_id AS 'Student ID', student_name AS 'Student Name', created_at AS 'Created At' FROM students ORDER BY student_name")


def load_faces():
    return read_df("SELECT session_time AS 'Time', faces_detected AS 'Faces Detected' FROM engagement_logs ORDER BY id DESC LIMIT 500")


def load_audio():
    return read_df("SELECT session_time AS 'Time', duration_seconds AS 'Duration (s)', rms_level AS 'RMS Level', activity_label AS 'Audio Activity' FROM audio_logs ORDER BY id DESC LIMIT 500")


def load_sessions():
    return read_df("""SELECT session_time AS 'Session Time', unique_tracks AS 'Unique Temporary Tracks',
                      confirmed_tracks AS 'Confirmed Tracks', face_samples AS 'Face Samples',
                      session_duration AS 'Duration (seconds)' FROM session_reports ORDER BY id DESC LIMIT 100""")


def load_recognition():
    return read_df("""SELECT session_time AS 'Time', student_id AS 'Student ID',
                      predicted_name AS 'Predicted Name', confidence AS 'LBPH Distance',
                      result AS 'Result' FROM recognition_logs ORDER BY id DESC LIMIT 500""")


def build_student_summary(attendance_df, total_classes):
    if attendance_df.empty:
        return pd.DataFrame()
    work = attendance_df.copy()
    work["Attendance Date"] = work["Timestamp"].str[:10]
    summary = work.groupby("Student ID", as_index=False).agg(
        **{"Student Name": ("Student Name", "last"), "Present Days": ("Attendance Date", "nunique")}
    )
    summary["Total Classes Held"] = int(total_classes)
    summary["Attendance %"] = (summary["Present Days"] / int(total_classes) * 100).clip(upper=100).round(1)
    summary["Status"] = summary["Attendance %"].apply(lambda v: "Low Attendance" if v < 75 else "OK")
    return summary


# ----------------------------- Face recognition -----------------------------
def detector():
    d = cv2.CascadeClassifier(str(CASCADE_PATH))
    if d.empty():
        raise RuntimeError(f"Could not load OpenCV Haar cascade: {CASCADE_PATH}")
    return d


def safe_student_folder(sid):
    # Avoid allowing path separators in student IDs.
    safe = "".join(ch for ch in sid if ch.isalnum() or ch in "-_")
    if not safe:
        raise ValueError("Student ID must contain letters or numbers.")
    return FACE_DIR / safe


def enroll_student(name, sid, camera_index=0, target_images=15):
    """Capture consented face samples locally. Returns count and error text."""
    if not name.strip() or not sid.strip():
        return 0, "Enter student name and ID first."
    if not LBPH_AVAILABLE:
        return 0, "LBPH recognition is unavailable. Install opencv-contrib-python in this virtual environment."
    d = detector()
    folder = safe_student_folder(sid)
    folder.mkdir(parents=True, exist_ok=True)
    cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    if not cam.isOpened():
        cam.release()
        return 0, "Could not open webcam. Close other camera apps and check Windows camera permission."
    saved = 0
    try:
        started = time.monotonic()
        while saved < target_images and time.monotonic() - started < 45:
            ok, frame = cam.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = d.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces):
                # Use largest face to reduce accidental capture of bystanders.
                x, y, w, h = max(faces, key=lambda b: int(b[2]) * int(b[3]))
                crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                out = folder / f"{int(time.time()*1000)}_{saved:03d}.png"
                if cv2.imwrite(str(out), crop):
                    saved += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 0), 2)
            cv2.putText(frame, f"Consent-based enrollment: {saved}/{target_images}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            # Keep camera capture in the UI while enrollment runs.
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=640)
            time.sleep(0.12)
    finally:
        cam.release()
    with db() as con:
        con.execute(
            "INSERT INTO students(student_id,student_name,created_at) VALUES(?,?,?) "
            "ON CONFLICT(student_id) DO UPDATE SET student_name=excluded.student_name",
            (sid.strip(), name.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
    return saved, None if saved else "No face images were captured. Improve lighting and face the camera."


def train_recognizer():
    if not LBPH_AVAILABLE:
        return None, {}, "LBPH is unavailable. Install opencv-contrib-python."
    images, labels, label_map = [], [], {}
    students = load_students()
    if students.empty:
        return None, {}, "No enrolled students yet."
    label = 0
    for _, row in students.iterrows():
        sid = str(row["Student ID"])
        folder = safe_student_folder(sid)
        files = sorted(folder.glob("*.png")) if folder.exists() else []
        valid = []
        for file in files:
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if img is not None and img.size:
                valid.append(img)
        if not valid:
            continue
        label_map[label] = (sid, str(row["Student Name"]))
        images.extend(valid)
        labels.extend([label] * len(valid))
        label += 1
    if len(images) < 2 or not label_map:
        return None, {}, "Not enough enrolled face samples. Enroll at least one student first."
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    return recognizer, label_map, None


def record_audio_activity(seconds=5, sample_rate=16000):
    if not AUDIO_AVAILABLE:
        raise RuntimeError("sounddevice is not installed.")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    if rms < 0.008:
        label = "Low sound activity"
    elif rms < 0.04:
        label = "Moderate sound activity"
    else:
        label = "High sound activity"
    return rms, label


init_db()

st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("Local prototype: manual attendance, optional enrolled-face recognition, audio-level sampling, analytics and teacher-support suggestions.")
with st.expander("Important scope & privacy notes", expanded=False):
    st.markdown(
        "- Use face enrollment only with informed consent and in accordance with your institution's rules.\n"
        "- Face samples are stored locally in `face_data/`; delete them when no longer needed.\n"
        "- Face recognition can make mistakes due to lighting, pose and camera quality. Always verify identity and attendance manually.\n"
        "- Audio feature measures sound amplitude only; it does not transcribe speech or measure attention.\n"
        "- Face presence and sound levels are not reliable measures of student engagement or learning."
    )

att = load_attendance()
today = date.today().isoformat()
today_df = att[att["Timestamp"].str.startswith(today)] if not att.empty else att
faces_df = load_faces()
audio_df = load_audio()
students_df = load_students()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Attendance records", len(att))
m2.metric("Present today", today_df["Student ID"].nunique())
m3.metric("Face samples", len(faces_df))
m4.metric("Enrolled students", len(students_df))

tabs = st.tabs(["📝 Attendance", "🧑‍🎓 Face Enrollment", "📷 Recognition & Tracking", "🎙️ Audio Activity", "📊 Analytics & Reports", "🧠 Teaching Suggestions"])

# ----------------------------- Attendance -----------------------------
with tabs[0]:
    st.subheader("Manual Attendance")
    with st.form("attendance_form", clear_on_submit=True):
        name = st.text_input("Student name")
        sid = st.text_input("Student ID")
        submit = st.form_submit_button("Save attendance")
        if submit:
            ok, msg = save_attendance(name, sid)
            (st.success if ok else st.warning)(msg)
            if ok:
                st.rerun()
    att = load_attendance()
    if not att.empty:
        st.dataframe(att, use_container_width=True, hide_index=True)
        st.download_button("Download attendance CSV", att.to_csv(index=False).encode("utf-8"), "attendance.csv", "text/csv")
    else:
        st.info("No attendance records yet.")

# ----------------------------- Enrollment -----------------------------
with tabs[1]:
    st.subheader("Consent-based face enrollment")
    if not LBPH_AVAILABLE:
        st.error("Face recognition library is not installed. Install opencv-contrib-python using the command in README, then restart the app.")
    st.warning("Only enroll a person who has agreed. Samples are saved as local grayscale face crops in the face_data folder.")
    with st.form("enroll_form"):
        en_name = st.text_input("Student name", key="en_name")
        en_sid = st.text_input("Student ID", key="en_sid")
        en_consent = st.checkbox("I have permission/consent to capture and store this student's face samples.")
        en_submit = st.form_submit_button("Capture 15 face samples")
    if en_submit:
        if not en_consent:
            st.error("Confirm consent before capturing samples.")
        else:
            with st.spinner("Capturing face samples. Keep the student centered and well-lit."):
                count, error = enroll_student(en_name, en_sid)
            if error:
                st.error(error)
            else:
                st.success(f"Captured {count} face samples for {en_name}.")
    students_df = load_students()
    if not students_df.empty:
        st.subheader("Enrolled students")
        st.dataframe(students_df, use_container_width=True, hide_index=True)
        st.caption("For deletion, stop the app and remove that student's folder from face_data, then remove the record from the students table using a database tool. Keep a backup before changing the database.")

# ----------------------------- Recognition & tracking -----------------------------
with tabs[2]:
    st.subheader("Recognize enrolled faces and track temporary presence")
    st.info("Recognition requires enrollment first. Predictions are suggestions only; confirm a person's identity before recording attendance.")
    threshold = st.slider("Recognition strictness (lower is stricter)", min_value=35, max_value=100, value=65)
    seconds = st.selectbox("Session duration", [15, 30, 60], index=1)
    if st.button("Start recognition session", type="primary"):
        recognizer, label_map, train_error = train_recognizer()
        if train_error:
            st.error(train_error)
        else:
            det = detector()
            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
            if not cam.isOpened():
                cam.release()
                st.error("Cannot open webcam. Close other camera apps and check Windows camera permissions.")
            else:
                image_area, status_area, metrics_area, progress = st.empty(), st.empty(), st.empty(), st.empty()
                start = time.monotonic()
                last_sample = start
                samples = 0
                seen_tracks = set()
                track_first_seen = {}
                track_last_seen = {}
                track_centers = {}
                next_track_id = 1
                confirmed_tracks = set()
                recognition_seen = set()
                last_recognition_time = {}
                try:
                    while time.monotonic() - start < seconds:
                        ok, frame = cam.read()
                        if not ok:
                            status_area.error("Could not read webcam frame.")
                            break
                        now = time.monotonic()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        detected = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
                        current_centers = []
                        for (x, y, w, h) in detected:
                            cx, cy = x + w // 2, y + h // 2
                            match_id, match_dist = None, 90
                            for tid, (tx, ty) in track_centers.items():
                                dist = math.hypot(cx-tx, cy-ty)
                                if dist < match_dist:
                                    match_id, match_dist = tid, dist
                            if match_id is None:
                                match_id = next_track_id
                                next_track_id += 1
                                track_first_seen[match_id] = now
                                seen_tracks.add(match_id)
                            track_centers[match_id] = (cx, cy)
                            track_last_seen[match_id] = now
                            current_centers.append(match_id)
                            duration = now - track_first_seen[match_id]
                            if duration >= 3:
                                confirmed_tracks.add(match_id)

                            face_crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                            predicted_name, predicted_sid, confidence, result = "Unknown", None, None, "Unknown"
                            try:
                                label, dist = recognizer.predict(face_crop)
                                confidence = float(dist)
                                if label in label_map and dist <= threshold:
                                    predicted_sid, predicted_name = label_map[label]
                                    result = "Candidate match"
                                else:
                                    result = "Unknown / low confidence"
                            except Exception:
                                result = "Recognition error"
                            color = (0, 200, 0) if predicted_sid else (0, 165, 255)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            caption = f"{predicted_name} | T{match_id} | {duration:.1f}s"
                            cv2.putText(frame, caption, (x, max(22, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
                            # Avoid logging every frame; one record per track/person every 5 seconds.
                            key = f"{match_id}:{predicted_sid or 'unknown'}"
                            if now - last_recognition_time.get(key, 0) >= 5:
                                log_recognition(predicted_sid, predicted_name, confidence, result)
                                last_recognition_time[key] = now
                            if predicted_sid and duration >= 3 and st.session_state.get("auto_attendance", False):
                                # Prevent duplicate same-day attendance via save_attendance.
                                ok_save, _ = save_attendance(predicted_name, predicted_sid)
                        cv2.putText(frame, f"Visible faces: {len(detected)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,0), 2)
                        image_area.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        if now - last_sample >= 2:
                            save_face_sample(len(detected))
                            samples += 1
                            last_sample = now
                        metrics_area.write(f"**Visible faces:** {len(detected)} · **Temporary tracks:** {len(seen_tracks)} · **Tracks present ≥3 seconds:** {len(confirmed_tracks)}")
                        progress.progress(min(int((now-start)/seconds*100), 100), text=f"Elapsed {int(now-start)} / {seconds} seconds")
                        time.sleep(0.03)
                    duration = time.monotonic() - start
                    save_session_report(len(seen_tracks), len(confirmed_tracks), samples, duration)
                    status_area.success("Session finished. Review recognition results and verify attendance manually.")
                finally:
                    cam.release()
    auto_attendance = st.checkbox("Automatically record attendance for recognized candidates after 3 seconds (experimental; verify results)", value=False, key="auto_attendance")
    rec = load_recognition()
    if not rec.empty:
        st.subheader("Recent recognition events")
        st.dataframe(rec, use_container_width=True, hide_index=True)
        st.download_button("Download recognition log CSV", rec.to_csv(index=False).encode("utf-8"), "recognition_log.csv", "text/csv")

# ----------------------------- Audio -----------------------------
with tabs[3]:
    st.subheader("Classroom sound-level sampling")
    st.write("Records one short sample and stores only RMS amplitude, duration and a rough sound-level label. No audio recording is saved.")
    if not AUDIO_AVAILABLE:
        st.warning("sounddevice is not installed in this environment. Install dependencies from requirements.txt.")
    else:
        if st.button("Record 5-second audio sample"):
            try:
                with st.spinner("Recording; check microphone permissions..."):
                    rms, label = record_audio_activity()
                save_audio_sample(5, rms, label)
                st.success(f"Activity label: {label}")
                st.metric("RMS sound level", f"{rms:.5f}")
            except Exception as exc:
                st.error(f"Audio capture failed: {exc}")
    audio_df = load_audio()
    if not audio_df.empty:
        st.dataframe(audio_df, use_container_width=True, hide_index=True)
        st.download_button("Download audio CSV", audio_df.to_csv(index=False).encode("utf-8"), "audio_activity.csv", "text/csv")

# ----------------------------- Analytics -----------------------------
with tabs[4]:
    st.subheader("Attendance analytics and reports")
    att = load_attendance()
    if not att.empty:
        st.bar_chart(att["Student Name"].value_counts())
        total_classes = st.number_input("Total classes held in reporting period", min_value=1, value=20, step=1)
        summary = build_student_summary(att, total_classes)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        low = summary[summary["Attendance %"] < 75]
        if not low.empty:
            st.warning(f"{len(low)} student(s) are below 75% attendance.")
        st.download_button("Download attendance summary CSV", summary.to_csv(index=False).encode("utf-8"), "attendance_summary.csv", "text/csv")
    else:
        st.info("Add attendance records to view attendance analytics.")
    st.divider()
    faces_df = load_faces()
    st.subheader("Face count trend")
    if not faces_df.empty:
        st.line_chart(faces_df.sort_values("Time").set_index("Time")["Faces Detected"])
    else:
        st.info("No face tracking samples recorded.")
    audio_df = load_audio()
    st.subheader("Audio RMS trend")
    if not audio_df.empty:
        st.line_chart(audio_df.sort_values("Time").set_index("Time")["RMS Level"])
    else:
        st.info("No audio samples recorded.")
    sessions = load_sessions()
    st.subheader("Saved session reports")
    if not sessions.empty:
        st.dataframe(sessions, use_container_width=True, hide_index=True)
        st.download_button("Download session reports CSV", sessions.to_csv(index=False).encode("utf-8"), "session_reports.csv", "text/csv")
    recognition = load_recognition()
    st.subheader("Recognition log")
    if not recognition.empty:
        st.dataframe(recognition, use_container_width=True, hide_index=True)

# ----------------------------- Teaching suggestions -----------------------------
with tabs[5]:
    st.subheader("Teacher-support suggestions (heuristics)")
    faces_df = load_faces()
    audio_df = load_audio()
    st.warning("These rules describe sensor signals only. They do not infer attention, comprehension, emotions or learning.")
    if not faces_df.empty:
        recent = faces_df.head(10)["Faces Detected"]
        st.metric("Average recent visible face count", f"{recent.mean():.1f}")
        if (recent == 0).any():
            st.write("📷 Camera check: some samples had zero detections. Check camera angle, lighting and classroom coverage.")
        else:
            st.write("📷 Vision check: faces were detected. Compare the counts with the actual classroom.")
    if not audio_df.empty:
        rms = float(audio_df.head(10)["RMS Level"].mean())
        st.metric("Average recent RMS", f"{rms:.5f}")
        if rms < 0.008:
            st.write("🎙️ Audio check: sound level was low. Check microphone placement and whether the room was quiet.")
        else:
            st.write("🎙️ Audio check: sound activity was present. This does not prove useful discussion or engagement.")
    if faces_df.empty and audio_df.empty:
        st.info("Collect some vision or audio samples to see basic suggestions.")
    st.write("👩‍🏫 Teaching practice: use questions, student feedback, participation and learning outcomes to assess teaching effectiveness.")

st.divider()
st.caption(f"Database: {DB_PATH.name} · Video frames are processed locally and not saved. Face enrollment images are stored locally in face_data/. Audio recordings are not saved. Recognition is probabilistic and requires human verification.")
