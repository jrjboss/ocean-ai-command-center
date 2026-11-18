import os
import sys
import json
import time
import fcntl
import subprocess
from io import BytesIO
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image
from roboflow_client import run_sam3

BASE = Path.home() / "ocean-ai-roboflow-test"
IMPORTS = BASE / "company_inspections/imports"
RESULTS = BASE / "company_inspections/batch_results"
JOBS = BASE / "company_inspections/batch_jobs"
KEY_FILE = Path.home() / ".config/ocean-ai/roboflow_api_key"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def saved_key():
    if KEY_FILE.is_file():
        return KEY_FILE.read_text().strip()
    return ""


def status_file(batch):
    return JOBS / (batch + ".status.json")


def read_status(batch):
    try:
        return json.loads(status_file(batch).read_text())
    except (OSError, ValueError):
        return {}


def write_status(batch, data):
    JOBS.mkdir(parents=True, exist_ok=True)
    target = status_file(batch)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(target)


def running(status):
    pid = status.get("pid")
    if not pid or status.get("state") != "running":
        return False
    try:
        command = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        return b"batch_ai.py" in command
    except (OSError, ValueError):
        return False


def images_in(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def run_worker(batch, limit):
    JOBS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    folder = IMPORTS / batch
    if not folder.is_dir() or folder.parent != IMPORTS:
        return

    destination = RESULTS / batch
    destination.mkdir(parents=True, exist_ok=True)

    with (JOBS / (batch + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return

        images = images_in(folder)
        pending = [
            p for p in images
            if not (destination / (p.name + ".json")).exists()
        ][:limit]

        status = {
            "state": "running",
            "pid": os.getpid(),
            "total": len(pending),
            "finished": 0,
            "success": 0,
            "failed": 0,
            "current": "",
            "last_error": ""
        }
        write_status(batch, status)

        key = saved_key()
        if not key:
            status["state"] = "failed"
            status["last_error"] = "Saved API key is missing"
            write_status(batch, status)
            return

        stop_file = JOBS / (batch + ".stop")

        for image in pending:
            if stop_file.exists():
                status["state"] = "stopped"
                break

            status["current"] = image.name
            write_status(batch, status)

            success = False

            for attempt in range(2):
                try:
                    data = image.read_bytes()

                    if len(data) > 30 * 1024 * 1024:
                        raise ValueError("Image exceeds 30 MB")

                    with Image.open(BytesIO(data)) as im:
                        im.verify()

                    annotated, predictions = run_sam3(data, key)

                    with Image.open(BytesIO(annotated)) as im:
                        im.verify()
                        suffix = (
                            ".png" if im.format == "PNG" else ".jpg"
                        )

                    annotation_path = (
                        destination / (image.name + suffix)
                    )
                    result_path = (
                        destination / (image.name + ".json")
                    )

                    annotation_path.write_bytes(annotated)

                    record = {
                        "source_image": str(image),
                        "annotated_image": str(annotation_path),
                        "predictions": predictions,
                        "processed_at": datetime.now().isoformat(),
                        "engineer_review": "pending"
                    }

                    result_path.write_text(
                        json.dumps(record, indent=2)
                    )

                    success = True
                    break

                except Exception as error:
                    status["last_error"] = (
                        f"{image.name}: {type(error).__name__}"
                    )
                    if attempt == 0:
                        time.sleep(3)

            status["finished"] += 1

            if success:
                status["success"] += 1
            else:
                status["failed"] += 1

            write_status(batch, status)

        if status["state"] == "running":
            status["state"] = "completed"

        status["current"] = ""
        write_status(batch, status)


def show_batch_ai():
    st.divider()
    st.header("🤖 SAM 3 Batch AI Processing")
    st.write(
        "Analyze imported inspection images directly "
        "from the OCEAN AI dashboard."
    )

    if not saved_key():
        st.error("Saved Roboflow API key not found.")
        return

    if not IMPORTS.exists():
        st.info("Import an inspection ZIP first.")
        return

    folders = sorted(
        [p for p in IMPORTS.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not folders:
        st.info("Import an inspection ZIP first.")
        return

    folder = st.selectbox(
        "Choose an imported inspection batch",
        folders,
        format_func=lambda p: (
            f"{p.name[:12]} — {len(images_in(p))} images"
        )
    )

    batch = folder.name
    images = images_in(folder)
    destination = RESULTS / batch

    def completed_images():
        return [
            p for p in images
            if (destination / (p.name + ".json")).is_file()
        ]

    completed = completed_images()
    remaining = len(images) - len(completed)

    a, b, c = st.columns(3)
    a.metric("Imported Images", len(images))
    b.metric("AI Completed", len(completed))
    c.metric("Remaining", remaining)

    amount = st.selectbox(
        "How many images should SAM 3 analyze?",
        [5, 25, 100, 500, "All remaining"]
    )

    consent = st.checkbox(
        "I have permission to send these images to "
        "Roboflow Cloud and understand that API "
        "requests may incur charges."
    )

    current = read_status(batch)
    active = running(current)

    if st.button(
        "▶ Start AI Batch Analysis",
        type="primary",
        disabled=not consent or active or remaining == 0
    ):
        JOBS.mkdir(parents=True, exist_ok=True)
        (JOBS / (batch + ".stop")).unlink(missing_ok=True)

        limit = remaining if amount == "All remaining" else amount
        log_path = JOBS / (batch + ".log")

        with log_path.open("ab") as log:
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    batch,
                    str(limit)
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(BASE)
            )

        st.success("AI batch started. Results save automatically.")

    @st.fragment(run_every="3s")
    def live_results():
        status = read_status(batch)
        is_active = running(status)

        if is_active:
            finished = status.get("finished", 0)
            total = max(status.get("total", 1), 1)

            st.info(
                f"Processing: {status.get('current', '')} "
                f"— {finished}/{total} attempted"
            )

            st.progress(min(finished / total, 1.0))

            if st.button(
                "Stop after current image",
                key="stop_" + batch
            ):
                JOBS.mkdir(parents=True, exist_ok=True)
                (JOBS / (batch + ".stop")).touch()
                st.warning("Stop requested.")

        elif status:
            st.caption(
                f"Last batch: {status.get('state', 'unknown')} | "
                f"Successful: {status.get('success', 0)} | "
                f"Failed: {status.get('failed', 0)}"
            )

        done = completed_images()

        st.subheader(f"Saved AI Results ({len(done)})")

        if not done:
            st.info("Results will appear here after processing.")
            return

        selected = st.selectbox(
            "Choose an image to review",
            done,
            format_func=lambda p: p.name,
            key="result_" + batch
        )

        try:
            record = json.loads(
                (destination / (selected.name + ".json")).read_text()
            )

            annotated = Path(record["annotated_image"])
            left, right = st.columns(2)

            with left:
                st.image(
                    selected,
                    caption="Original inspection image",
                    use_container_width=True
                )

            with right:
                if annotated.is_file():
                    st.image(
                        annotated,
                        caption="SAM 3 detection overlay",
                        use_container_width=True
                    )
                else:
                    st.warning("Saved overlay not found.")

            st.write(
                "Predicted regions:",
                len(record.get("predictions", []))
            )
            st.caption(
                "Engineer review: pending. "
                "AI regions are not confirmed corrosion defects."
            )

        except (OSError, ValueError, KeyError) as error:
            st.error(f"Could not display this result: {error}")

    live_results()


if __name__ == "__main__" and "--worker" in sys.argv:
    run_worker(sys.argv[2], int(sys.argv[3]))
