"""OCEAN AI three-view inspection controls, live inference, and saved reviews."""
import hashlib
import json
import uuid
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from roboflow_client import run_sam3_full


def _image_ext(image_bytes):
    with Image.open(BytesIO(image_bytes)) as image:
        image.verify()
        name = image.format
    if name == "PNG":
        return ".png"
    if name == "JPEG":
        return ".jpg"
    raise ValueError("Only PNG/JPEG Roboflow output images are supported")


def _show_view(name, source, clean=None, labeled=None, key="view"):
    options = ["Original"]
    if clean is not None:
        options.append("Clean Masks")
    if labeled is not None:
        options.append("Labeled Detection")
    if len(options) == 1:
        st.warning("Only the original image is available for this record")
    else:
        st.caption("Switch image views without rerunning SAM 3 or using API credits.")
    choice = st.segmented_control(
        "Display mode", options,
        default="Clean Masks" if "Clean Masks" in options else options[-1],
        key=key,
    )
    if choice == "Clean Masks":
        display = clean
    elif choice == "Labeled Detection":
        display = labeled
    else:
        display = source
    if display is not None:
        st.image(display, caption=f"{name} — {choice}", use_container_width=True)


def inspection_studio_ui(manual_folder, key_loader, snapshot):
    st.header("Inspection Studio")
    st.caption("Analyze one image, save all Roboflow outputs, and switch visualizations instantly.")
    key = key_loader()
    if not key:
        st.error("Roboflow API key not configured in this environment")
        return

    uploads = st.file_uploader(
        "Upload inspection images",
        type=["jpg", "jpeg", "png"], accept_multiple_files=True,
        key="studio_three_uploads"
    )
    if not uploads:
        st.info("Upload a public test photo. Use Batch Center for many images.")
        return

    chosen = st.selectbox("Select an image", uploads,
                          format_func=lambda f: f.name, key="studio_three_selected")
    raw = chosen.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    consent = st.checkbox(
        "I have permission to send this image to Roboflow Cloud; API usage may be charged.",
        key="studio_three_permission"
    )

    if st.button("Run SAM 3 Analysis", type="primary", disabled=not consent,
                 key="studio_three_run"):
        try:
            if len(raw) > 20 * 1024 * 1024:
                raise ValueError("Maximum one-image size in Inspection Studio is 20 MB")
            with Image.open(BytesIO(raw)) as im:
                im.verify()

            with st.spinner("Running SAM 3 — saving three outputs..."):
                result = run_sam3_full(raw, key)

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            output = manual_folder / run_id
            output.mkdir(parents=True, exist_ok=False)
            original_ext = Path(chosen.name).suffix.lower()
            if original_ext not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Unsupported image format")
            original_path = output / ("original" + original_ext)
            original_path.write_bytes(raw)

            clean_path = None
            labeled_path = None
            if result["clean"] is not None:
                clean_path = output / ("clean_preview" + _image_ext(result["clean"]))
                clean_path.write_bytes(result["clean"])
            if result["labeled"] is not None:
                labeled_path = output / ("labeled_detection" + _image_ext(result["labeled"]))
                labeled_path.write_bytes(result["labeled"])

            # annotated_image remains a valid path for older display readers.
            primary = clean_path or labeled_path
            record = {
                "original_filename": chosen.name,
                "source_image": str(original_path),
                "clean_preview": str(clean_path) if clean_path else "",
                "labeled_image": str(labeled_path) if labeled_path else "",
                "annotated_image": str(primary),
                "predictions": result["predictions"],
                "processed_at": datetime.now().isoformat(),
                "engineer_review": "pending",
                "review_notes": "",
                "model": "SAM 3 - Roboflow Cloud",
            }
            (output / "result.json").write_text(json.dumps(record, indent=2))
            st.session_state["studio_three_latest"] = {
                "sha": digest, "name": chosen.name, "record": record,
            }
            snapshot.clear()
            st.success("SAM 3 analysis saved. Switch between the available views below.")
            if not clean_path:
                st.warning("Roboflow did not return clean_preview. Publish the 3-output workflow in Roboflow.")
        except Exception as error:
            st.error(f"Analysis failed: {error}")

    latest = st.session_state.get("studio_three_latest")
    if latest and latest.get("sha") == digest:
        data = latest["record"]
        clean = Path(data["clean_preview"]) if data.get("clean_preview") else None
        labeled = Path(data["labeled_image"]) if data.get("labeled_image") else None
        st.metric("Candidate regions", len(data["predictions"]))
        _show_view(
            chosen.name, raw,
            clean if clean and clean.exists() else None,
            labeled if labeled and labeled.exists() else None,
            key="studio_three_view"
        )
        counts = Counter(
            str(pred.get("class", pred.get("class_name", "Other")))
            for pred in data["predictions"] if isinstance(pred, dict)
        )
        if counts:
            st.dataframe(pd.DataFrame([
                {"Class": label, "Predicted regions": count}
                for label, count in counts.most_common()
            ]), hide_index=True, use_container_width=True)
        st.caption("Predicted regions require engineer review; counts do not measure accuracy.")
    else:
        st.image(raw, caption=f"Original: {chosen.name}", use_container_width=True)


def results_library_ui(snapshot, historical_folder):
    st.header("Results Library")
    st.caption("Original, clean masks and labeled output are separate views; no repeat Roboflow request needed.")
    imported, records = snapshot()
    if not records:
        st.info("No saved inspections found in this installation.")
        return

    st.caption(f"{len(records):,} completed inspections in this current installation")
    st.dataframe(pd.DataFrame([
        {
            "Image": r["name"],
            "Regions": r["regions"],
            "Review": r["review"],
            "Type": "Batch" if r["batch"] else "Manual",
        }
        for r in records
    ]), hide_index=True, use_container_width=True)

    selected_path = st.selectbox(
        "Select inspection", [r["path"] for r in records],
        format_func=lambda p: next(
            (f'{r["name"]} · {r["regions"]} regions' for r in records if r["path"] == p),
            Path(p).stem,
        ), key="three_library_selected"
    )
    chosen = next(r for r in records if r["path"] == selected_path)
    try:
        data = json.loads(Path(selected_path).read_text())
    except (OSError, ValueError) as error:
        st.error(f"Cannot read inspection JSON: {error}")
        return

    def path_if_valid(key):
        v = data.get(key)
        if not isinstance(v, str) or not v:
            return None
        path = Path(v)
        return path if path.is_file() else None

    original = path_if_valid("source_image")
    clean = path_if_valid("clean_preview")
    labeled = path_if_valid("labeled_image")
    old_annotation = path_if_valid("annotated_image")

    # Old records contain only annotated_image. Don't misrepresent it as clean.
    if clean is None and labeled is None:
        labeled = old_annotation

    if original is None:
        st.warning("Original image was not found in this installation")
    else:
        _show_view(chosen["name"], original, clean, labeled,
                   key="three_library_view_" + Path(selected_path).parent.name)

    a, b = st.columns(2)
    a.metric("Candidate regions", chosen["regions"])
    b.metric("Prediction classes", len(chosen["classes"]))
    if chosen["classes"]:
        st.dataframe(pd.DataFrame([
            {"Class": label, "Regions": count}
            for label, count in chosen["classes"].items()
        ]), hide_index=True, use_container_width=True)

    st.subheader("Human Inspection Review")
    st.caption("Review records are not authenticated engineering sign-offs.")
    decisions = ["pending", "possible_corrosion", "false_detection", "requires_followup"]
    old_decision = chosen["review"]
    choice = st.selectbox(
        "Review decision", decisions,
        index=decisions.index(old_decision) if old_decision in decisions else 0,
        key="three_review_decision_" + selected_path,
    )
    notes = st.text_area(
        "Review notes", value=chosen.get("review_notes", ""),
        placeholder="Describe the observation or required follow-up.",
        key="three_review_notes_" + selected_path,
    )
    if st.button("Save Review", type="primary", key="three_review_save_" + selected_path):
        try:
            data["engineer_review"] = choice
            data["review_notes"] = notes
            data["reviewed_at"] = datetime.now().isoformat()
            path = Path(selected_path)
            tmp = path.with_suffix(".review.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(path)
            snapshot.clear()
            st.success("Review saved locally in this installation")
            st.rerun()
        except OSError as error:
            st.error(f"Could not save review: {error}")

    with st.expander("View original 30-image SAM 3 experiment"):
        originals = historical_folder / "original"
        annotations = historical_folder / "annotated"
        if not originals.is_dir():
            st.caption("Historical dataset is only on your Ubuntu workstation.")
        else:
            photos = sorted(p for p in originals.iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if photos:
                selected = st.selectbox("Historical image", photos,
                                        format_func=lambda p: p.name)
                x, y = st.columns(2)
                with x:
                    st.image(selected, caption="Original", use_container_width=True)
                with y:
                    old = annotations / selected.name
                    if old.is_file():
                        st.image(old, caption="Saved historical overlay",
                                 use_container_width=True)
                    else:
                        st.info("No old annotated image saved")
