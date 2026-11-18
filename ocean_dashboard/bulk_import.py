import streamlit as st
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from io import BytesIO
from uuid import uuid4
import shutil

BASE = Path.home() / "ocean-ai-roboflow-test"
IMPORTS = BASE / "company_inspections" / "imports"
EXTENSIONS = {".jpg", ".jpeg", ".png"}

def show_bulk_import():
    st.header("Bulk Image Import")
    st.write(
        "Import hundreds or thousands of inspection "
        "images from a ZIP file."
    )

    uploaded = st.file_uploader(
        "Upload an inspection image ZIP",
        type=["zip"],
        key="ocean_bulk_zip"
    )

    if uploaded is None:
        return

    try:
        with ZipFile(BytesIO(uploaded.getvalue())) as z:
            files = [
                info for info in z.infolist()
                if not info.is_dir()
                and Path(info.filename).suffix.lower()
                in EXTENSIONS
            ]

            st.metric("Images found", len(files))

            if not files:
                st.warning("No JPG or PNG images found.")
                return

            if len(files) > 10000:
                st.error("Maximum 10,000 images per import.")
                return

            if any(
                info.file_size > 30 * 1024 * 1024
                for info in files
            ):
                st.error("An image exceeds the 30 MB limit.")
                return

            if sum(info.file_size for info in files) > 4 * 1024**3:
                st.error("Uncompressed images exceed 4 GB.")
                return

            if st.button("Import Images", type="primary"):
                folder = IMPORTS / uuid4().hex
                folder.mkdir(parents=True, exist_ok=False)

                progress = st.progress(0)
                saved = 0

                try:
                    for number, info in enumerate(files, 1):
                        suffix = Path(info.filename).suffix.lower()
                        destination = folder / f"{number:06d}{suffix}"

                        with z.open(info) as source:
                            with destination.open("wb") as target:
                                shutil.copyfileobj(
                                    source, target,
                                    length=1024 * 1024
                                )

                        saved += 1
                        progress.progress(number / len(files))

                except Exception:
                    shutil.rmtree(folder, ignore_errors=True)
                    raise

                st.success(f"{saved} images imported successfully!")
                st.write("Saved locally:", str(folder))
                st.info(
                    "Images are stored for inspection. "
                    "Automatic SAM 3 batch processing "
                    "will be added separately."
                )

    except (BadZipFile, OSError, ValueError) as e:
        st.error(f"Import failed: {e}")
