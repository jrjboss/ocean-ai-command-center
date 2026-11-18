import streamlit as st
import pandas as pd
import json
import html
import uuid
from io import BytesIO
from pathlib import Path
from datetime import datetime
from collections import Counter
from PIL import Image

from roboflow_client import run_sam3
from bulk_import import show_bulk_import

BASE = Path.home() / "ocean-ai-roboflow-test"
COMPANY = BASE / "company_inspections"
IMPORTS = COMPANY / "imports"
BATCH_RESULTS = COMPANY / "batch_results"
MANUAL = COMPANY / "manual_inspections"
HISTORICAL = BASE / "batch30_20261118_012012"
KEY_FILE = Path.home() / ".config/ocean-ai/roboflow_api_key"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

st.set_page_config(
    page_title="OCEAN AI | Inspection Command Center",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    --ocean-bg: #071522;
    --ocean-panel: #102438;
    --ocean-line: #284458;
    --ocean-cyan: #39d6db;
    --ocean-gold: #d8b777;
    --ocean-text: #eff7fa;
    --ocean-muted: #9bb2c3;
}

html, body, [data-testid="stAppViewContainer"] {
    background: #071522;
    color: var(--ocean-text);
}

[data-testid="stAppViewContainer"] {
    background-image:
      radial-gradient(ellipse at 94% 1%,
      rgba(20,112,151,.20),transparent 43%),
      linear-gradient(160deg,#081c2c,#071522 58%);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #091b2b;
    border-right: 1px solid #254052;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #c1d3df;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 11px;
    padding: 6px 10px;
    margin-bottom: 5px;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #143148;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f3f8fb !important;
    letter-spacing: -.025em;
}

h1 {font-weight: 750 !important;}
h2 {font-weight: 680 !important;}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #284458;
}

div[data-testid="stMetric"] {
    background: #102438;
    border: 1px solid #29495c;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricValue"] {
    color: #f3fbfd;
}

div[data-testid="stMetricLabel"] {
    color: #aac0d0;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(100deg,#087fa7,#05b2bb);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
}

.stButton > button {
    border-radius: 10px;
    min-height: 43px;
    font-weight: 600;
}

div[data-testid="stFileUploader"],
div[data-testid="stDataFrame"],
div[data-testid="stSelectbox"] {
    border-radius: 12px;
}

.ocean-brand {
    padding: 8px 0 23px;
    border-bottom: 1px solid #29495c;
    margin-bottom: 20px;
}

.ocean-wordmark {
    font-size: 26px;
    font-weight: 850;
    color: #f2f9fc;
    letter-spacing: -1px;
}

.ocean-wordmark span {
    color: #3bd1df;
}

.ocean-small {
    color: #91aec0;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 4px;
}

.ocean-hero {
    border: 1px solid #30546b;
    border-radius: 22px;
    padding: 35px 38px;
    margin-bottom: 26px;
    background:
      radial-gradient(circle at 87% 30%,
        rgba(26,194,206,.20),transparent 29%),
      linear-gradient(112deg,#11374d,#0a2236 76%);
    position: relative;
    overflow: hidden;
}

.ocean-eyebrow {
    color: #d8b777;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.1px;
    text-transform: uppercase;
}

.ocean-hero-title {
    color: #f4fafc;
    font-size: clamp(30px,3.3vw,49px);
    font-weight: 820;
    line-height: 1.1;
    letter-spacing: -1.4px;
    margin: 14px 0;
}

.ocean-hero-subtitle {
    color: #b5d0df;
    max-width: 700px;
    line-height: 1.65;
    font-size: 15px;
}

.ocean-metric {
    background: linear-gradient(160deg,#122b40,#0c2031);
    border: 1px solid #2b4a5d;
    border-radius: 17px;
    padding: 23px 21px;
    min-height: 145px;
    margin-bottom: 12px;
}

.ocean-metric-label {
    color: #a9c0d0;
    font-size: 12px;
    font-weight: 650;
}

.ocean-metric-value {
    color: #f4fafb;
    font-size: clamp(27px,2.5vw,41px);
    font-weight: 790;
    letter-spacing: -1px;
    margin-top: 12px;
}

.ocean-metric-foot {
    color: #5cd4d4;
    font-size: 11px;
    margin-top: 5px;
}

.ocean-section {
    color: #f0f8fc;
    font-size: 20px;
    font-weight: 750;
    margin: 23px 0 12px;
}

.ocean-panel {
    background: #102438;
    border: 1px solid #2a485c;
    border-radius: 15px;
    padding: 20px;
    color: #c6d9e3;
    line-height: 1.8;
}

.ocean-tag {
    display: inline-block;
    color: #69dfde;
    background: #113e49;
    border: 1px solid #276e75;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 750;
    padding: 5px 12px;
    margin-top: 14px;
}

.ocean-note {
    color: #9db5c6;
    font-size: 12px;
    margin: 12px 0 18px;
}

hr {border-color: #29485c !important;}

@media (max-width: 700px) {
    .ocean-hero {padding: 24px;}
    .ocean-metric {min-height: 125px;}
}
</style>
""", unsafe_allow_html=True)


def escaped(value):
    return html.escape(str(value))


def secret_key():
    try:
        return KEY_FILE.read_text().strip()
    except OSError:
        return ""


@st.cache_data(ttl=5, show_spinner=False)
def snapshot():
    imported = 0
    records = []

    if IMPORTS.is_dir():
        for folder in IMPORTS.iterdir():
            if folder.is_dir():
                imported += sum(
                    1 for item in folder.iterdir()
                    if item.is_file()
                    and item.suffix.lower() in IMAGE_EXTENSIONS
                )

    sources = [
        (BATCH_RESULTS, True),
        (MANUAL, False)
    ]

    for root, is_batch in sources:
        if not root.is_dir():
            continue

        for path in root.glob("*/*.json"):
            try:
                data = json.loads(path.read_text())

                predictions = data.get("predictions")
                if not isinstance(predictions, list):
                    continue

                classes = Counter(
                    str(p.get("class", p.get("class_name", "Other")))
                    for p in predictions
                    if isinstance(p, dict)
                )

                records.append({
                    "path": str(path),
                    "source": data.get("source_image", ""),
                    "overlay": data.get("annotated_image", ""),
                    "name": Path(
                        data.get("source_image", path.stem)
                    ).name,
                    "regions": len(predictions),
                    "classes": dict(classes),
                    "review": data.get(
                        "engineer_review", "pending"
                    ),
                    "review_notes": data.get(
                        "review_notes", ""
                    ),
                    "date": data.get(
                        "processed_at", ""
                    ),
                    "batch": is_batch,
                })

            except (OSError, ValueError, TypeError):
                continue

    records.sort(
        key=lambda r: Path(r["path"]).stat().st_mtime
        if Path(r["path"]).exists() else 0,
        reverse=True
    )

    return imported, records


def section_heading(title, subtitle):
    st.markdown(
        f"<h2>{escaped(title)}</h2>"
        f"<div class='ocean-note'>{escaped(subtitle)}</div>",
        unsafe_allow_html=True
    )


def metric_card(label, value, foot):
    st.markdown(
        "<div class='ocean-metric'>"
        f"<div class='ocean-metric-label'>{escaped(label)}</div>"
        f"<div class='ocean-metric-value'>{escaped(value)}</div>"
        f"<div class='ocean-metric-foot'>{escaped(foot)}</div>"
        "</div>",
        unsafe_allow_html=True
    )


def overview():
    st.markdown("""
    <div class="ocean-hero">
      <div class="ocean-eyebrow">
        Maritime Intelligence / Inspection Operations
      </div>
      <div class="ocean-hero-title">
        See beneath the surface.
      </div>
      <div class="ocean-hero-subtitle">
        One command center for underwater inspection images,
        SAM 3 corrosion candidates, batch processing,
        and human review.
      </div>
      <div class="ocean-tag">● LOCAL INSPECTION PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    @st.fragment(run_every="6s")
    def live_overview():
        imported, records = snapshot()

        batch_records = [
            r for r in records if r["batch"]
        ]
        total_regions = sum(
            r["regions"] for r in records
        )
        reviewed = sum(
            r["review"] != "pending" for r in records
        )
        completed = len(batch_records)
        remaining = max(0, imported - completed)

        cols = st.columns(4)

        with cols[0]:
            metric_card(
                "IMPORTED IMAGES",
                f"{imported:,}",
                "Local inspection library"
            )
        with cols[1]:
            metric_card(
                "BATCH AI ANALYZED",
                f"{completed:,}",
                "Saved SAM 3 results"
            )
        with cols[2]:
            metric_card(
                "AI REGIONS FLAGGED",
                f"{total_regions:,}",
                "Predicted regions, not verified defects"
            )
        with cols[3]:
            metric_card(
                "REVIEWS RECORDED",
                f"{reviewed:,}",
                "Human review decisions"
            )

        left, right = st.columns([1.3, 1])

        with left:
            st.markdown(
                "<div class='ocean-section'>"
                "Inspection Processing</div>",
                unsafe_allow_html=True
            )

            ratio = min(
                completed / imported, 1
            ) if imported else 0

            st.progress(ratio)

            st.caption(
                f"{completed:,} of {imported:,} imported "
                f"images analyzed — {remaining:,} remaining"
            )

            st.markdown(
                "<div class='ocean-section'>"
                "Recent AI Results</div>",
                unsafe_allow_html=True
            )

            if records:
                table = pd.DataFrame([
                    {
                        "Image": r["name"],
                        "Regions": r["regions"],
                        "Review": r["review"],
                        "Source": (
                            "Batch" if r["batch"] else "Manual"
                        )
                    }
                    for r in records[:12]
                ])
                st.dataframe(
                    table,
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info(
                    "AI results will appear after your "
                    "first completed inspection."
                )

        with right:
            st.markdown(
                "<div class='ocean-section'>"
                "System Readiness</div>",
                unsafe_allow_html=True
            )

            key_status = (
                "Stored securely on local machine"
                if secret_key()
                else "Not configured"
            )

            st.markdown(
                "<div class='ocean-panel'>"
                "🟢 <b>SAM 3 Cloud Connector</b><br>"
                "Working prototype<br><br>"
                "🔑 <b>Roboflow Credentials</b><br>"
                f"{escaped(key_status)}<br><br>"
                "🟡 <b>ROV Hardware</b><br>"
                "Integration pending<br><br>"
                "🟡 <b>Company Server</b><br>"
                "Not deployed"
                "</div>",
                unsafe_allow_html=True
            )

    live_overview()

    st.info(
        "AI predictions require engineer review. "
        "Processing completion is not model accuracy."
    )


def inspection_studio():
    from ocean_three_views import inspection_studio_ui
    inspection_studio_ui(MANUAL, secret_key, snapshot)


def batch_center():
    section_heading(
        "Batch Processing Center",
        "Import large image sets, start AI jobs, "
        "and review progress in one place."
    )

    st.markdown(
        "<div class='ocean-panel'>"
        "<b>Bulk inspection workflow</b><br>"
        "Import a ZIP → Select a batch → "
        "Run SAM 3 → Review saved results.<br>"
        "The system skips previously completed "
        "images when processing a batch."
        "</div>",
        unsafe_allow_html=True
    )

    st.write("")
    show_bulk_import()

    try:
        from batch_ai import show_batch_ai
        show_batch_ai()
    except ImportError:
        st.warning(
            "The batch-processing module is not installed. "
            "Existing imported images remain safe."
        )


def results_library():
    from ocean_three_views import results_library_ui
    results_library_ui(snapshot, HISTORICAL)


def ai_performance():
    section_heading(
        "AI Performance & Quality",
        "Live operational statistics and measured "
        "model accuracy are shown separately."
    )

    imported, records = snapshot()

    batch_count = sum(
        r["batch"] for r in records
    )
    total_regions = sum(
        r["regions"] for r in records
    )
    classes = Counter()

    for record in records:
        classes.update(record["classes"])

    cols = st.columns(3)

    with cols[0]:
        metric_card(
            "COMPLETED BATCH IMAGES",
            f"{batch_count:,}",
            "Saved prediction records"
        )

    with cols[1]:
        metric_card(
            "REGIONS PREDICTED",
            f"{total_regions:,}",
            "Unverified segmentation candidates"
        )

    with cols[2]:
        average = (
            total_regions / len(records)
            if records else 0
        )
        metric_card(
            "AVERAGE REGIONS / IMAGE",
            f"{average:.1f}",
            "Not an accuracy measure"
        )

    st.markdown(
        "<div class='ocean-section'>"
        "Detection Classes</div>",
        unsafe_allow_html=True
    )

    if classes:
        chart = pd.DataFrame([
            {"Class": name, "Regions": count}
            for name, count in classes.most_common()
        ])
        st.bar_chart(
            chart.set_index("Class"),
            y="Regions",
            use_container_width=True
        )
    else:
        st.info("No predictions recorded yet.")

    st.divider()

    st.subheader("Measured SAM 3 Validation")

    st.caption(
        "Historical evaluation on 14 annotated "
        "pipeline images containing reference corrosion."
    )

    a, b, c, d = st.columns(4)
    a.metric("Pixel IoU", "17.76%")
    b.metric("Precision", "17.90%")
    c.metric("Recall", "96.04%")
    d.metric("F1 Score", "30.17%")

    st.warning(
        "SAM 3 currently over-segments many pipe "
        "surfaces. A large predicted area is not "
        "proof of corrosion. The 110 imported images "
        "do not yet have independently established "
        "accuracy metrics."
    )

    st.caption(
        "The earlier 30-image cloud test had 29 "
        "successful API requests. API success rate "
        "must not be confused with detection accuracy."
    )


def rov_operations():
    section_heading(
        "ROV Operations",
        "Hardware integration workspace — "
        "planned for the next development phase."
    )

    a, b, c = st.columns(3)
    a.metric("ROV Connection", "Offline")
    b.metric("Camera Stream", "Not connected")
    c.metric("Sensor Telemetry", "Not connected")

    st.markdown(
        "<div class='ocean-panel'>"
        "<b>Planned inspection architecture</b><br><br>"
        "ROV camera and sensors → Raspberry Pi 4 → "
        "Surface station → OCEAN AI → "
        "Engineer inspection dashboard.<br><br>"
        "Heavy SAM 3 inference runs on the surface "
        "computer or approved server, rather than "
        "assuming full SAM 3 runs on the Raspberry Pi."
        "</div>",
        unsafe_allow_html=True
    )

    st.info(
        "No real camera, sonar, depth sensor, "
        "or propulsion connection has been "
        "implemented yet."
    )


st.sidebar.markdown("""
<div class="ocean-brand">
  <div class="ocean-wordmark">🌊 OCEAN <span>AI</span></div>
  <div class="ocean-small">INSPECTION COMMAND CENTER</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "Inspection Studio",
        "Batch Center",
        "Results Library",
        "AI Performance",
        "ROV Operations"
    ]
)

st.sidebar.divider()
st.sidebar.caption("OCEAN AI · DEVELOPMENT EDITION")
st.sidebar.caption("Local workstation / SAM 3 Cloud")

if st.sidebar.button("Refresh Inspection Data"):
    snapshot.clear()
    st.rerun()

if page == "Overview":
    overview()
elif page == "Inspection Studio":
    inspection_studio()
elif page == "Batch Center":
    batch_center()
elif page == "Results Library":
    results_library()
elif page == "AI Performance":
    ai_performance()
elif page == "ROV Operations":
    rov_operations()

st.divider()
st.caption(
    "OCEAN AI | Underwater Inspection Intelligence "
    "· Development prototype · "
    "Engineering verification required"
)
