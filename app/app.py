"""Streamlit demo: predict the period of a held-out tablet (or an uploaded .ply).

Launch from the repository root:

    .\\.venv\\Scripts\\python.exe -m streamlit run app/app.py

All modelling goes through :class:`cuneiform.inference.Predictor`; this file only
handles layout.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from cuneiform.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config
from cuneiform.data.cache import DepthMapCache
from cuneiform.data.manifest import read_manifest
from cuneiform.inference import Prediction, Predictor
from cuneiform.modeling.bundle import ModelBundleNotFoundError
from cuneiform.visualization import depth_to_rgb, gradcam_overlay, shaded_to_rgb

CONFIG_PATH = Path(os.environ.get("CUNEIFORM_CONFIG", DEFAULT_CONFIG_PATH))
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"

st.set_page_config(page_title="Cuneiform Period Classifier", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> Predictor:
    return Predictor.from_latest(CONFIG_PATH)


@st.cache_data(show_spinner=False)
def held_out_tablets(split: str) -> list[tuple[str, str]]:
    """``[(tablet_id, true period)]`` for cached tablets in ``split``."""
    config = load_config(CONFIG_PATH)
    cache = DepthMapCache(config.paths.processed_dir, config.preprocessing)
    rows = read_manifest(config.paths.manifest)
    return sorted((r.tablet_id, r.period) for r in rows if r.split == split and cache.has(r.tablet_id))


def show_prediction(result: Prediction, true_period: str | None = None) -> None:
    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted period", result.predicted_period)
        st.metric("Model confidence", f"{result.confidence:.1%}")
        if true_period:
            if true_period == result.predicted_period:
                st.success(f"Correct. Catalogued period (CDLI): **{true_period}**")
            else:
                st.error(f"Incorrect. Catalogued period (CDLI): **{true_period}**")
        st.caption("Class probabilities")
        for name, probability in sorted(result.probabilities.items(), key=lambda kv: -kv[1]):
            st.progress(probability, text=f"{name}: {probability:.1%}")
        st.caption(
            "Confidence is the model's softmax probability, not a calibrated likelihood. "
            f"Model run: `{result.model_run}`."
        )
    with right:
        for face in result.faces:
            st.markdown(f"**Face: {face.face}**")
            c1, c2, c3 = st.columns(3)
            c1.image(depth_to_rgb(face.depth_map), caption="Depth map (yellow = higher)", width="stretch")
            c2.image(shaded_to_rgb(face.depth_map), caption="Shaded relief", width="stretch")
            c3.image(gradcam_overlay(face.depth_map, face.gradcam),
                     caption=f"Grad-CAM for '{result.predicted_period}'", width="stretch")
        st.info(
            "The Grad-CAM overlay highlights **regions that influenced the prediction** (red = stronger). "
            "It shows where the network looked, not why a scholar would date the tablet this way."
        )


st.title("3D Cuneiform Tablet Period Classifier")
st.write(
    "A ResNet18 fine-tuned on depth maps rendered from 3D scans in the "
    "[HeiCuBeDa Hilprecht](https://doi.org/10.11588/data/IE8CCN) dataset (CC BY-SA 4.0) "
    "predicts whether a tablet is **Ur III** or **Old Babylonian**."
)

try:
    predictor = get_predictor()
except ModelBundleNotFoundError as error:
    st.error(f"**No trained model available.** {error}")
    st.markdown(
        "Create one from the repository root (see README):\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\python.exe -m cuneiform.cli prepare\n"
        ".\\.venv\\Scripts\\python.exe -m cuneiform.cli train\n"
        "```"
    )
    st.stop()

with st.sidebar:
    st.header("Model")
    metadata = predictor.bundle.metadata
    st.write(f"Run: `{metadata.get('run_name')}`")
    st.write(f"Classes: {', '.join(predictor.class_names)}")
    st.write(f"Faces used: {', '.join(predictor.config.preprocessing.faces)}")
    test_metrics = metadata.get("metrics_test")
    if test_metrics:
        st.write(
            f"Test ({test_metrics['num_tablets']} tablets): accuracy "
            f"**{test_metrics['model']['accuracy']:.1%}**, macro F1 **{test_metrics['model']['macro_f1']:.2f}** "
            f"(majority baseline {test_metrics['majority_baseline']['accuracy']:.1%})"
        )
    else:
        st.write("Test metrics: not computed yet (`cuneiform.cli evaluate`).")

tab_sample, tab_upload = st.tabs(["Held-out tablet", "Upload a .ply scan"])

with tab_sample:
    tablets = held_out_tablets("test")
    if not tablets:
        st.warning("No held-out (test split) tablets have cached depth maps. Run `python -m cuneiform.cli prepare`.")
    else:
        hide_label = st.checkbox("Hide the catalogued period until after analysis", value=True)
        labels = {tid: period for tid, period in tablets}
        choice = st.selectbox(
            f"Held-out tablet ({len(tablets)} in the test split, never used for training or model selection)",
            [tid for tid, _ in tablets],
            format_func=lambda tid: tid if hide_label else f"{tid} ({labels[tid]})",
        )
        if st.button("Analyze", type="primary", key="analyze_sample"):
            with st.spinner(f"Running the model on {choice}..."):
                st.session_state["sample_result"] = (predictor.predict_tablet(choice), labels[choice])
        if "sample_result" in st.session_state:
            result, true_period = st.session_state["sample_result"]
            st.subheader(result.tablet_id)
            show_prediction(result, true_period)

with tab_upload:
    st.write(
        "Upload a triangle mesh (.ply, millimetre units recommended). It is rendered with the same "
        "orientation and depth-map settings used in training. Large HeiCuBeDa scans (100+ MB) take a minute."
    )
    upload = st.file_uploader("Scan file", type=["ply"])
    if upload is not None and st.button("Analyze upload", type="primary"):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=".ply", delete=False) as handle:
            handle.write(upload.getbuffer())
            temporary = Path(handle.name)
        try:
            with st.spinner("Rendering depth map and predicting..."):
                st.session_state["upload_result"] = predictor.predict_scan(temporary, tablet_id=Path(upload.name).stem)
        except ValueError as error:  # InvalidScanError is a ValueError
            st.error(f"Could not process this scan: {error}")
        finally:
            temporary.unlink(missing_ok=True)
    if "upload_result" in st.session_state:
        result = st.session_state["upload_result"]
        st.subheader(result.tablet_id or "Uploaded scan")
        show_prediction(result)
