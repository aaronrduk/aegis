import streamlit as st
import numpy as np
import tempfile
import zipfile
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go

from src.config import CLASS_NAMES, CLASS_COLORS
from src.inference import SVAMITVAInference
from src.postprocess import postprocess_multiclass_mask
from src.vectorize import mask_to_shapefiles
from src.utils import calculate_area, count_objects

st.set_page_config(
    page_title="SVAMITVA Feature Extraction",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path):
    return SVAMITVAInference(checkpoint_path, use_tta=True)


def create_color_legend():
    fig = go.Figure()
    for idx, name in enumerate(CLASS_NAMES):
        if idx == 0:
            continue
        color = CLASS_COLORS[idx]
        fig.add_trace(go.Bar(
            x=[1],
            y=[name],
            orientation="h",
            marker=dict(color=f"rgb({color[0]},{color[1]},{color[2]})"),
            name=name,
            showlegend=False,
        ))
    fig.update_layout(
        title="Feature Classes",
        xaxis=dict(visible=False),
        yaxis=dict(title=""),
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def calculate_statistics(mask, pixel_size=1.0):
    stats = []
    for class_idx in range(1, len(CLASS_NAMES)):
        class_name = CLASS_NAMES[class_idx]
        area = calculate_area(mask, class_idx, pixel_size)
        count = count_objects(mask, class_idx)
        if area > 0 or count > 0:
            stats.append({
                "Class": class_name,
                "Count": count,
                "Area (m²)": f"{area:.2f}",
                "Color": f"rgb({CLASS_COLORS[class_idx][0]},{CLASS_COLORS[class_idx][1]},{CLASS_COLORS[class_idx][2]})",
            })
    return stats


def create_statistics_chart(stats):
    if not stats:
        return None
    classes = [s["Class"] for s in stats]
    areas = [float(s["Area (m²)"]) for s in stats]
    colors = [s["Color"] for s in stats]
    fig = go.Figure([go.Bar(
        x=classes,
        y=areas,
        marker_color=colors,
        text=areas,
        texttemplate="%{text:.0f} m²",
        textposition="auto",
    )])
    fig.update_layout(
        title="Feature Areas",
        xaxis_title="Feature Class",
        yaxis_title="Area (m²)",
        height=400,
    )
    return fig


def build_colored_mask(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        colored[mask == class_idx] = color
    return colored


def main():
    st.markdown(
        '<div class="main-header">🛰️ SVAMITVA Feature Extraction</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    **AI-powered feature extraction from drone imagery for SVAMITVA Scheme**

    This application extracts:
    - 🏠 Building footprints (with roof-type classification: RCC, Tiled, Tin, Others)
    - 🛣️ Roads
    - 💧 Waterbodies
    - ⚡ Infrastructure (Transformers, Tanks, Wells)
    """)

    with st.sidebar:
        st.header("⚙️ Configuration")

        checkpoint_path = st.text_input(
            "Model Checkpoint Path",
            value="checkpoints/best_model.pth",
            help="Path to the trained model checkpoint",
        )

        if not Path(checkpoint_path).exists():
            st.warning(f"⚠️ Model checkpoint not found: {checkpoint_path}")
            st.info("💡 Using demo model (random weights) for demonstration purposes.")

        model = None
        with st.spinner("Loading model..."):
            try:
                model = load_model(checkpoint_path)
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load model: {e}")

        st.subheader("🎯 Features to Extract")
        extract_buildings = st.checkbox("Buildings", value=True)
        extract_roads = st.checkbox("Roads", value=True)
        extract_waterbodies = st.checkbox("Waterbodies", value=True)
        extract_infrastructure = st.checkbox("Infrastructure", value=True)

        st.subheader("🔧 Post-processing")
        apply_postprocess = st.checkbox("Apply post-processing", value=True)
        simplify_tolerance = st.slider("Polygon simplification", 0.0, 5.0, 1.0, 0.5)

        st.subheader("📤 Output Options")
        separate_shapefiles = st.checkbox("Separate shapefiles per class", value=True)
        pixel_size = st.number_input("Pixel size (m)", value=0.1, min_value=0.01, step=0.01)

    if model is None:
        st.warning("Model is not available. Please check your checkpoint path in the sidebar.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<div class="sub-header">📁 Upload Image</div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Choose a drone image (TIF, JPEG, PNG)",
                type=["tif", "tiff", "jpg", "jpeg", "png"],
            )

            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded_file.name).suffix
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    input_path = tmp_file.name

                st.image(uploaded_file, caption="Original Image", use_container_width=True)

                if st.button("🚀 Extract Features", type="primary", use_container_width=True):
                    with st.spinner("Processing image..."):
                        try:
                            mask, probs, metadata = model.predict_file(input_path)

                            if apply_postprocess:
                                from src.config import POSTPROCESS_CONFIG
                                mask = postprocess_multiclass_mask(
                                    mask, POSTPROCESS_CONFIG["min_area"], len(CLASS_NAMES)
                                )

                            if not extract_buildings:
                                mask[np.isin(mask, [1, 2, 3, 4])] = 0
                            if not extract_roads:
                                mask[mask == 5] = 0
                            if not extract_waterbodies:
                                mask[mask == 6] = 0
                            if not extract_infrastructure:
                                mask[np.isin(mask, [7, 8, 9])] = 0

                            st.session_state["mask"] = mask
                            st.session_state["colored_mask"] = build_colored_mask(mask)
                            st.session_state["probs"] = probs
                            st.session_state["metadata"] = metadata
                            st.session_state["input_path"] = input_path

                            st.success("✅ Feature extraction completed!")
                        except Exception as e:
                            st.error(f"❌ Error during processing: {e}")

        with col2:
            if "mask" in st.session_state:
                st.markdown(
                    '<div class="sub-header">🎨 Prediction Results</div>',
                    unsafe_allow_html=True,
                )
                st.image(
                    st.session_state["colored_mask"],
                    caption="Extracted Features",
                    use_container_width=True,
                )
                st.plotly_chart(create_color_legend(), use_container_width=True)

        if "mask" in st.session_state:
            mask = st.session_state["mask"]
            colored_mask = st.session_state["colored_mask"]

            st.markdown('<div class="sub-header">📊 Statistics</div>', unsafe_allow_html=True)

            stats = calculate_statistics(mask, pixel_size)
            if stats:
                stat_col1, stat_col2 = st.columns([1, 1])
                with stat_col1:
                    st.dataframe(stats, use_container_width=True, hide_index=True)
                with stat_col2:
                    chart = create_statistics_chart(stats)
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
            else:
                st.info("No features detected in the image.")

            st.markdown(
                '<div class="sub-header">⬇️ Download Results</div>',
                unsafe_allow_html=True,
            )

            dl_col1, dl_col2, dl_col3 = st.columns(3)

            with dl_col1:
                mask_img = Image.fromarray(mask.astype(np.uint8))
                mask_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                mask_img.save(mask_tmp.name)
                with open(mask_tmp.name, "rb") as f:
                    st.download_button(
                        label="📥 Download Mask (PNG)",
                        data=f,
                        file_name="segmentation_mask.png",
                        mime="image/png",
                        use_container_width=True,
                    )

            with dl_col2:
                colored_img = Image.fromarray(colored_mask)
                colored_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                colored_img.save(colored_tmp.name)
                with open(colored_tmp.name, "rb") as f:
                    st.download_button(
                        label="📥 Download Colored View",
                        data=f,
                        file_name="colored_mask.png",
                        mime="image/png",
                        use_container_width=True,
                    )

            with dl_col3:
                if st.button("🗺️ Generate Shapefiles", use_container_width=True):
                    with st.spinner("Creating shapefiles..."):
                        try:
                            output_dir = tempfile.mkdtemp()
                            transform = st.session_state["metadata"].get("transform", None)
                            crs = st.session_state["metadata"].get("crs", None)

                            mask_to_shapefiles(
                                mask,
                                output_dir=output_dir,
                                base_name="features",
                                class_names=CLASS_NAMES,
                                transform=transform,
                                crs=crs,
                                simplify_tolerance=simplify_tolerance,
                                separate_classes=separate_shapefiles,
                            )

                            zip_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                            with zipfile.ZipFile(zip_tmp.name, "w") as zipf:
                                for file in Path(output_dir).rglob("*"):
                                    if file.is_file():
                                        zipf.write(file, file.name)

                            with open(zip_tmp.name, "rb") as f:
                                st.download_button(
                                    label="📥 Download Shapefiles (ZIP)",
                                    data=f,
                                    file_name="shapefiles.zip",
                                    mime="application/zip",
                                    use_container_width=True,
                                )
                            st.success("✅ Shapefiles generated!")
                        except Exception as e:
                            st.error(f"❌ Error creating shapefiles: {e}")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>SVAMITVA Feature Extraction System | Built for Hackathon 2026</p>
        <p>Powered by DeepLabV3+ and PyTorch</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
