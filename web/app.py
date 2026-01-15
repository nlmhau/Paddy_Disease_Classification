import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cho phép chạy `streamlit run web/app.py` mà không cần biến `src/` thành package.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import (
    LABEL_MAP,
    add_image_path_column,
    build_image_generators,
    default_data_paths,
    load_train_df,
    split_train_val,
)


APP_TITLE = "Hệ thống phân loại bệnh lá lúa"
DEFAULT_MODEL_PATHS = [
    "monster_cnn_best.keras",
    "src/monster_cnn_best.keras",
    "data/monster_cnn_best.keras",
]


@st.cache_data(show_spinner=False)
def discover_keras_models() -> List[str]:
    """Tự động quét các file *.keras trong project để người dùng chọn."""

    models: List[str] = []
    try:
        for p in ROOT_DIR.rglob("*.keras"):
            if p.is_file():
                models.append(str(p.relative_to(ROOT_DIR)))
    except Exception:
        models = []

    def sort_key(x: str) -> Tuple[int, str]:
        return (0 if os.path.basename(x).lower() == "monster_cnn_best.keras" else 1, x.lower())

    models = sorted(set(models), key=sort_key)

    # Giữ một vài gợi ý mặc định nếu không tìm được.
    if not models:
        models = DEFAULT_MODEL_PATHS.copy()

    return models


def model_picker(label: str) -> str:
    """UI chọn model: dropdown các file .keras + ô nhập tuỳ chọn."""

    models = discover_keras_models()
    default_idx = 0
    for i, p in enumerate(models):
        if os.path.basename(p).lower() == "monster_cnn_best.keras":
            default_idx = i
            break

    selected = st.selectbox(label, models, index=default_idx)
    custom = st.text_input("Hoặc nhập đường dẫn model (.keras) khác", value="")
    return custom.strip() if custom.strip() else selected


def _resolve_model_path(candidate: str) -> str | None:
    p = Path(candidate)
    if p.exists() and p.is_file():
        return str(p)
    return None


@st.cache_data(show_spinner=False)
def load_dataframe() -> pd.DataFrame:
    paths = default_data_paths()
    df = load_train_df(paths.train_csv, label_map=LABEL_MAP)
    df = add_image_path_column(df, paths.train_img_dir)
    return df


@st.cache_resource(show_spinner=False)
def get_class_indices(img_size: Tuple[int, int], batch_size: int) -> Dict[str, int]:
    df = load_dataframe()
    train_df, val_df = split_train_val(df, test_size=0.2, random_state=42)
    make_generators = build_image_generators(img_size=img_size, batch_size=batch_size)
    train_generator, _ = make_generators(train_df, val_df)
    return train_generator.class_indices


@st.cache_resource(show_spinner=False)
def load_tf_model(model_path: str, model_mtime: float | None = None):
    import tensorflow as tf

    if model_mtime is None:
        try:
            model_mtime = os.path.getmtime(model_path)
        except OSError:
            model_mtime = None

    # Try TensorFlow/Keras loader first.
    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        pass

    # Compatibility: some older models include Dense config with `quantization_config` which
    # newer/older Keras may not accept. Patch Dense to ignore this kwarg.
    try:
        from tensorflow.keras.layers import Dense as TfDense

        class PatchedDense(TfDense):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"Dense": PatchedDense},
        )
    except Exception:
        pass

    # Fallback for mismatched TF/Keras versions: avoid deserialization issues.
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        pass

    # Keras 3: safe_mode can block loading some older configs.
    try:
        return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:
        # safe_mode not supported in this TF/Keras version.
        pass
    except Exception:
        pass

    # If the model was saved with standalone Keras (keras==3), tf.keras loader may fail.
    try:
        import keras
    except Exception as e:
        raise RuntimeError(
            "Không thể load model bằng tf.keras (có thể lệch version save/load). Đồng thời không import được package 'keras'."
        ) from e

    def _keras_load_with_patched_dense(*, safe_mode: bool | None):
        from keras.layers import Dense as KDense

        class PatchedDenseKeras(KDense):
            def __init__(self, *args, **kwargs):
                kwargs.pop("quantization_config", None)
                super().__init__(*args, **kwargs)

        import keras.layers as _kl
        import keras

        try:
            from keras.utils import get_custom_objects as _get_custom_objects
        except Exception:
            _get_custom_objects = None

        # Keras deserialization often resolves the class from internal module path
        # `keras.src.layers.core.dense.Dense`, so we patch both aliases.
        try:
            import keras.src.layers.core.dense as _kd
        except Exception:
            _kd = None

        orig_dense = getattr(_kl, "Dense", None)
        _kl.Dense = PatchedDenseKeras
        orig_dense_src = getattr(_kd, "Dense", None) if _kd is not None else None
        if _kd is not None:
            _kd.Dense = PatchedDenseKeras

        # Also patch Keras global custom object registry, which is another resolution path
        # used during deserialization.
        orig_custom_dense = None
        orig_custom_keras_layers_dense = None
        orig_custom_keras_src_dense = None
        if _get_custom_objects is not None:
            custom = _get_custom_objects()
            orig_custom_dense = custom.get("Dense")
            orig_custom_keras_layers_dense = custom.get("keras.layers.Dense")
            orig_custom_keras_src_dense = custom.get("keras.src.layers.core.dense.Dense")
            custom["Dense"] = PatchedDenseKeras
            custom["keras.layers.Dense"] = PatchedDenseKeras
            custom["keras.src.layers.core.dense.Dense"] = PatchedDenseKeras
        try:
            if safe_mode is None:
                return keras.models.load_model(model_path, compile=False)
            return keras.models.load_model(model_path, compile=False, safe_mode=safe_mode)
        finally:
            if orig_dense is not None:
                _kl.Dense = orig_dense
            if _kd is not None and orig_dense_src is not None:
                _kd.Dense = orig_dense_src
            if _get_custom_objects is not None:
                custom = _get_custom_objects()
                if orig_custom_dense is None:
                    custom.pop("Dense", None)
                else:
                    custom["Dense"] = orig_custom_dense
                if orig_custom_keras_layers_dense is None:
                    custom.pop("keras.layers.Dense", None)
                else:
                    custom["keras.layers.Dense"] = orig_custom_keras_layers_dense
                if orig_custom_keras_src_dense is None:
                    custom.pop("keras.src.layers.core.dense.Dense", None)
                else:
                    custom["keras.src.layers.core.dense.Dense"] = orig_custom_keras_src_dense

    try:
        return keras.models.load_model(model_path)
    except Exception:
        pass

    try:
        return _keras_load_with_patched_dense(safe_mode=None)
    except Exception:
        pass

    try:
        return _keras_load_with_patched_dense(safe_mode=False)
    except TypeError:
        return _keras_load_with_patched_dense(safe_mode=None)


def predict_pil_image(model, img_pil: Image.Image, img_size: Tuple[int, int], idx_to_class: Dict[int, str], model_path: str | None = None) -> pd.DataFrame:
    img = img_pil.convert("RGB").resize(img_size)
    img_arr = np.array(img, dtype=np.float32)

    # Chọn preprocessing phù hợp với model
    if model_path:
        name = os.path.basename(model_path).lower()
        if "efficientnetb1" in name or "efnet_b1" in name:
            from tensorflow.keras.applications.efficientnet import preprocess_input
            img_arr = preprocess_input(img_arr)
        elif "efficientnetv2" in name or "efnetv2" in name:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
            img_arr = preprocess_input(img_arr)
        else:
            img_arr = img_arr / 255.0
    else:
        img_arr = img_arr / 255.0

    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr, verbose=0)[0]
    df_prob = pd.DataFrame(
        {
            "Loại bệnh": [idx_to_class[i] for i in range(len(preds))],
            "Xác suất (%)": preds * 100,
        }
    ).sort_values("Xác suất (%)", ascending=False)

    return df_prob


def _iter_layers_recursive(model):
    import tensorflow as tf

    for layer in getattr(model, "layers", []):
        yield layer
        if isinstance(layer, tf.keras.Model):
            yield from _iter_layers_recursive(layer)


def _find_last_conv_layer_name(model) -> str | None:
    import tensorflow as tf

    conv_layers = []
    for layer in _iter_layers_recursive(model):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer)
    if not conv_layers:
        return None
    return conv_layers[-1].name


def grad_cam_heatmap(model, img_array: np.ndarray, layer_name: str | None = None) -> np.ndarray:
    import tensorflow as tf

    # đảm bảo model đã có input/output graph
    _ = model.predict(img_array, verbose=0)

    if layer_name is None:
        layer_name = _find_last_conv_layer_name(model)
    if layer_name is None:
        raise ValueError("Không tìm thấy Conv2D layer để tạo Grad-CAM")

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(img_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = img_pil.convert("RGB")
    w, h = base.size

    heat = Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), resample=Image.BILINEAR)
    heat_arr = np.array(heat, dtype=np.float32) / 255.0

    cmap = cm.get_cmap("jet")
    colored = cmap(heat_arr)[:, :, :3]
    colored_img = Image.fromarray(np.uint8(colored * 255)).convert("RGB")

    return Image.blend(base, colored_img, alpha=alpha)


def card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="border-radius:16px;padding:16px 18px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12)">
          <div style="font-size:18px;font-weight:700;margin-bottom:6px">{title}</div>
          <div style="color:rgba(255,255,255,0.85);line-height:1.6">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_home() -> None:
    st.header("Trang chủ")
    st.subheader("Giới thiệu tổng quan")

    card(
        "Mục tiêu",
        "Xây dựng hệ thống phân loại bệnh lá lúa dựa trên ảnh. Pipeline bám sát theo bài: EDA → tiền xử lý/augmentation → huấn luyện mô hình → đánh giá → dự đoán ảnh đầu vào.",
    )

    df = load_dataframe()

    c1, c2, c3 = st.columns(3)
    c1.metric("Số ảnh (train.csv)", f"{len(df):,}")
    c2.metric("Số giống lúa", f"{df['variety'].nunique():,}")
    c3.metric("Số loại bệnh", f"{df['label_vi'].nunique():,}")

    st.markdown("---")

    st.subheader("Sơ đồ trang")
    st.markdown(
        """
- **Phân tích dữ liệu**: biểu đồ 2D/3D tương tác (Plotly)
- **Đánh giá mô hình**: confusion matrix, classification report
- **Dự đoán ảnh**: upload ảnh và dự đoán bệnh (model `.keras`)
        """
    )


def page_eda() -> None:
    st.header("Phân tích dữ liệu (EDA)")
    df = load_dataframe()

    st.subheader("Phân bố số lượng ảnh theo loại bệnh (2D)")
    counts = df["label_vi"].value_counts().reset_index()
    counts.columns = ["Loại bệnh", "Số lượng"]
    fig_bar = px.bar(
        counts,
        x="Loại bệnh",
        y="Số lượng",
        title="Phân bố các loại bệnh trên lúa",
    )
    fig_bar.update_layout(xaxis_tickangle=-35, height=520)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Tỉ lệ phân bố bệnh (2D - Pie)")
    fig_pie = px.pie(counts, names="Loại bệnh", values="Số lượng", title="Tỉ lệ phân bố các loại bệnh")
    fig_pie.update_layout(height=520)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Phân bố tuổi lúa theo bệnh (2D - Boxplot)")
    fig_box = px.box(df, x="label_vi", y="age", points="outliers", title="Phân bố tuổi cây lúa theo từng loại bệnh")
    fig_box.update_layout(xaxis_tickangle=-35, height=560)
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Biểu đồ 3D tương tác: Số lượng mẫu theo (Giống lúa, Loại bệnh)")
    agg = df.groupby(["variety", "label_vi"]).size().reset_index(name="count")
    fig_3d = px.scatter_3d(
        agg,
        x="variety",
        y="label_vi",
        z="count",
        color="label_vi",
        size="count",
        title="3D: Variety × Label → Count",
    )
    fig_3d.update_layout(height=700)
    st.plotly_chart(fig_3d, use_container_width=True)

    with st.expander("Xem dữ liệu mẫu"):
        st.dataframe(df.head(50), use_container_width=True)

def page_evaluation() -> None:
    st.header("Đánh giá mô hình")
 
    df = load_dataframe()
    st.caption("Trang này chỉ so sánh 3 mô hình chính theo bài (cùng validation).")
 
    st.subheader("So sánh 3 mô hình (cùng validation)")
    models = discover_keras_models()
 
    wanted = {
        "monster_cnn_best.keras",
        "efnet_b1_best.keras",
        "efnetv2_v2s_best.keras",
    }
    selected_models = []
    for m in models:
        if os.path.basename(m).lower() in wanted:
            selected_models.append(m)
 
    # Fallback nếu không discover được
    if not selected_models:
        selected_models = [
            "monster_cnn_best.keras",
            "efnet_b1_best.keras",
            "efnetv2_v2s_best.keras",
        ]

    img_size = (224, 224)
    batch_size = 32

    cbtn1, cbtn2 = st.columns([1, 2])
    with cbtn1:
        if st.button("Clear cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    run_compare = st.button("Chạy so sánh", use_container_width=True)
    if run_compare:
        with st.spinner("Đang chuẩn bị validation và đánh giá 3 model..."):
            from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effv2_preprocess
            import traceback
 
            train_df, val_df = split_train_val(df, test_size=0.2, random_state=42)
 
            def make_generator_for_model(model_path: str):
                name = os.path.basename(model_path).lower()
                current_img_size = (224, 224)
                if "efficientnetb1" in name or "efnet_b1" in name:
                    current_img_size = (240, 240)
                    train_gen = ImageDataGenerator(preprocessing_function=eff_preprocess)
                    val_gen = ImageDataGenerator(preprocessing_function=eff_preprocess)
                elif "efficientnetv2" in name or "efnetv2" in name:
                    train_gen = ImageDataGenerator(preprocessing_function=effv2_preprocess)
                    val_gen = ImageDataGenerator(preprocessing_function=effv2_preprocess)
                else:
                    train_gen = ImageDataGenerator(rescale=1.0 / 255)
                    val_gen = ImageDataGenerator(rescale=1.0 / 255)
                make = build_image_generators(
                    img_size=current_img_size,
                    batch_size=batch_size,
                    train_datagen=train_gen,
                    val_datagen=val_gen,
                )
                return make(train_df, val_df)[1]
 
            results = []
            error_traces = []
            for mp in selected_models:
                resolved_mp = _resolve_model_path(str(ROOT_DIR / mp)) or _resolve_model_path(mp)
                if resolved_mp is None:
                    continue
                try:
                    model = load_tf_model(resolved_mp, model_mtime=os.path.getmtime(resolved_mp))
                except Exception as e:
                    error_traces.append(
                        {
                            "model": os.path.basename(resolved_mp),
                            "stage": "load",
                            "trace": traceback.format_exc(),
                        }
                    )
                    results.append(
                        {
                            "Mô hình": os.path.basename(resolved_mp),
                            "Val loss": None,
                            "Val accuracy": None,
                            "Ghi chú": f"Không load được model: {type(e).__name__}: {e}",
                        }
                    )
                    continue
 
                val_generator = make_generator_for_model(resolved_mp)
                val_generator.reset()
                try:
                    loss, acc = model.evaluate(val_generator, verbose=0)
                    results.append(
                        {
                            "Mô hình": os.path.basename(resolved_mp),
                            "Val loss": float(loss),
                            "Val accuracy": float(acc),
                            "Ghi chú": "",
                        }
                    )
                except Exception as e:
                    error_traces.append(
                        {
                            "model": os.path.basename(resolved_mp),
                            "stage": "evaluate",
                            "trace": traceback.format_exc(),
                        }
                    )
                    results.append(
                        {
                            "Mô hình": os.path.basename(resolved_mp),
                            "Val loss": None,
                            "Val accuracy": None,
                            "Ghi chú": f"Evaluate lỗi: {type(e).__name__}: {e}",
                        }
                    )
 
        if results:
            res_df = pd.DataFrame(results)
            res_df = res_df.sort_values("Val accuracy", ascending=False, na_position="last")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(res_df, use_container_width=True)
            with c2:
                chart_df = res_df.dropna(subset=["Val accuracy"])
                fig_cmp = px.bar(chart_df, x="Mô hình", y="Val accuracy", title="So sánh Val Accuracy")
                fig_cmp.update_layout(height=420)
                st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.warning("Không có model hợp lệ để so sánh.")

        if "error_traces" in locals() and error_traces:
            with st.expander("Chi tiết lỗi (traceback)"):
                for it in error_traces:
                    st.markdown(f"**{it['model']}** — `{it['stage']}`")
                    st.code(it["trace"], language="text")


def page_predict() -> None:
    st.header("Dự đoán bệnh từ ảnh")
    st.caption("Upload ảnh lá lúa để model dự đoán loại bệnh (bám sát phần upload & predict trong notebook).")

    model_path = model_picker("Model dùng để dự đoán")

    if not model_path:
        st.info("Bạn chưa nhập đường dẫn model. Nếu chưa có model, hãy train trong các file `src/model_*.py`. ")
        return

    resolved = _resolve_model_path(model_path)
    if resolved is None:
        st.error("Không tìm thấy file model. Hãy kiểm tra lại đường dẫn.")
        return

    img_size = (224, 224)
    batch_size = 32

    with st.spinner("Đang load model..."):
        try:
            model = load_tf_model(resolved)
        except Exception as e:
            st.error(
                "Không load được model. Model có thể được train sai input (ví dụ ảnh grayscale 1 kênh) hoặc khác kích thước đầu vào."
            )
            st.code(str(e))
            return

    class_indices = get_class_indices(img_size=img_size, batch_size=batch_size)
    idx_to_class = {v: k for k, v in class_indices.items()}

    uploaded = st.file_uploader("Chọn ảnh (jpg/png)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.stop()

    img = Image.open(uploaded)
    st.image(img, caption="Ảnh bạn đã upload", use_container_width=True)

    with st.spinner("Đang dự đoán..."):
        df_prob = predict_pil_image(model, img, img_size=img_size, idx_to_class=idx_to_class, model_path=resolved)

    top = df_prob.iloc[0]
    st.success(f"Dự đoán: {top['Loại bệnh']} — Độ tin cậy: {top['Xác suất (%)']:.2f}%")

    st.subheader("Bảng xác suất")
    st.dataframe(df_prob, use_container_width=True)

    st.subheader("Biểu đồ xác suất (2D)")
    fig = px.bar(df_prob.head(10), x="Loại bệnh", y="Xác suất (%)", title="Top xác suất dự đoán")
    fig.update_layout(xaxis_tickangle=-35, height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    show_cam = st.toggle("Hiển thị Grad-CAM (Monster CNN)", value=False)
    if show_cam:
        st.subheader("Giải thích dự đoán (Grad-CAM)")
        st.caption(f"Heatmap cho model: {os.path.basename(resolved)}")

        try:
            img_resized = img.convert("RGB").resize(img_size)
            
            # Xử lý Preprocessing tương ứng cho từng loại model để Grad-CAM chính xác
            name = os.path.basename(resolved).lower()
            arr = np.array(img_resized, dtype=np.float32)
            if "efnet" in name:
                from tensorflow.keras.applications.efficientnet import preprocess_input
                arr = preprocess_input(arr)
            else:
                arr = arr / 255.0
            
            arr = np.expand_dims(arr, axis=0)

            # Tự động tìm lớp Convolutional cuối cùng của từng model
            layer_name = _find_last_conv_layer_name(model)
            
            if layer_name:
                heatmap = grad_cam_heatmap(model, arr, layer_name=layer_name)
                overlay = overlay_heatmap_on_image(img_resized, heatmap, alpha=0.45)
                st.image(overlay, caption=f"Vùng tập trung của mô hình (Layer: {layer_name})", use_container_width=True)
            else:
                st.warning("Không tìm thấy lớp Conv2D phù hợp để tạo Grad-CAM.")
        except Exception as e:
            st.error(f"Lỗi tạo Grad-CAM: {e}")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
          .stApp {
            background: radial-gradient(1200px 600px at 10% 10%, rgba(80, 180, 255, 0.18), transparent 60%),
                        radial-gradient(1000px 500px at 90% 20%, rgba(0, 255, 170, 0.12), transparent 55%),
                        linear-gradient(180deg, #0B1220 0%, #070B14 100%);
            color: #E6EDF3;
          }
          h1, h2, h3, h4, h5, h6 { color: #E6EDF3; }
          [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.04);
            border-right: 1px solid rgba(255,255,255,0.10);
          }
          [data-testid="stMetric"] {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            padding: 12px 14px;
            border-radius: 14px;
          }
          .block-container { padding-top: 1.5rem; }
          .stButton>button {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(255,255,255,0.08);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(APP_TITLE)

    menu = st.sidebar.radio(
        "Điều hướng",
        ["Trang chủ", "Phân tích dữ liệu", "Đánh giá mô hình", "Dự đoán ảnh"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Bám sát theo notebook: EDA → preprocess → model → evaluation → predict")

    if menu == "Trang chủ":
        page_home()
    elif menu == "Phân tích dữ liệu":
        page_eda()
    elif menu == "Đánh giá mô hình":
        page_evaluation()
    else:
        page_predict()


if __name__ == "__main__":
    main()
