import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator


LABEL_MAP: Dict[str, str] = {
    "bacterial_leaf_blight": "Bệnh bạc lá (vi khuẩn)",
    "bacterial_leaf_streak": "Bệnh sọc lá vi khuẩn",
    "bacterial_panicle_blight": "Bệnh thối bông vi khuẩn",
    "blast": "Bệnh đạo ôn",
    "brown_spot": "Bệnh đốm nâu",
    "dead_heart": "Bệnh chết tim",
    "downy_mildew": "Bệnh sương mai",
    "hispa": "Bệnh sâu hispa",
    "normal": "Lá khỏe mạnh",
    "tungro": "Bệnh vàng lùn (Tungro)",
}


@dataclass
class DataPaths:
    """Nhóm đường dẫn dữ liệu dùng chung cho pipeline."""

    train_csv: str
    train_img_dir: str


def default_data_paths(base_dir: str | None = None) -> DataPaths:
    """Tạo đường dẫn mặc định theo cấu trúc project hiện tại.

    - Notebook đang đặt `BASE_DIR = ".."` từ thư mục `src/`.
    - Data thật trong workspace đang nằm dưới `data/`.
    """

    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    data_dir = os.path.join(base_dir, "data")
    return DataPaths(
        train_csv=os.path.join(data_dir, "train.csv"),
        train_img_dir=os.path.join(data_dir, "train_images"),
    )


def load_train_df(train_csv: str, label_map: Dict[str, str] | None = None) -> pd.DataFrame:
    """Đọc `train.csv` và tạo cột `label_vi` giống notebook."""

    if label_map is None:
        label_map = LABEL_MAP

    df = pd.read_csv(train_csv)
    df["label_vi"] = df["label"].map(label_map).fillna(df["label"])
    return df


def add_image_path_column(df: pd.DataFrame, train_img_dir: str) -> pd.DataFrame:
    """Tạo cột `image_path` theo logic notebook: train_images / label / image_id."""

    def get_img_path(row: pd.Series) -> str:
        return os.path.join(train_img_dir, row["label"], row["image_id"])

    out = df.copy()
    out["image_path"] = out.apply(get_img_path, axis=1)
    return out


def split_train_val(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chia train/validation theo `label_vi` (stratify) giống notebook."""

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_vi"],
    )
    return train_df, val_df


def build_image_generators(
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    train_datagen: ImageDataGenerator | None = None,
    val_datagen: ImageDataGenerator | None = None,
):
    """Khởi tạo train/val generator bằng `flow_from_dataframe` (y_col=`label_vi`)."""

    if train_datagen is None:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

    if val_datagen is None:
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    def make_generators(train_df: pd.DataFrame, val_df: pd.DataFrame):
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col="image_path",
            y_col="label_vi",
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
        )

        val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col="image_path",
            y_col="label_vi",
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
        )

        return train_generator, val_generator

    return make_generators


def compute_class_weights(train_df: pd.DataFrame, class_indices: Dict[str, int]) -> Dict[int, float]:
    """Tính `class_weights_dict` và map đúng với output neuron như notebook."""

    labels = train_df["label_vi"].values
    classes = np.unique(labels)

    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    return {class_indices[label]: weight for label, weight in zip(classes, weights)}
