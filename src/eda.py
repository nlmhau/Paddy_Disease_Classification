import os
import random
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def plot_variety_distribution(df: pd.DataFrame) -> None:
    """Vẽ phân bố giống lúa (variety) như notebook."""

    variety_order = df["variety"].value_counts().index
    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x="variety", order=variety_order)
    plt.title("Phân bố giống lúa", fontsize=14)
    plt.xlabel("Giống lúa")
    plt.ylabel("Số lượng")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_label_distribution_vi(df: pd.DataFrame) -> None:
    """Vẽ phân bố nhãn bệnh (label_vi) dạng bar + pie như notebook."""

    label_counts = df["label_vi"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title("Phân bố các loại bệnh trên lúa", fontsize=14)
    plt.xlabel("Loại bệnh")
    plt.ylabel("Số lượng ảnh")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.pie(label_counts.values, labels=label_counts.index, autopct="%.1f%%", startangle=90)
    plt.title("Tỉ lệ phân bố các loại bệnh trên lúa", fontsize=14)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def age_statistics_by_disease(df: pd.DataFrame) -> pd.DataFrame:
    """Thống kê tuổi lúa theo từng bệnh (groupby+describe) như notebook."""

    return df.groupby("label_vi")["age"].describe().round(2)


def plot_age_boxplot(df: pd.DataFrame) -> None:
    """Boxplot tuổi lúa theo từng bệnh như notebook."""

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="label_vi", y="age")
    plt.xticks(rotation=45, ha="right")
    plt.title("Phân bố tuổi cây lúa theo từng loại bệnh")
    plt.xlabel("Loại bệnh")
    plt.ylabel("Tuổi cây lúa (ngày)")
    plt.tight_layout()
    plt.show()


def analyze_image_size_and_channels(train_img_dir: str) -> Tuple[Tuple[int, int, float], Tuple[int, int, float], set[int]]:
    """Quét toàn bộ ảnh để lấy thống kê width/height/channels giống notebook."""

    widths: list[int] = []
    heights: list[int] = []
    channels: set[int] = set()

    for label in os.listdir(train_img_dir):
        label_path = os.path.join(train_img_dir, label)
        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    channels.add(len(img.getbands()))
            except Exception:
                continue

    width_stats = (min(widths), max(widths), sum(widths) / len(widths))
    height_stats = (min(heights), max(heights), sum(heights) / len(heights))
    return width_stats, height_stats, channels


def plot_image_size_histograms(widths: Iterable[int], heights: Iterable[int]) -> None:
    """Vẽ histogram width/height giống notebook."""

    widths = list(widths)
    heights = list(heights)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=10, color="skyblue", edgecolor="black")
    plt.axvline(sum(widths) / len(widths), color="red", linestyle="--", label=f"Mean: {sum(widths)/len(widths):.0f}")
    plt.title("Phân bố chiều rộng của ảnh")
    plt.xlabel("Chiều rộng (pixels)")
    plt.ylabel("Số lượng")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=10, color="lightgreen", edgecolor="black")
    plt.axvline(sum(heights) / len(heights), color="red", linestyle="--", label=f"Mean: {sum(heights)/len(heights):.0f}")
    plt.title("Phân bố chiều cao của ảnh")
    plt.xlabel("Chiều cao (pixels)")
    plt.ylabel("Số lượng")
    plt.legend()

    plt.tight_layout()
    plt.show()


def show_sample_images_by_class(
    data_dir: str,
    label_vi_map: Dict[str, str],
    samples_per_class: int = 3,
    img_size: Tuple[int, int] = (224, 224),
) -> None:
    """Hiển thị ảnh mẫu của mỗi class giống hàm `show_anh_mau_cua_banh` trong notebook."""

    classes = sorted(os.listdir(data_dir))
    classes = [c for c in classes if os.path.isdir(os.path.join(data_dir, c))]

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        images = os.listdir(class_path)

        label_vi = label_vi_map.get(class_name, class_name)
        sample_images = random.sample(images, min(samples_per_class, len(images)))

        plt.figure(figsize=(10, 3))
        plt.suptitle(f"Loại bệnh: {label_vi}", fontsize=14)

        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).resize(img_size)

            plt.subplot(1, samples_per_class, i + 1)
            plt.imshow(img)
            plt.axis("off")

        plt.tight_layout()
        plt.show()


def analyze_brightness_contrast(train_img_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Tính brightness/contrast (mean/std grayscale) như notebook."""

    brightness_values: list[float] = []
    contrast_values: list[float] = []

    for label in os.listdir(train_img_dir):
        label_path = os.path.join(train_img_dir, label)
        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            try:
                img = Image.open(img_path).convert("L")
                img_array = np.array(img)
                brightness_values.append(float(img_array.mean()))
                contrast_values.append(float(img_array.std()))
            except Exception:
                continue

    return np.array(brightness_values), np.array(contrast_values)


def plot_brightness_contrast_histograms(brightness_values: np.ndarray, contrast_values: np.ndarray) -> None:
    """Vẽ histogram brightness/contrast như notebook."""

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(brightness_values, bins=30, color="orange", edgecolor="black")
    plt.axvline(np.mean(brightness_values), color="red", linestyle="--", label=f"Mean: {np.mean(brightness_values):.1f}")
    plt.title("Phân bố độ sáng ảnh")
    plt.xlabel("Brightness (0–255)")
    plt.ylabel("Số lượng")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(contrast_values, bins=30, color="purple", edgecolor="black")
    plt.axvline(np.mean(contrast_values), color="red", linestyle="--", label=f"Mean: {np.mean(contrast_values):.1f}")
    plt.title("Phân bố độ tương phản ảnh")
    plt.xlabel("Contrast (Std of pixel values)")
    plt.ylabel("Số lượng")
    plt.legend()

    plt.tight_layout()
    plt.show()
