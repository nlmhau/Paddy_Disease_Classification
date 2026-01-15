from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def plot_history(history) -> None:
    """Vẽ Accuracy/Loss train vs validation giống notebook."""

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("So sánh Accuracy Train vs Validation")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("So sánh Loss Train vs Validation")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def predict_on_generator(model, generator) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Dự đoán trên generator (reset -> classes -> predict -> argmax) như notebook."""

    generator.reset()
    y_true = generator.classes
    y_pred_prob = model.predict(generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    class_names = list(generator.class_indices.keys())
    return y_true, y_pred, class_names, y_pred_prob


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> np.ndarray:
    """Vẽ confusion matrix heatmap giống notebook."""

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 11},
    )

    plt.xlabel("Nhãn dự đoán", fontsize=13)
    plt.ylabel("Nhãn thực", fontsize=13)
    plt.title("Ma trận nhầm lẫn – Phân loại bệnh lá lúa", fontsize=15)

    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.show()

    return cm


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> str:
    """In classification report giống notebook (digits=4)."""

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("BÁO CÁO PHÂN LOẠI CHI TIẾT:\n")
    print(report)
    return report


def print_per_class_accuracy(cm: np.ndarray, class_names: List[str]) -> np.ndarray:
    """Tính và in accuracy theo từng lớp giống notebook."""

    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("ĐỘ CHÍNH XÁC THEO TỪNG LOẠI BỆNH\n")
    for i, acc in enumerate(per_class_accuracy):
        print(f"{class_names[i]:25s} : {acc*100:.2f}%")
    return per_class_accuracy


def show_correct_predictions(
    generator,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_prob: np.ndarray,
    class_names: List[str],
    num_images: int = 12,
    img_size: Tuple[int, int] = (224, 224),
) -> None:
    """Hiển thị ảnh dự đoán đúng như notebook (lấy từ generator.filepaths)."""

    correct_idx = np.where(y_true == y_pred)[0]

    print("THỐNG KÊ ẢNH DỰ ĐOÁN ĐÚNG\n")
    print(f"Tổng số ảnh validation : {len(y_true)}")
    print(f"Số ảnh dự đoán đúng    : {len(correct_idx)}")

    sample_idx = random.sample(list(correct_idx), min(num_images, len(correct_idx)))

    plt.figure(figsize=(18, 12))

    for i, idx in enumerate(sample_idx):
        img = load_img(generator.filepaths[idx], target_size=img_size)
        img_arr = img_to_array(img) / 255.0

        probs = y_pred_prob[idx]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class] * 100)
        true_class = int(y_true[idx])

        plt.subplot(3, 4, i + 1)
        plt.imshow(img_arr.astype(np.float32))
        plt.axis("off")
        plt.title(f"{class_names[true_class]}\nĐộ tin cậy: {confidence:.2f}%")

    plt.tight_layout()
    plt.show()
