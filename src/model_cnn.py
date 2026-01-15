import os
import numpy as np
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import preprocessing functions
from preprocessing import (
    LABEL_MAP,
    default_data_paths,
    load_train_df,
    add_image_path_column,
    split_train_val,
    build_image_generators,
    compute_class_weights
)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Get data paths
paths = default_data_paths()
TRAIN_CSV = paths.train_csv
TRAIN_IMG_DIR = paths.train_img_dir


def build_monster_cnn(input_shape=(224, 224, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    
    # --- BLOCK 1 ---
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # --- BLOCK 2 ---
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # --- BLOCK 3 ---
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # --- BLOCK 4 ---
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # --- HEAD ---
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(
        4096, activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    
    x = layers.Dense(
        1024, activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Monster_Paddy_CNN")
    return model


def prepare_data_generators():
    """
    Prepare train and validation data generators with augmentation
    
    Returns:
        train_generator, val_generator, class_weights_dict
    """
    # Load and prepare data using preprocessing functions
    df = load_train_df(TRAIN_CSV, LABEL_MAP)
    df = add_image_path_column(df, TRAIN_IMG_DIR)
    
    # Split train/validation
    train_df, val_df = split_train_val(df)
    
    print(f"Tổng số ảnh: {len(df)}")
    print(f"Train set: {len(train_df)} ảnh")
    print(f"Validation set: {len(val_df)} ảnh")
    
    # Build image generators with advanced preprocessing
    make_generators = build_image_generators(IMG_SIZE, BATCH_SIZE)
    train_generator, val_generator = make_generators(train_df, val_df)
    
    # Calculate class weights
    class_weights_dict = compute_class_weights(train_df, train_generator.class_indices)
    
    print(f"Class indices: {train_generator.class_indices}")
    print(f"Class weights: {class_weights_dict}")
    
    return train_generator, val_generator, class_weights_dict


def get_callbacks():
    callbacks = [
        ModelCheckpoint(
            filepath="monster_cnn_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    return callbacks


def train_model(epochs=100):

    # Prepare data
    train_generator, val_generator, class_weights_dict = prepare_data_generators()
    
    # Build and compile model
    model = build_monster_cnn()
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Get callbacks
    callbacks = get_callbacks()
    
    # Train model
    print("Bắt đầu huấn luyện Monster CNN...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )
    
    print("Huấn luyện hoàn tất")
    return model, history


if __name__ == "__main__":
    # Train the model
    model, history = train_model(epochs=100)
