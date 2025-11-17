# mesonet/train.py
import os
import time
import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from mesonet.data import get_train_val_datasets
from mesonet.visualization import plot_loss_curve


def train_model(
    model,
    train_data_dir,
    validation_split=None,
    batch_size=16,
    use_default_augmentation=True,  # handled in build_model() when you created the model
    augmentations=None,             # (not used, kept for CLI compatibility)
    epochs=30,
    compile=True,
    lr=1e-3,
    loss="binary_crossentropy",
    lr_decay=True,
    decay_rate=0.10,
    decay_limit=1e-6,
    checkpoint=True,
    stop_early=True,
    monitor="val_accuracy",

    mode="max",
    patience=20,
    tensorboard=True,
    loss_curve=True,
):
    """
    Train the model with:
    - EarlyStopping
    - ModelCheckpoint (best model)
    - TensorBoard logging
    - ReduceLROnPlateau learning rate decay

    Returns:
      history (tf.keras.callbacks.History),
      best_model_path (str)
    """

    # Prepare datasets
    train_ds, val_ds = get_train_val_datasets(
        train_data_dir=train_data_dir,
        validation_split=validation_split,
        batch_size=batch_size,
        img_size=(256, 256),
    )

    # Compile model if requested
    if compile:
        opt = Adam(learning_rate=lr)
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

    # Prepare callbacks
    callbacks = []

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    best_model_path = os.path.join(save_dir, "mesonet_best_full.keras")

    if checkpoint:
        cp_cb = ModelCheckpoint(
            best_model_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,  # save full model for easy reload
            verbose=1,
        )
        callbacks.append(cp_cb)

    if stop_early and val_ds is not None:
        es_cb = EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(es_cb)

    if lr_decay:
        lr_cb = ReduceLROnPlateau(
            monitor="val_loss" if val_ds is not None else "loss",
            factor=decay_rate,
            patience=2,
            min_lr=decay_limit,
            verbose=1,
        )
        callbacks.append(lr_cb)

    if tensorboard:
        log_dir = os.path.join("runs", f"mesonet_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        tb_cb = TensorBoard(log_dir=log_dir)
        callbacks.append(tb_cb)
        print(f"[Train] TensorBoard log dir: {log_dir}")

    # Fit
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save curve PNG if asked
    if loss_curve:
        curve_path = os.path.join(save_dir, "training_curve.png")
        plot_loss_curve(history, save_path=curve_path, show_plot=True)

    print(f"[Train] Best model path: {best_model_path}")
    return history, best_model_path
