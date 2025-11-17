# mesonet/model.py
import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

############################################
# 1. Build the MesoNet model
############################################

def build_model(
    input_shape=(256, 256, 3),
    use_default_augmentation=True,
):

    inputs = tf.keras.Input(shape=input_shape, dtype="float32", name="input_image")

    x = inputs

    # Optional data augmentation
    if use_default_augmentation:
        aug_layers = tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),
                layers.RandomZoom(0.1),
            ],
            name="augmentation",
        )
        x = aug_layers(x)

    # Normalize to [0,1]
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(x)

    # Block 1
    x = layers.Conv2D(8, (3, 3), padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.Dropout(0.1, name="drop1")(x)

    # Block 2
    x = layers.Conv2D(8, (5, 5), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.Dropout(0.1, name="drop2")(x)

    # Block 3
    x = layers.Conv2D(16, (5, 5), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.Dropout(0.1, name="drop3")(x)

    # Block 4
    x = layers.Conv2D(16, (5, 5), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(4, 4), name="pool4")(x)
    x = layers.Dropout(0.1, name="drop4")(x)

    # Dense head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(16, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="drop_fc1")(x)

    # Very important when using mixed_float16:
    # final layer must be float32 so loss/metrics are stable
    output = layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="prediction",
    )(x)

    model = models.Model(inputs=inputs, outputs=output, name="MesoNet")

    return model


############################################
# 2. Helper for loading a trained model
############################################

def load_trained_model(model_path):
    """
    Load a saved model (full .keras) from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Couldn't find model at {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


############################################
# 3. Prediction / evaluation on a folder
############################################

def _load_image_paths(data_dir):
    """
    Walks data_dir expecting structure:
      data_dir/
        real/
        fake/
    Returns:
      - filepaths: [str, ...]
      - labels: [int, ...]  (mapped from subfolder)
      - class_names: [str, ...] (sorted so it's stable)
    """
    subdirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    # Sort for consistency, e.g. ['fake','real']
    class_names = sorted(subdirs)

    filepaths = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        imgs = glob.glob(os.path.join(class_dir, "*.jpg")) + \
               glob.glob(os.path.join(class_dir, "*.jpeg")) + \
               glob.glob(os.path.join(class_dir, "*.png"))

        for img_path in imgs:
            filepaths.append(img_path)
            labels.append(class_idx)

    return filepaths, np.array(labels, dtype="int32"), class_names


def _preprocess_for_inference(img_path, img_size=(256, 256)):
    """
    Load + resize + normalize single image for inference.
    Returns float32 array of shape (256,256,3).
    """
    img = tf.keras.utils.load_img(
        img_path,
        target_size=img_size,
        color_mode="rgb",
    )
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0  # match training normalization
    return arr


def make_prediction(
    model_path,
    data_dir,
    threshold=0.5,
    return_probs=False,
    return_report=False,
):
    """
    Run inference on data_dir and (optional) produce a classification report.
    Returns:
      preds_table: np.array([...]) rows = filename, pred_label, [prob]
      report_str: str or None
    """
    print(f"[Predict] Loading model from: {model_path}")
    model = load_trained_model(model_path)

    filepaths, y_true, class_names = _load_image_paths(data_dir)
    if len(filepaths) == 0:
        raise RuntimeError(f"No images found under {data_dir}")

    # Prepare batch
    X = np.stack(
        [_preprocess_for_inference(p) for p in filepaths],
        axis=0
    ).astype("float32")

    # Predict
    y_prob = model.predict(X, verbose=0).flatten()  # probability of class "1"
    # Our datasets assign int labels in alphabetical order of subfolders.
    # If class_names == ['fake', 'real'], then:
    #   label 0 -> 'fake'
    #   label 1 -> 'real'
    # model output is sigmoid -> "probability of class 1"
    y_pred = (y_prob >= threshold).astype("int32")

    # Assemble table for display
    rows = []
    for i, path in enumerate(filepaths):
        pred_label_name = class_names[y_pred[i]]
        if return_probs:
            # show probability of the predicted class 1 ("real" if that's class idx 1)
            rows.append([os.path.basename(path), pred_label_name, float(y_prob[i])])
        else:
            rows.append([os.path.basename(path), pred_label_name])

    preds_table = np.array(rows, dtype=object)

    # Optional classification report
    report = None
    if return_report:
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )

    return preds_table, report
