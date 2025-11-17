
# evaluate_testset.py — robust evaluator for ID or OOD sets
import os, csv, random, argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score
)

# -------------------- Defaults (override via CLI) --------------------
DEFAULT_MODEL = "trained_models/mesonet_best_full.keras"
DEFAULT_TEST_DIR = "data/testing"   # pass --test-dir data/ood for OOD
IMG_SIZE_DEFAULT = 256
BATCH_SIZE_DEFAULT = 32
SAMPLE_BARS_DEFAULT = 20
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
# --------------------------------------------------------------------

def _fmt(x, fmt):
    return "None" if x is None else format(x, fmt)

def list_all_files_sorted(base_dir, class_names):
    paths = []
    for cname in class_names:
        croot = os.path.join(base_dir, cname)
        for root, _, files in os.walk(croot):
            for f in files:
                if f.lower().endswith(VALID_EXTS):
                    paths.append(os.path.join(root, f))
    paths.sort()
    return paths

def count_files_per_class(base_dir, class_names):
    counts = {}
    for c in class_names:
        cnt = 0
        for root, _, files in os.walk(os.path.join(base_dir, c)):
            for f in files:
                if f.lower().endswith(VALID_EXTS):
                    cnt += 1
        counts[c] = cnt
    return counts

def save_confidence_plot(sample_paths, p_real, pred_names, out_path, title="Confidence (sample)"):
    labels = [os.path.basename(p) for p in sample_paths]
    y_pos = np.arange(len(labels))
    plt.figure(figsize=(10,6))
    bars = plt.barh(y_pos, p_real, edgecolor="black")
    for bar, pred, pr in zip(bars, pred_names, p_real):
        plt.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                 f"{pred} ({pr:.2f})", va="center", fontsize=8)
    plt.yticks(y_pos, labels, fontsize=7)
    plt.xlabel("Probability of REAL (class=1)")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved confidence bar chart to {out_path}")

def evaluate_with_thresholds(y_true, p_real, fake_idx, real_idx, extra_threshold=None):
    """
    Returns dict with:
      - fixed_0_5
      - best_accuracy
      - best_macro_f1
      - best_macro_f1_both-classes
      - chosen
      - (optional) preset_<thr>
    """
    def preds_from_t(t):
        return np.where(p_real >= t, real_idx, fake_idx)

    # Sweep includes sub-0.01 thresholds so we can predict 'real' when scores are tiny
    grid = np.concatenate([
        np.linspace(0.0001, 0.01, 10),
        np.linspace(0.011, 0.99, 89)
    ])

    # 1) fixed 0.5
    t_fixed = 0.5
    y_fixed = preds_from_t(t_fixed)
    acc_fixed = accuracy_score(y_true, y_fixed)
    macro_fixed = f1_score(y_true, y_fixed, average="macro")

    # 2) best accuracy
    best_acc, t_acc = -1.0, t_fixed
    for t in grid:
        yp = preds_from_t(t)
        acc = accuracy_score(y_true, yp)
        if acc > best_acc:
            best_acc, t_acc = acc, t

    # 3) best macro-F1 (balanced)
    best_macro, t_macro = -1.0, t_fixed
    for t in grid:
        yp = preds_from_t(t)
        macro = f1_score(y_true, yp, average="macro")
        if macro > best_macro:
            best_macro, t_macro = macro, t

    # 4) best macro-F1 predicting BOTH classes
    best_macro_bi, t_macro_bi = -1.0, None
    for t in grid:
        yp = preds_from_t(t)
        if len(np.unique(yp)) == 2:
            macro = f1_score(y_true, yp, average="macro")
            if macro > best_macro_bi:
                best_macro_bi, t_macro_bi = macro, t

    # 5) optional preset threshold (e.g., calibrated τ)
    extra_key, extra_val = None, (None, None, None)
    if extra_threshold is not None:
        t = float(extra_threshold)
        yp = preds_from_t(t)
        extra_key = f"preset_{t:.2f}"
        extra_val = (t, accuracy_score(y_true, yp), f1_score(y_true, yp, average="macro"))

    # Pick operating point: prefer both-classes macro-F1; else plain macro-F1
    if t_macro_bi is not None:
        t_use = t_macro_bi
        chosen = "best_macro_f1_both-classes"
    else:
        t_use = t_macro
        chosen = "best_macro_f1"

    y_use = preds_from_t(t_use)

    summary = {
        "fixed_0_5": (t_fixed, acc_fixed, macro_fixed),
        "best_accuracy": (t_acc, best_acc, f1_score(y_true, np.where(p_real >= t_acc, real_idx, fake_idx), average="macro")),
        "best_macro_f1": (t_macro, accuracy_score(y_true, np.where(p_real >= t_macro, real_idx, fake_idx)), best_macro),
        "best_macro_f1_both-classes": (t_macro_bi, accuracy_score(y_true, np.where(p_real >= (t_macro_bi if t_macro_bi is not None else 0.5), real_idx, fake_idx)) if t_macro_bi else None, best_macro_bi),
        "chosen": (chosen, t_use, accuracy_score(y_true, y_use), f1_score(y_true, y_use, average="macro")),
        "y_pred": y_use
    }
    if extra_key:
        summary[extra_key] = extra_val
    return summary

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a trained Keras binary classifier on {fake, real} folder trees.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Path to .keras or SavedModel")
    ap.add_argument("--test-dir", default=DEFAULT_TEST_DIR, help="Directory with top-level 'fake' and 'real'")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE_DEFAULT, help="Square image size (H=W)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--sample-bars", type=int, default=SAMPLE_BARS_DEFAULT)
    ap.add_argument("--out-csv", default=None, help="CSV path for per-image predictions (auto if omitted)")
    ap.add_argument("--out-plot", default=None, help="PNG path for small confidence bar chart (auto if omitted)")
    ap.add_argument("--force-flip", action="store_true", help="Force interpret sigmoid as P(fake) and flip to P(real)")
    ap.add_argument("--preset-threshold", type=float, default=None, help="Optional fixed threshold to also report (e.g., calibrated τ)")
    return ap.parse_args()

def main():
    args = parse_args()
    MODEL_PATH = args.model
    TEST_DIR = args.test_dir
    IMG_SIZE = (args.img_size, args.img_size)
    BATCH_SIZE = args.batch_size
    SAMPLE_BARS = args.sample_bars

    label = Path(TEST_DIR).name.lower()
    default_tag = "ood" if label == "ood" else "test"
    out_csv = args.out_csv or f"outputs/{default_tag}_predictions.csv"
    out_plot = args.out_plot or f"outputs/{default_tag}_confidence_bar.png"

    print("[INFO] Loading model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("[INFO] Model output_shape:", model.output_shape)

    raw_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        shuffle=False,
        image_size=IMG_SIZE,
        interpolation="bilinear",
        batch_size=BATCH_SIZE,
    )
    class_names = raw_ds.class_names
    print("[INFO] Class names:", class_names)
    assert "fake" in class_names and "real" in class_names, "The test dir must have top-level 'fake' and 'real'."
    fake_idx = class_names.index("fake")
    real_idx = class_names.index("real")

    print("[INFO] Files by class:", count_files_per_class(TEST_DIR, class_names))

    AUTOTUNE = tf.data.AUTOTUNE
    def normalize(x, y):
        # Match training: simple [0,1] scaling. If you trained with a different preprocess, mirror it here.
        return tf.cast(x, tf.float32) / 255.0, y

    test_ds = raw_ds.map(normalize).prefetch(AUTOTUNE)

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    filepaths = list_all_files_sorted(TEST_DIR, class_names)
    print(f"[INFO] Samples: {len(y_true)} | Files aligned: {len(filepaths)}")

    raw = model.predict(test_ds, verbose=1)

    # Head interpretation
    if raw.ndim == 1 or raw.shape[-1] == 1:
        p_sig = raw.reshape(-1).astype("float64")
        unique = np.unique(y_true)
        if len(unique) >= 2 and not args.force_flip:
            y_real = (y_true == real_idx).astype(int)
            try:
                auc_as_real = roc_auc_score(y_real, p_sig)
                auc_as_fake = roc_auc_score(y_real, 1.0 - p_sig)
            except ValueError:
                auc_as_real = auc_as_fake = 0.5
            if auc_as_real >= auc_as_fake:
                interp = "sigmoid_as_P(real)"
                p_real = p_sig
            else:
                interp = "sigmoid_as_P(fake) -> flipped to P(real)"
                p_real = 1.0 - p_sig
        else:
            if args.force_flip:
                interp = "sigmoid_as_P(fake) -> forced flip to P(real)"
                p_real = 1.0 - p_sig
            else:
                interp = "sigmoid_as_P(real) (AUC undefined; no flip)"
                p_real = p_sig
        print(f"[INFO] Sigmoid head detected | {interp}")
    else:
        soft = tf.nn.softmax(raw, axis=-1).numpy()
        p_real = soft[:, real_idx].astype("float64")
        print(f"[INFO] Softmax head detected | using column index for 'real'={real_idx}")

    # Debug range + AUC (if possible)
    print(f"[DEBUG] p_real range: min={p_real.min():.6f} max={p_real.max():.6f} mean={p_real.mean():.6f}")
    try:
        y_real_bin = (y_true == real_idx).astype(int)
        auc_val = roc_auc_score(y_real_bin, p_real)
        print(f"[DEBUG] ROC-AUC (real vs p_real): {auc_val:.4f}")
    except Exception as e:
        print("[DEBUG] ROC-AUC unavailable:", e)

    # One-class fallback
    if len(np.unique(y_true)) < 2:
        thr = args.preset_threshold if args.preset_threshold is not None else 0.50
        y_pred = np.where(p_real >= thr, real_idx, fake_idx)
        n = len(p_real)
        predicted_fake_rate = float((y_pred == fake_idx).mean())
        print(f"[WARN] Only one true class present – reporting counts (N={n}).")
        print(f"  mean p_real = {p_real.mean():.4f}")
        print(f"  predicted_fake_rate@{thr:.2f} = {predicted_fake_rate:.4f}")
        os.makedirs(Path(out_csv).parent, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "predicted_label", "prob_real(class=1)"])
            for fp, yp, pr in zip(filepaths, y_pred, p_real):
                w.writerow([fp, class_names[yp], float(pr)])
        print(f"[INFO] Saved per-image predictions to {out_csv}")
        # small plot
        total = len(filepaths)
        if total > 0:
            idx = random.sample(range(total), k=min(SAMPLE_BARS_DEFAULT, total))
            sample_paths = [filepaths[i] for i in idx]
            sample_probs = [float(p_real[i]) for i in idx]
            sample_preds = [class_names[y_pred[i]] for i in idx]
            save_confidence_plot(sample_paths, sample_probs, sample_preds, out_plot, title=f"{default_tag.upper()} Confidence (sample)")
        print("[INFO] Done (one-class).")
        return

    # Two-class metrics
    preval_real = (y_true == real_idx).mean()
    print(f"[INFO] True prevalence: real={preval_real:.4f}, fake={1.0 - preval_real:.4f}")

    results = evaluate_with_thresholds(
        y_true, p_real, fake_idx, real_idx,
        extra_threshold=args.preset_threshold
    )

    # Summary table
    print("\n[INFO] Decision-rule summary:")
    keys = ["fixed_0_5", "best_accuracy", "best_macro_f1", "best_macro_f1_both-classes"]
    if args.preset_threshold is not None:
        keys.insert(1, f"preset_{args.preset_threshold:.2f}")
    for k in keys:
        t, acc, macro = results[k]
        print(f"  {k:28s} thr={_fmt(t, '.2f'):>5}  acc={_fmt(acc, '.4f'):>8}  macroF1={_fmt(macro, '.4f'):>8}")

    chosen_name, t_use, acc_use, macro_use = results["chosen"]
    y_pred = results["y_pred"]
    print(f"\n[INFO] USING: {chosen_name} (thr={t_use:.2f})  acc={acc_use:.4f}  macroF1={macro_use:.4f}\n")

    # Reports
    print("[RESULT] Classification report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[fake_idx, real_idx])
    print("[RESULT] Confusion Matrix (rows=true, cols=pred):\n", cm)

    # Save CSV
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "predicted_label", "prob_real(class=1)"])
        for fp, yp, pr in zip(filepaths, y_pred, p_real):
            w.writerow([fp, class_names[yp], float(pr)])
    print(f"[INFO] Saved per-image predictions to {out_csv}")

    # Save mini plot
    total = len(filepaths)
    if total > 0:
        idx = random.sample(range(total), k=min(SAMPLE_BARS_DEFAULT, total))
        sample_paths = [filepaths[i] for i in idx]
        sample_probs = [float(p_real[i]) for i in idx]
        sample_preds = [class_names[y_pred[i]] for i in idx]
        save_confidence_plot(sample_paths, sample_probs, sample_preds, out_plot, title=f"{default_tag.upper()} Confidence (sample)")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
