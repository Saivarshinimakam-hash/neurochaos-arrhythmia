from pathlib import Path
import sys
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.data.splitting import build_datasets
from src.models.bilstm import prepare_bilstm_inputs
from src.training.keras_utils import keras_predict
from src.evaluation.metrics import compute_metrics


cfg = load_config()

train_ds, val_ds, test_ds = build_datasets(
    cfg,
    download=True,
    normalize=True,
    save_manifest=False,
)

X_val = prepare_bilstm_inputs(val_ds.waveforms)
X_test = prepare_bilstm_inputs(test_ds.waveforms)

models = {
    "Original": (
        Path("results/experiments/20261003_013425_bilstm/bilstm_best.keras")
    ),
    "Resumed": (
        Path("results/experiments/20261003_090143_bilstm_resume/bilstm_resume_best.keras")
    ),
}

for name, model_path in models.items():

    print("\n" + "=" * 70)
    print(f"{name} MODEL")
    print("=" * 70)
    print(f"Model: {model_path}")

    model = tf.keras.models.load_model(model_path)

    for split_name, X, y in [
        ("Validation", X_val, val_ds.labels),
        ("Test", X_test, test_ds.labels),
    ]:

        print(f"\n--- {split_name} ---")

        y_pred, y_prob = keras_predict(model, X)

        metrics = compute_metrics(
            y,
            y_pred,
            y_prob,
            split=split_name.lower(),
        )

        print(
            f"Accuracy          : {metrics.get('accuracy')}"
        )
        print(
            f"Balanced Accuracy  : {metrics.get('balanced_accuracy')}"
        )
        print(
            f"Macro F1           : {metrics.get('macro_f1')}"
        )
        print(
            f"Weighted F1        : {metrics.get('weighted_f1')}"
        )

        print("\nClassification report:")
        print(metrics["classification_report"])