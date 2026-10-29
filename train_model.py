
from pathlib import Path

import joblib
import numpy as np
import wfdb

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from ecg_utils import filter_ecg, prepare_beat


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mit-bih-arrhythmia-database-1.0.0"
MODEL_PATH = ROOT / "ecg_model.joblib"

BEAT_LENGTH = 180
MAX_TRAIN_PER_CLASS = 8000
RANDOM_STATE = 42

LABEL_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F"
}


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}\n"
            "Extract the ZIP into the project's data folder."
        )

    headers = sorted(DATA_DIR.glob("*.hea"))
    records = [
        header.stem
        for header in headers
        if header.stem.isdigit()
        and (DATA_DIR / f"{header.stem}.atr").exists()
        and (DATA_DIR / f"{header.stem}.dat").exists()
    ]

    if not records:
        raise FileNotFoundError(
            "No complete numbered records with .hea, .dat "
            "and .atr files were found."
        )

    print("Dataset folder:", DATA_DIR)
    print("Available annotated records:", len(records))

    X_list = []
    y_list = []
    groups = []

    for index, record_id in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] Reading record {record_id}")

        base = DATA_DIR / record_id
        record = wfdb.rdrecord(str(base))
        annotation = wfdb.rdann(str(base), "atr")

        # Use the first ECG channel consistently
        signal = record.p_signal[:, 0].astype(float)
        signal = filter_ecg(signal, record.fs)

        for sample, symbol in zip(
            annotation.sample, annotation.symbol
        ):
            if symbol not in LABEL_MAP:
                continue

            beat = prepare_beat(
                signal,
                int(sample),
                record.fs,
                BEAT_LENGTH
            )

            if beat is None:
                continue

            X_list.append(beat)
            y_list.append(LABEL_MAP[symbol])
            groups.append(record_id)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list)
    groups = np.asarray(groups)

    if len(X) == 0 or len(np.unique(y)) < 2:
        raise ValueError(
            "Not enough valid annotated beats to train a model."
        )

    print("\nTotal usable beats:", len(X))
    print("Class counts:", {
        label: int(np.sum(y == label))
        for label in sorted(np.unique(y))
    })

    # Split by recording to reduce train/test leakage.
    # Try multiple group splits to retain the available classes.
    splitter = GroupShuffleSplit(
        n_splits=40,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    required_classes = set(LABEL_MAP.values())
    chosen_split = None
    best_score = -1

    for train_idx, test_idx in splitter.split(X, y, groups):
        train_classes = set(y[train_idx])
        test_classes = set(y[test_idx])

        if not required_classes.issubset(train_classes):
            continue

        score = len(test_classes.intersection(required_classes))

        if score > best_score:
            chosen_split = (train_idx, test_idx)
            best_score = score

        if required_classes.issubset(test_classes):
            break

    if chosen_split is None:
        raise ValueError(
            "Could not find a record-level split containing "
            "all four classes in training. Check dataset files."
        )

    train_idx, test_idx = chosen_split

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("\nTraining records:", len(np.unique(groups[train_idx])))
    print("Testing records:", len(np.unique(groups[test_idx])))

    # Cap very frequent classes to keep training and model size manageable.
    rng = np.random.default_rng(RANDOM_STATE)
    selected = []

    for label in sorted(np.unique(y_train)):
        indices = np.flatnonzero(y_train == label)

        if len(indices) > MAX_TRAIN_PER_CLASS:
            indices = rng.choice(
                indices,
                size=MAX_TRAIN_PER_CLASS,
                replace=False
            )

        selected.extend(indices.tolist())

    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)

    X_fit = X_train[selected]
    y_fit = y_train[selected]

    print("Training samples after balancing:", len(X_fit))

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_fit, y_fit)

    predictions = model.predict(X_test)

    print("\nTEST ACCURACY:", round(
        accuracy_score(y_test, predictions), 4
    ))

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(
        y_test,
        predictions,
        labels=sorted(required_classes),
        zero_division=0
    ))

    print("CONFUSION MATRIX (rows=true, columns=predicted):")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=sorted(required_classes)
    ))

    joblib.dump(
        {
            "model": model,
            "classes": list(model.classes_),
            "beat_length": BEAT_LENGTH,
            "training_records": sorted(
                np.unique(groups[train_idx]).tolist()
            ),
            "testing_records": sorted(
                np.unique(groups[test_idx]).tolist()
            )
        },
        MODEL_PATH,
        compress=3
    )

    print("\nModel saved to:", MODEL_PATH)
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
