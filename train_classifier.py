"""
DI modelio mokymas: šviesos dėmių klasifikatorius.

Generuoja sintetinius kadrus, iš jų išskiria dėmes ir požymius, tada moko ir
palygina du modelius: atsitiktinį mišką (Random Forest) ir daugiasluoksnį
perceptroną (MLP). Geresnis pagal F1 (transporto priemonių klasėms) išsaugomas.

Paleidimas:  py train_classifier.py   (~3 min, 4000 kadrų)
"""
import argparse, json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from synth import random_scene, render, CLASS_NAMES
from features import extract_blobs, blob_features, blob_labels


def build_dataset(n_frames, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_frames):
        img, lab = render(random_scene(rng), rng)
        blobs, hsv = extract_blobs(img)
        if not blobs:
            continue
        X.append(blob_features(img, blobs, hsv))
        y.append(blob_labels(blobs, lab))
    return np.vstack(X), np.concatenate(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--out", default="blob_classifier.joblib")
    ap.add_argument("--seed", type=int, default=1, help="atsitiktinių duomenų ir modelio pradmuo")
    ap.add_argument("--rf-only", action="store_true", help="mokyti tik atsitiktinį mišką (greičiau)")
    args = ap.parse_args()

    X, y = build_dataset(args.frames, seed=args.seed)
    print(f"Dėmių: {len(y)}; pasiskirstymas pagal klases:",
          {CLASS_NAMES[k]: int(c) for k, c in enumerate(np.bincount(y, minlength=5))})
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=14,
                                               class_weight="balanced", n_jobs=-1, random_state=args.seed),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600,
                                           early_stopping=True, random_state=args.seed)),
    }
    if args.rf_only:
        models.pop("MLP")
    results, best = {}, None
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        veh_true, veh_pred = np.isin(yte, (0, 1)), np.isin(p, (0, 1))
        f1_veh = f1_score(veh_true, veh_pred)
        results[name] = {"macro_f1": float(f1_score(yte, p, average="macro")),
                         "vehicle_f1": float(f1_veh)}
        print(f"\n=== {name} ===")
        print(classification_report(yte, p, target_names=CLASS_NAMES, digits=3, zero_division=0))
        print("Painiavos matrica:\n", confusion_matrix(yte, p, labels=range(5)))
        print(f"Transporto priemonės / ne (dvejetainis) F1 = {f1_veh:.3f}")
        if best is None or f1_veh > results[best]["vehicle_f1"]:
            best = name

    print(f"\nIšsaugomas modelis: {best}")
    joblib.dump(models[best], args.out, compress=3)   # suspausta: < 25 MB GitHub naršyklei
    with open(args.out.replace(".joblib", "") + "_train.json" if args.out != "blob_classifier.joblib" else "train_results.json", "w") as f:
        json.dump({"n_blobs": int(len(y)), "best": best, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
