"""
Mokymas su tikrais neigiamais pavyzdžiais (v5).

Sintetiniai duomenys nepakankamai atkartoja tikrus kelio stulpelius ir linijas.
Tačiau tikruose įrašuose, kai automobilio nėra (pagal *_cars.json žymes),
VISOS šviesos dėmės garantuotai nėra automobiliai – jos naudojamos kaip tikri
neigiami pavyzdžiai (angl. hard negative mining). Rankinio žymėjimo nereikia.

Kad vertinimas būtų sąžiningas, naudojamas „vieno įrašo palikimas“:
modelis mokomas su neigiamais pavyzdžiais iš kitų įrašų, o tikrinamas įrašu,
kurio jis nematė.

    py train_hardneg.py --neg-from night vingis --out blob_classifier_v5.joblib
    py train_hardneg.py --neg-from night --out v5_be_vingis.joblib     # testui su vingis
"""
import argparse, json, os
import numpy as np
import cv2
import joblib
from sklearn.ensemble import RandomForestClassifier

from features import extract_blobs, blob_features
from main import resize_for_model
from train_classifier import build_dataset

CACHE = "synthetic_dataset.npz"


def synthetic(frames, seed=1):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if int(d["frames"]) == frames and int(d["seed"]) == seed:
            return d["X"], d["y"]
    X, y = build_dataset(frames, seed)
    np.savez_compressed(CACHE, X=X, y=y, frames=frames, seed=seed)
    return X, y


def real_negatives(name, step=3, margin_before=1.0, margin_after=2.5):
    """Dėmės iš kadrų, kuriuose (su atsarga) nėra jokio automobilio."""
    cars = json.load(open(f"real/{name}_cars.json", encoding="utf-8"))["cars"]
    cap = cv2.VideoCapture(f"real/{name}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    X, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = i / fps
        busy = any(c["visible"] - margin_before <= t < c["passed"] + margin_after for c in cars)
        if i % step == 0 and not busy:
            f = resize_for_model(f)
            blobs, hsv = extract_blobs(f)
            if blobs:
                X.append(blob_features(f, blobs, hsv))
        i += 1
    cap.release()
    return np.vstack(X) if X else np.zeros((0, 17), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-from", nargs="+", default=["night", "vingis"],
                    help="įrašai (aplanke real), iš kurių imami neigiami pavyzdžiai")
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--weight", type=float, default=3.0, help="tikrų pavyzdžių svoris")
    ap.add_argument("--out", default="blob_classifier_v5.joblib")
    args = ap.parse_args()

    Xs, ys = synthetic(args.frames)
    parts, info = [Xs], {}
    for name in args.neg_from:
        Xn = real_negatives(name)
        parts.append(Xn)
        info[name] = int(len(Xn))
    Xr = np.vstack(parts[1:]) if len(parts) > 1 else np.zeros((0, Xs.shape[1]))
    X = np.vstack([Xs, Xr])
    y = np.concatenate([ys, np.full(len(Xr), 3)])          # 3 = atšvaitas / ne automobilis
    w = np.concatenate([np.ones(len(ys)), np.full(len(Xr), args.weight)])
    print(f"Sintetinių dėmių: {len(ys)}, tikrų neigiamų: {info}")
    rf = RandomForestClassifier(n_estimators=200, max_depth=14, class_weight="balanced",
                                n_jobs=-1, random_state=1)
    rf.fit(X, y, sample_weight=w)
    joblib.dump(rf, args.out, compress=3)
    print("Išsaugota:", args.out)


if __name__ == "__main__":
    main()
