"""
Kadrų lygio metrikos tikruose įrašuose (Accuracy, Precision, Recall, F1).

Teigiama klasė – „reikia artimųjų“: kadre matomas automobilis (pagal *_cars.json).
Kadrai per 2,5 s po prasilenkimo nevertinami (leistinas laukimas prieš grąžinant tolimąsias).
Naudoja evaluate_real.py išsaugotas būsenas real/{įrašas}_{žymė}_states.npy.

    py frame_metrics.py --tags rule v4 ynight_f ycoco_f
"""
import argparse, json, os
import numpy as np
import cv2


def metrics(video, tag, grace=2.5):
    high = np.load(f"real/{video}_{tag}_states.npy")
    cap = cv2.VideoCapture(f"real/{video}.mp4"); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; cap.release()
    cars = json.load(open(f"real/{video}_cars.json", encoding="utf-8"))["cars"]
    t = np.arange(len(high)) / fps
    truth = np.zeros(len(high), bool); skip = np.zeros(len(high), bool)
    for c in cars:
        truth |= (t >= c["visible"]) & (t < c["passed"])
        skip |= (t >= c["passed"]) & (t < c["passed"] + grace)
    skip &= ~truth
    y, p = truth[~skip], ~high[~skip]          # p = sistema laiko artimąsias
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def summary(c):
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / (tp + fp + fn + tn), "precision": P, "recall": R,
            "f1": 2 * P * R / (P + R) if P + R else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["night", "vingis", "naktis"])
    ap.add_argument("--tags", nargs="+", default=["rule", "v4", "ynight_f", "ycoco_f"])
    a = ap.parse_args()
    out = {}
    for tag in a.tags:
        tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        out[tag] = {}
        for v in a.videos:
            if not os.path.exists(f"real/{v}_{tag}_states.npy"):
                continue
            c = metrics(v, tag)
            out[tag][v] = {**c, **summary(c)}
            for k in tot: tot[k] += c[k]
        out[tag]["viso"] = {**tot, **summary(tot)}
        s = out[tag]["viso"]
        print(f"{tag:10s} Acc={s['accuracy']:.3f} P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")
    json.dump(out, open("real/kadru_metrikos.json", "w"), indent=2)


if __name__ == "__main__":
    main()
