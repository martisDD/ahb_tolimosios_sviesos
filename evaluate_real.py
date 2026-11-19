"""
Vertinimas tikru vaizdo įrašu su rankiniu būdu pažymėtais automobiliais.

    py evaluate_real.py --video real/night.mp4 --labels real/night_cars.json
    py evaluate_real.py --video real/night.mp4 --labels real/night_cars.json --model blob_classifier_v1.joblib --tag v1 --window 0.067 --vote 1.0 --no-track

Žymių faile kiekvienam priešpriešiniam automobiliui nurodyta, nuo kada jis matomas
(visible) ir kada prasilenkia (passed). Tuo metu turi degti artimosios.
Matuojama:
  * aptikimo vėlavimas – kiek sekundžių po pasirodymo įjungtos artimosios;
  * akinimo laikas – kiek sekundžių degė tolimosios, nors automobilis matomas;
  * bereikalingos artimosios – artimosios, kai automobilio nėra
    (neskaičiuojamos leistinos 2 s po prasilenkimo);
  * perjungimų skaičius.
"""
import argparse, json, os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detector import BlobDetector, OnnxYoloDetector, RuleDetector
from controller import BeamController
from main import draw, resize_for_model


def parse_classes(spec):
    """'coco' – COCO transporto klasės; 'all' – visos; '1,6' – nurodytos klasės.
    Pabaigoje '+filtras' – atmesti rėmelius kadro apačioje ir per didelius."""
    spec = spec.replace("+filtras", "")
    if spec == "all":
        return None
    if spec == "coco":
        return OnnxYoloDetector.COCO_VEHICLES
    return tuple(int(x) for x in spec.split(","))


def run(video, model, save_video=None, window_s=0.3, vote_ratio=0.6, track=True,
        min_conf=0.6, track_conf=0.35, yolo_classes="coco"):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Nepavyko atidaryti: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if model.endswith(".onnx"):
        det = OnnxYoloDetector(model, vehicle_classes=parse_classes(yolo_classes),
                               geometry_filter=yolo_classes.endswith("+filtras"))
    elif model == "rule":          # bazinis metodas be DI
        det = RuleDetector()
    else:
        det = BlobDetector(model)
    ctl = BeamController(fps=fps, window_s=window_s, vote_ratio=vote_ratio, track=track,
                         min_conf=min_conf, track_conf=track_conf)
    states, writer = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_for_model(frame)
        dets = det.detect(frame)
        high = ctl.update([d["prob"] for d in dets if d["cls"] in (0, 1)], frame, dets)
        states.append(high)
        if save_video:
            out = draw(frame.copy(), dets, high, fps)
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(out)
    cap.release()
    if writer is not None:
        writer.release()
    return np.array(states, bool), fps, ctl


def score(high, fps, cars, hold_s=2.0):
    t = np.arange(len(high)) / fps
    allowed_low = np.zeros(len(high), bool)
    per_car = []
    for c in cars:
        vis = (t >= c["visible"]) & (t < c["passed"])
        allowed_low |= (t >= c["visible"]) & (t < c["passed"] + hold_s + 0.5)
        low_idx = np.where(vis & ~high)[0]
        delay = round(float(t[low_idx[0]] - c["visible"]), 2) if len(low_idx) else None
        per_car.append({"car": c["name"],
                        "visible_s": c["visible"], "passed_s": c["passed"],
                        "detected_after_s": delay,
                        "glare_s": round(float((vis & high).sum() / fps), 2)})
    needless = (~high) & ~allowed_low
    sw = int(np.sum(high[1:] != high[:-1]))
    return {"duration_s": round(len(high) / fps, 1),
            "cars": per_car,
            "glare_total_s": round(sum(c["glare_s"] for c in per_car), 2),
            "needless_low_s": round(float(needless.sum() / fps), 2),
            "switches": sw,
            "high_frac": round(float(high.mean()), 3)}


def plot(results, cars, fps, out):
    fig, axes = plt.subplots(len(results), 1, figsize=(10, 1.3 + 1.1 * len(results)),
                             sharex=True, squeeze=False)
    for ax, (name, high) in zip(axes[:, 0], results.items()):
        t = np.arange(len(high)) / fps
        for c in cars:
            ax.axvspan(c["visible"], c["passed"], color="#e67e22", alpha=0.18, lw=0)
            ax.text((c["visible"] + c["passed"]) / 2, 1.12, f"automobilis {c['name']}",
                    ha="center", fontsize=8, color="#a04000")
        ax.step(t, high.astype(int), where="post", color="#1e8449", lw=1.6)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Artimosios", "Tolimosios"])
        ax.set_ylim(-0.15, 1.3)
        ax.set_title(name, loc="left", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Laikas, s")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model", default="blob_classifier.joblib")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--window", type=float, default=0.3, help="balsavimo langas, s")
    ap.add_argument("--vote", type=float, default=0.6, help="reikalinga lango dalis")
    ap.add_argument("--no-track", action="store_true", help="išjungti objektų sekimą (v2 logika)")
    ap.add_argument("--min-conf", type=float, default=None,
                    help="tikimybės riba (numatyta: 0,6 dėmių modeliui, 0,25 YOLO)")
    ap.add_argument("--yolo-classes", default="coco", metavar="coco|all|1,6[+filtras]",
                    help="coco – iš anksto išmokytas COCO modelis; all – savas naktinis modelis")
    ap.add_argument("--track-conf", type=float, default=None,
                    help="sekimo riba (numatyta: 0,35 dėmių modeliui, 0,15 YOLO)")
    args = ap.parse_args()
    cars = json.load(open(args.labels, encoding="utf-8"))["cars"]
    base = os.path.splitext(args.video)[0]
    vid_out = f"{base}_{args.tag}.mp4" if args.save_video else None
    yolo = args.model.endswith(".onnx")
    mc = args.min_conf if args.min_conf is not None else (0.25 if yolo else 0.6)
    tc = args.track_conf if args.track_conf is not None else (0.15 if yolo else 0.35)
    high, fps, _ = run(args.video, args.model, vid_out, args.window, args.vote,
                       not args.no_track, mc, tc, args.yolo_classes)
    res = score(high, fps, cars)
    res["model"] = args.model
    res["window_s"], res["vote_ratio"], res["tracking"] = args.window, args.vote, not args.no_track
    res["min_conf"], res["track_conf"] = mc, tc
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(f"{base}_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.save(f"{base}_{args.tag}_states.npy", high)
    plot({args.tag: high}, cars, fps, f"{base}_{args.tag}.png")


if __name__ == "__main__":
    main()
