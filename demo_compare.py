"""
Demonstracinis vaizdo įrašas gynimui: du metodai greta tame pačiame įraše.

    py demo_compare.py --video real/vingis.mp4 --out palyginimas_vingis.mp4
    py demo_compare.py --video real/night.mp4 --out palyginimas_night.mp4

Kairėje – dėmių modelis (v4), dešinėje – naktinis YOLO su geometriniu filtru.
Abiem naudojama ta pati sprendimų logika (balsavimas, sekimas, histerezė).
Viršuje – metodo pavadinimas, apačioje – ar kadre iš tikrųjų yra automobilis
(jei yra žymių failas *_cars.json).
"""
import argparse, json, os
import numpy as np
import cv2

from detector import BlobDetector, OnnxYoloDetector
from controller import BeamController
from main import draw, resize_for_model


def label_bar(width, text, color=(255, 255, 255), bg=(30, 30, 30), h=34):
    bar = np.full((h, width, 3), bg, np.uint8)
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blob-model", default="blob_classifier.joblib")
    ap.add_argument("--yolo", default="best_night.onnx")
    ap.add_argument("--yolo-classes", default="all+filtras")
    ap.add_argument("--max-s", type=float, default=None, help="apdoroti tik pirmas N sekundžių")
    args = ap.parse_args()

    from evaluate_real import parse_classes
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    blob = BlobDetector(args.blob_model)
    yolo = OnnxYoloDetector(args.yolo, vehicle_classes=parse_classes(args.yolo_classes),
                            geometry_filter=args.yolo_classes.endswith("+filtras"))
    c_blob = BeamController(fps=fps)
    c_yolo = BeamController(fps=fps, min_conf=0.25, track_conf=0.15)
    lab = os.path.splitext(args.video)[0] + "_cars.json"
    cars = json.load(open(lab, encoding="utf-8"))["cars"] if os.path.exists(lab) else None

    writer, i = None, 0
    print("Apdorojama... (sustabdyti – Ctrl+C, įrašas bus išsaugotas iki tos vietos)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_s and i / fps >= args.max_s):
                break
            f = resize_for_model(frame)
            d1, d2 = blob.detect(f), yolo.detect(f)
            h1 = c_blob.update([d["prob"] for d in d1 if d["cls"] in (0, 1)], f, d1)
            h2 = c_yolo.update([d["prob"] for d in d2], f, d2)
            a = draw(f.copy(), d1, h1, fps, 0.6, show_fps=False)
            b = draw(f.copy(), d2, h2, fps, 0.25, show_fps=False)
            W = a.shape[1]
            top = np.hstack([label_bar(W, "Demiu modelis (v4)"), label_bar(W, "Naktinis YOLO11n")])
            mid = np.hstack([a, np.full((a.shape[0], 6, 3), 255, np.uint8), b])
            top = cv2.resize(top, (mid.shape[1], top.shape[0]))
            t = i / fps
            if cars is not None:
                car = any(c["visible"] <= t < c["passed"] for c in cars)
                txt = f"{t:5.1f} s   Tiesa: " + ("automobilis matomas -> reikia ARTIMUJU" if car
                                                  else "automobilio nera -> galima TOLIMOSIOS")
                bot = label_bar(mid.shape[1], txt, (0, 165, 255) if car else (0, 255, 0))
            else:
                bot = label_bar(mid.shape[1], f"{t:5.1f} s")
            out = np.vstack([top, mid, bot])
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (out.shape[1], out.shape[0]))
            writer.write(out)
            i += 1
            if i % 100 == 0:
                print(f"  {t:.0f} s")
    except KeyboardInterrupt:
        print("Sustabdyta.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    print("Išsaugota:", args.out)


if __name__ == "__main__":
    main()
