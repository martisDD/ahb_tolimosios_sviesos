"""
Sistemos paleidimas realiu laiku: kamera arba vaizdo įrašas.

    py main.py --source 0                          # internetinė kamera
    py main.py --source naktis.mp4                 # vaizdo įrašas -> naktis_rezultatas.mp4
    py main.py --source naktis.mp4 --save mano.mp4 # rezultatą išsaugoti kitu pavadinimu
    py main.py --source naktis.mp4 --no-show       # be lango (greičiau), tik išsaugoti
    py main.py --source naktis.mp4 --yolo yolo11n.onnx                 # iš anksto išmokytas YOLO (COCO)
    py main.py --source naktis.mp4 --yolo best.onnx --yolo-classes all # savas naktinis YOLO

Kiekvienas kadras sumažinamas iki 640 px pločio (kaip mokymo duomenys).
Ekrane rodomi aptikti žibintai ir šviesų būsena; Esc – sustabdyti.

Režimai (žmogaus kontrolė, metodinių nurodymų reikalavimas):
    --rezimas patarimas    (numatytasis) sistema tik SIŪLO vairuotojui perjungti
                           šviesas; jokia valdymo komanda nesiunčiama
    --rezimas automatinis  imituojama CAN komanda šviesų valdymo blokui; tikrame
                           automobilyje šį režimą vairuotojas įjungtų pats ir
                           rankinis jungiklis visada turėtų pirmenybę
Prototipas realių žibintų nevaldo nė viename režime.
"""
import argparse, os, time
import cv2

from detector import BlobDetector, YoloDetector, OnnxYoloDetector
from controller import BeamController

COLORS = {0: (255, 255, 255), 1: (0, 0, 255), 2: (0, 200, 255), 3: (255, 160, 0), 4: (90, 90, 90)}
WORK_W = 640   # modelis mokytas 640 x 360 kadrais


def draw(frame, dets, high, fps, thr=0.6, show_fps=True, prefix=""):
    for d in dets:
        x, y, w, h = d["bbox"]
        if d["prob"] >= thr:
            big = w * h > 400          # YOLO rėmeliai – storesni, su tikimybe
            cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), COLORS[d["cls"]], 2 if big else 1)
            if big:
                cv2.putText(frame, f"{d['prob']:.2f}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, COLORS[d["cls"]], 1)
    label = prefix + ("TOLIMOSIOS" if high else "ARTIMOSIOS")
    cv2.putText(frame, f"{label}  {fps:.0f} FPS" if show_fps else label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if high else (0, 165, 255), 2)
    return frame


def resize_for_model(frame):
    h, w = frame.shape[:2]
    if w == WORK_W:
        return frame
    new_h = int(round(h * WORK_W / w))
    return cv2.resize(frame, (WORK_W, new_h), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="kameros numeris arba vaizdo failas")
    ap.add_argument("--yolo", default=None, help="YOLO modelis: .onnx (OpenCV) arba .pt (ultralytics)")
    ap.add_argument("--yolo-classes", default="coco", metavar="coco|all|1,6[+filtras]",
                    help="coco – iš anksto išmokytas COCO modelis; all – savas naktinis modelis")
    ap.add_argument("--save", default=None, help="kur išsaugoti rezultatą (.mp4)")
    ap.add_argument("--no-show", action="store_true", help="nerodyti lango")
    ap.add_argument("--rezimas", choices=["patarimas", "automatinis"], default="patarimas",
                    help="patarimas – tik siūlymas vairuotojui; automatinis – imituojama CAN komanda")
    args = ap.parse_args()

    is_cam = args.source.isdigit()
    src = int(args.source) if is_cam else args.source
    if not is_cam and not os.path.exists(src):
        raise SystemExit(f"Failas nerastas: {src}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Nepavyko atidaryti: {args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps > 120:
        src_fps = 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0
    save = args.save or (None if is_cam else os.path.splitext(src)[0] + "_rezultatas.mp4")

    if args.yolo and args.yolo.endswith(".onnx"):
        from evaluate_real import parse_classes
        det = OnnxYoloDetector(args.yolo, vehicle_classes=parse_classes(args.yolo_classes),
                               geometry_filter=args.yolo_classes.endswith("+filtras"))
        ctl = BeamController(fps=src_fps, min_conf=0.25, track_conf=0.15)
    elif args.yolo:
        det = YoloDetector(args.yolo)
        ctl = BeamController(fps=src_fps, min_conf=0.25, track_conf=0.15)
    else:
        det = BlobDetector()
        ctl = BeamController(fps=src_fps)
    writer = None
    n = n_high = 0
    t_prev = t_start = time.time()
    print(f"Apdorojama: {args.source} ({src_fps:.0f} k/s), režimas: {args.rezimas}. "
          "Sustabdyti: Esc lange arba Ctrl+C.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_for_model(frame)
            dets = det.detect(frame)
            veh = [d["prob"] for d in dets if d["cls"] in (0, 1)]
            was_high = ctl.high
            high = ctl.update(veh, frame, dets)
            if high != was_high:
                if args.rezimas == "automatinis":
                    print(f"{n / src_fps:7.1f} s  CAN ->", "HIGH_BEAM_ON" if high else "LOW_BEAM_ON")
                else:
                    print(f"{n / src_fps:7.1f} s  Pasiūlymas vairuotojui:",
                          "galima įjungti tolimąsias" if high else "įjunkite artimąsias")
            n += 1
            n_high += high
            now = time.time()
            out = draw(frame, dets, high, 1 / max(now - t_prev, 1e-6), ctl.min_conf,
                       prefix="SIULOMA: " if args.rezimas == "patarimas" else "")
            t_prev = now
            if save:
                if writer is None:
                    h, w = out.shape[:2]
                    writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                writer.write(out)
            if not args.no_show:
                cv2.imshow("AHB", out)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            elif total and n % 100 == 0:
                print(f"  {n}/{total} kadrų ({100 * n / total:.0f} %)")
    except KeyboardInterrupt:
        print("Sustabdyta.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if n:
        print(f"\nApdorota kadrų: {n} ({n / src_fps:.1f} s įrašo) per {time.time() - t_start:.1f} s")
        print(f"Tolimosios degė {100 * n_high / n:.0f} % laiko, artimosios {100 * (n - n_high) / n:.0f} %")
        print(f"Perjungimų: {ctl.switches}")
        if save:
            print(f"Rezultatas išsaugotas: {save}")


if __name__ == "__main__":
    main()
