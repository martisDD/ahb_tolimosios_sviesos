"""
YOLO11 mokymas tikrais naktiniais vaizdais (antrasis, „pilnas“ DI kelias).

Duomenys: pvz., PVDN (Provident Vehicle Detection at Night) arba L-VAD, konvertuoti
į YOLO formatą su klasėmis 0 = headlight, 1 = taillight.
Katalogų struktūra:
    data/images/{train,val}/*.jpg
    data/labels/{train,val}/*.txt   # <klasė> <xc> <yc> <w> <h> (normalizuota)

Paleidimas (rekomenduojama GPU, pvz., Google Colab):
    pip install ultralytics
    python train_yolo.py
"""
from pathlib import Path
from ultralytics import YOLO

Path("night_lights.yaml").write_text(
    "path: data\ntrain: images/train\nval: images/val\nnames:\n  0: headlight\n  1: taillight\n"
)

model = YOLO("yolo11n.pt")          # COCO iš anksto išmokytas modelis -> perkeliamasis mokymas
model.train(data="night_lights.yaml", epochs=100, imgsz=960,   # didesnė raiška – maži tolimi žibintai
            batch=16, hsv_v=0.2, fliplr=0.0)                  # fliplr=0: eismo pusė svarbi
metrics = model.val()
print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)
model.export(format="onnx")          # įdiegimui automobilio kompiuteryje (pvz., Jetson, TensorRT)
