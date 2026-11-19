"""
Ataskaitos paveikslai apie duomenis ir modelį (v4):
  fig/sintetiniai_pavyzdziai.png – 6 sintetiniai kadrai ir jų žymių žemėlapiai
  fig/klasiu_pasiskirstymas.png  – dėmių skaičius pagal klases
  fig/painiavos_matrica.png      – v4 painiavos matrica atskirame testo rinkinyje
  fig/pozymiu_svarba.png         – atsitiktinio miško požymių svarba (XAI)
  fig/metrikos_sintetinis.json   – tikslumas, precision, recall, F1

    py make_figures.py
"""
import json, os
import numpy as np
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_recall_fscore_support,
                             classification_report)

from synth import random_scene, render, CLASS_NAMES
from features import extract_blobs, blob_features, blob_labels, FEATURE_NAMES

LT = ["priekinis žib.", "galinis žib.", "gatvės žib.", "atšvaitas / ženklas", "triukšmas"]
os.makedirs("fig", exist_ok=True)

# 1. Pavyzdžiai
rng = np.random.default_rng(5)
pal = np.array([(255, 255, 255), (0, 0, 255), (0, 200, 255), (255, 160, 0), (25, 25, 25)], np.uint8)
tiles = []
for _ in range(6):
    img, lab = render(random_scene(rng), rng)
    tiles.append(np.vstack([img, pal[lab]]))
grid = np.hstack([np.hstack([t, np.full((t.shape[0], 6, 3), 255, np.uint8)]) for t in tiles])
cv2.imwrite("fig/sintetiniai_pavyzdziai.png", cv2.resize(grid, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA))

# 2. Atskiras testo rinkinys (kitas pradmuo nei mokymo – modelis šių kadrų nematė)
rng = np.random.default_rng(2026)
X, y = [], []
for _ in range(800):
    img, lab = render(random_scene(rng), rng)
    b, hsv = extract_blobs(img)
    if b:
        X.append(blob_features(img, b, hsv)); y.append(blob_labels(b, lab))
X, y = np.vstack(X), np.concatenate(y)
model = joblib.load("blob_classifier.joblib")
p = model.predict(X[:, :model.n_features_in_])

counts = np.bincount(y, minlength=5)
fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.barh(LT[::-1], counts[::-1], color="#1f6fb2")
for i, c in enumerate(counts[::-1]):
    ax.text(c, i, f" {c}", va="center", fontsize=8)
ax.set_xlabel("Dėmių skaičius (800 testo kadrų)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("fig/klasiu_pasiskirstymas.png", dpi=160); plt.close(fig)

cm = confusion_matrix(y, p, labels=range(5))
cmn = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm[i, j]}\n{100 * cmn[i, j]:.0f} %", ha="center", va="center", fontsize=7.5,
                color="white" if cmn[i, j] > 0.5 else "black")
ax.set_xticks(range(5)); ax.set_xticklabels(LT, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(5)); ax.set_yticklabels(LT, fontsize=8)
ax.set_xlabel("Prognozė"); ax.set_ylabel("Tikroji klasė")
fig.tight_layout(); fig.savefig("fig/painiavos_matrica.png", dpi=160); plt.close(fig)

veh_t, veh_p = np.isin(y, (0, 1)), np.isin(p, (0, 1))
pr, rc, f1, _ = precision_recall_fscore_support(veh_t, veh_p, average="binary")
res = {"blobs": int(len(y)), "accuracy_5class": round(float(accuracy_score(y, p)), 4),
       "macro_f1_5class": round(float(precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]), 4),
       "vehicle_accuracy": round(float(np.mean(veh_t == veh_p)), 4),
       "vehicle_precision": round(float(pr), 4), "vehicle_recall": round(float(rc), 4), "vehicle_f1": round(float(f1), 4),
       "counts": {CLASS_NAMES[k]: int(c) for k, c in enumerate(counts)}}
json.dump(res, open("fig/metrikos_sintetinis.json", "w"), indent=2)
print(json.dumps(res, indent=2))
print(classification_report(y, p, target_names=CLASS_NAMES, digits=3, zero_division=0))

# 3. Požymių svarba
if hasattr(model, "feature_importances_"):
    imp = model.feature_importances_
    order = np.argsort(imp)
    names = {"area_log": "plotas (log)", "v_max": "didž. šviesumas", "v_mean": "vid. šviesumas",
             "sat_frac": "persotintų px dalis", "r_b_ratio": "R/B", "r_g_ratio": "R/G",
             "hue_red": "raudonumas", "circularity": "apvalumas", "aspect": "pločio/aukščio sant.",
             "fill": "užpildymas", "halo_ratio": "švytėjimas", "y_rel": "aukštis kadre",
             "has_pair": "turi porą", "pair_dy": "poros aukščio skirt.", "ring_sat": "krašto sodrumas",
             "ring_v": "krašto šviesumas", "vstack": "dėmės virš/po (vstack)"}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh([names.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in order], imp[order], color="#7d3c98")
    ax.set_xlabel("Svarba (vidutinis Gini sumažėjimas)")
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("fig/pozymiu_svarba.png", dpi=160); plt.close(fig)
    print("top:", [(FEATURE_NAMES[i], round(float(imp[i]), 3)) for i in order[::-1][:6]])
