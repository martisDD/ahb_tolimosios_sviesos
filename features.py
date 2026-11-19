"""
Šviesos dėmių (blob) išskyrimas ir požymių skaičiavimas.

v2: pašalintas požymis x_rel (horizontali padėtis kadre). v1 modelis buvo
išmokęs, kad ženklai visada dešinėje – posūkyje tai nebegalioja.
v2: pridėti ring_sat ir ring_v – spalvos sodrumas ir šviesumas 3 px žiede aplink
dėmę. Ryškaus geltono atšvaito centras kameroje „išbąla“, bet kraštas lieka
sodriai geltonas, o automobilio žibinto kraštas beveik bespalvis.
v4: pridėtas vstack – kiek kitų dėmių yra tiesiai virš ar po šia dėme. Baltą
kelio stulpelį juoda juosta suskaido į kelias dėmes viena virš kitos, o
automobilio žibintai išsidėsto horizontaliai.

1 žingsnis: ryškių sričių išskyrimas (adaptyvus slenkstis pagal V kanalą).
2 žingsnis: kiekvienai sričiai apskaičiuojamas požymių vektorius, kurį
            klasifikuoja mašininio mokymosi modelis (žr. train_classifier.py).
"""
import numpy as np
import cv2

FEATURE_NAMES = [
    "area_log", "v_max", "v_mean", "sat_frac",      # dydis ir šviesumas
    "r_b_ratio", "r_g_ratio", "hue_red",            # spalva
    "circularity", "aspect", "fill",                # forma
    "halo_ratio", "y_rel",                          # švytėjimas ir aukštis kadre
    "has_pair", "pair_dy",                           # porinis žibintas
    "ring_sat", "ring_v",                            # krašto spalvos sodrumas ir šviesumas
    "vstack",                                        # kiek dėmių tiesiai virš / po šia (stulpeliai)
]


def _components(mask, min_area, blobs):
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            blobs.append({"bbox": (x, y, w, h), "area": int(a),
                          "c": tuple(cents[i]), "mask": lab == i})


def extract_blobs(frame, core_thr=220, thr=150, min_area=2):
    """
    Trys kandidatų kaukės (kad tolimas žibintas „nesusilietų“ su atšvaitais
    prie horizonto):
      1) persotintos šerdys (V >= core_thr) – priekiniai ir gatvės žibintai;
      2) vidutinio šviesumo sritys (V >= thr), nepersidengiančios su šerdimis;
      3) raudonos sritys – galiniai žibintai būna blankesni už priekinius.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    b, g, r = [frame[..., i].astype(np.int16) for i in range(3)]
    core = v >= core_thr
    taken = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    mid = (v >= thr) & ~taken
    red = (r >= 90) & (r > 1.8 * g) & (r > 1.8 * b) & ~taken & ~mid
    blobs = []
    for m in (core, mid, red):
        _components(m, min_area, blobs)
    return blobs, hsv


def _pair_info(blobs, k):
    """Ar dėmė turi „porą“ – panašaus dydžio dėmę toje pačioje aukštyje."""
    cx, cy = blobs[k]["c"]
    ak = blobs[k]["area"]
    best = 1e9
    for j, b in enumerate(blobs):
        if j == k:
            continue
        dx = abs(b["c"][0] - cx)
        dy = abs(b["c"][1] - cy)
        ratio = max(ak, b["area"]) / max(1, min(ak, b["area"]))
        size = np.sqrt(ak)
        if dy < 3 + size and 1.5 * size < dx < 25 * size + 10 and ratio < 3:
            best = min(best, dy)
    return (1.0, best) if best < 1e9 else (0.0, 20.0)


def _vstack(blobs, k):
    x, y, w, h = blobs[k]["bbox"]
    cx = blobs[k]["c"][0]
    n = 0
    for j, b in enumerate(blobs):
        if j == k:
            continue
        bx, by, bw, bh = b["bbox"]
        if abs(b["c"][0] - cx) > max(w, bw, 4) * 0.8 + 1:
            continue
        gap = max(by - (y + h), y - (by + bh))       # tarpas vertikaliai (neigiamas – persidengia)
        if gap <= 2 * max(h, bh) + 4 and bw <= 3 * max(w, 3):
            n += 1
    return min(n, 3)


def blob_features(frame, blobs, hsv):
    H, W = frame.shape[:2]
    feats = []
    gray = hsv[..., 2].astype(np.float32)
    for k, b in enumerate(blobs):
        x, y, w, h = b["bbox"]
        m = b["mask"][y:y + h, x:x + w]
        px = frame[y:y + h, x:x + w][m].astype(np.float32)
        hv = hsv[y:y + h, x:x + w][m].astype(np.float32)
        B, G, R = px[:, 0].mean() + 1, px[:, 1].mean() + 1, px[:, 2].mean() + 1
        a = b["area"]
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = max(1.0, cv2.arcLength(cnts[0], True)) if cnts else 1.0
        circ = min(1.0, 4 * np.pi * a / per ** 2)
        # švytėjimas: vidutinis šviesumas žiede aplink dėmę / dėmės šviesumas
        pad = int(max(w, h) * 1.5) + 3
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        ring = gray[y0:y1, x0:x1].copy()
        inner = np.zeros_like(ring, bool)
        inner[y - y0:y - y0 + h, x - x0:x - x0 + w] = True
        halo = ring[~inner].mean() / (hv[:, 2].mean() + 1)
        hue = hv[:, 0]
        hue_red = np.mean((hue < 10) | (hue > 170))
        has_pair, pair_dy = _pair_info(blobs, k)
        # krašto žiedas (3 px aplink dėmę)
        crop_m = b["mask"][y0:y1, x0:x1].astype(np.uint8)
        edge = cv2.dilate(crop_m, np.ones((7, 7), np.uint8)).astype(bool) & ~crop_m.astype(bool)
        hs = hsv[y0:y1, x0:x1]
        ring_sat = hs[..., 1][edge].mean() / 255 if edge.any() else 0.0
        ring_v = hs[..., 2][edge].mean() / 255 if edge.any() else 0.0
        feats.append([
            np.log1p(a), hv[:, 2].max() / 255, hv[:, 2].mean() / 255,
            np.mean(hv[:, 2] >= 250),
            R / B, R / G, hue_red,
            circ, w / max(1, h), a / max(1, w * h),
            halo, b["c"][1] / H,
            has_pair, pair_dy,
            ring_sat, ring_v,
            _vstack(blobs, k),
        ])
    return np.array(feats, np.float32).reshape(-1, len(FEATURE_NAMES))


def blob_labels(blobs, label_map):
    """
    Tikroji dėmės klasė: jei dėmėje yra bent 2 automobilio žibinto pikseliai –
    automobilio klasė (saugos požiūriu svarbiausia), kitaip dažniausia reikšmė.
    """
    out = []
    for b in blobs:
        cnt = np.bincount(label_map[b["mask"]], minlength=5)
        if cnt[0] + cnt[1] >= 2:
            out.append(0 if cnt[0] >= cnt[1] else 1)
        else:
            out.append(int(cnt.argmax()))
    return np.array(out, np.int64)
