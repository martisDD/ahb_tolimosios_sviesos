"""
Sistemos vertinimas 60 s scenarijuje (žr. synth.scenario).

Matuojama:
  * kokiu atstumu iki priešpriešinio automobilio įjungtos artimosios
    (KET 95.2: ne mažiau kaip 150 m);
  * kadrų dalis, kai sistema elgėsi teisingai;
  * akinimo kadrai (tolimosios, nors automobilis < 150 m arba priekyje važiuojantis);
  * bereikalingos artimosios (be priežasties) ir perjungimų skaičius.
Rezultatai: eval_results.json, timeline.png, demo.mp4
"""
import argparse, json, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth import scenario, render, CLASS_NAMES
from detector import BlobDetector, RuleDetector
from controller import BeamController
from main import draw

FPS = 20
DETECT_RANGE = 800   # m – gamintojų sistemos žibintus aptinka iki ~800–850 m


def required_low(s):
    """Tikroji būsena: ar turi degti artimosios (KET 95.1–95.3)."""
    if s.lit_road:
        return True
    if s.oncoming_dist is not None and s.oncoming_dist <= 150:
        return True
    if s.preceding_dist is not None:
        return True
    return False


def desirable_low(s):
    return required_low(s) or (s.oncoming_dist is not None and s.oncoming_dist <= DETECT_RANGE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["ai", "rule"], default="ai")
    args = ap.parse_args()
    tag = "" if args.detector == "ai" else "_rule"
    rng = np.random.default_rng(123)
    det = BlobDetector() if args.detector == "ai" else RuleDetector()
    ctl = BeamController(fps=FPS)
    scenes = scenario(FPS)
    vw = cv2.VideoWriter(f"demo{tag}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 360))
    log, t_proc = [], []
    for i, s in enumerate(scenes):
        img, _ = render(s, rng)
        t0 = time.perf_counter()
        dets = det.detect(img)
        high = ctl.update([d["prob"] for d in dets if d["cls"] in (0, 1)], img, dets)
        t_proc.append(time.perf_counter() - t0)
        vw.write(draw(img.copy(), dets, high, FPS))
        log.append({"t": i / FPS, "high": high, "req_low": required_low(s),
                    "des_low": desirable_low(s), "onc": s.oncoming_dist,
                    "pre": s.preceding_dist, "lit": s.lit_road})
    vw.release()

    high = np.array([r["high"] for r in log])
    req = np.array([r["req_low"] for r in log])
    des = np.array([r["des_low"] for r in log])
    glare = int(np.sum(high & req))
    # artimosios be priežasties, neskaičiuojant leistino 2 s „palaikymo“ po priežasties
    hold = int(ctl.hold_s * FPS)
    recent = np.array([des[max(0, i - hold):i + 1].any() for i in range(len(des))])
    needless = int(np.sum(~high & ~recent))
    # atstumas, kuriuo įjungtos artimosios kiekvienam priešpriešiniam epizodui
    switch_dist = []
    for t0, t1 in ((10, 22), (48, 60)):
        idx = [i for i, r in enumerate(log) if t0 <= r["t"] < t1 and not r["high"]]
        if idx:
            switch_dist.append(round(log[idx[0]]["onc"], 1))
    res = {
        "frames": len(log),
        "glare_frames": glare,
        "needless_low_frames": needless,
        "required_low_frames": int(req.sum()),
        "compliance_frac": round(1 - glare / max(1, int(req.sum())), 4),
        "free_road_frames": int((~recent).sum()),
        "high_on_free_road_frac": round(float(np.mean(high[~recent])), 4),
        "switches": ctl.switches,
        "switch_distance_m": switch_dist,
        "ms_per_frame": round(1000 * float(np.mean(t_proc)), 2),
    }
    print(json.dumps(res, indent=2))
    with open(f"eval_results{tag}.json", "w") as f:
        json.dump(res, f, indent=2)

    t = np.array([r["t"] for r in log])
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    onc = np.array([r["onc"] if r["onc"] is not None else np.nan for r in log])
    ax[0].plot(t, onc, color="#2a6fdb", lw=1.6, label="Atstumas iki priešpriešinio automobilio")
    ax[0].axhline(150, color="#c0392b", ls="--", lw=1, label="KET 95.2 riba – 150 m")
    ax[0].set_ylabel("m"); ax[0].legend(loc="upper right", fontsize=8, frameon=False)
    ax[0].set_ylim(0, 750)
    for a, b, lbl in ((26, 34, "priekyje automobilis"), (38, 46, "gyvenvietė")):
        ax[0].axvspan(a, b, color="#999", alpha=0.15)
        ax[0].text((a + b) / 2, 690, lbl, ha="center", fontsize=8, color="#555")
    ax[1].step(t, high.astype(int), where="post", color="#1e8449", lw=1.6, label="Sistema")
    ax[1].step(t, (~req).astype(int) * 0.9 + 0.05, where="post", color="#888", lw=1, ls=":",
               label="Leidžiama (KET)")
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["Artimosios", "Tolimosios"])
    ax[1].set_xlabel("Laikas, s"); ax[1].legend(loc="lower right", fontsize=8, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"timeline{tag}.png", dpi=150)


if __name__ == "__main__":
    main()
