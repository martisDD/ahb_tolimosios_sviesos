"""
Sprendimų logika: kada įjungti artimąsias, o kada grąžinti tolimąsias.

Saugos principas: į artimąsias perjungiama greitai, atgal į tolimąsias – lėtai
(tik po hold_s sekundžių be jokio automobilio). Tokia asimetriška histerezė
apsaugo nuo „mirksėjimo“ ir nuo akinimo, kai detektorius trumpam „pameta“
automobilį (pvz., už posūkio ar kalvos).

v2: vietoj „2 kadrai iš eilės“ naudojamas balsavimas laiko lange: artimosios
įjungiamos, kai automobilis matomas bent vote_ratio dalyje paskutinių window_s
sekundžių kadrų. Pro šalį prabėgantys atšvaitai klaidingai „aptinkami“ tik 4–6
kadrus, o tikras automobilis matomas ilgai.

v3: objektų sekimas. Kai automobilis patvirtintas, sistema prisimena jo padėtį ir
toliau jį seka su žemesne tikimybės riba (track_conf), jei dėmė yra netoli
(track_radius_px) paskutinės padėties. Testas parodė, kad artėjančio automobilio
tikimybė gali kelioms sekundėms nukristi žemiau 0,6 (pvz., kai žibintai susilieja
į vieną gelsvą dėmę su spinduliais); sekimas neleidžia jo „pamesti“.
"""
from collections import deque
import cv2
from dataclasses import dataclass
import numpy as np

from synth import F_PX, HEADLIGHT_GAP


def estimate_distance(pair_px):
    """Atstumo įvertis pagal tarpą tarp žibintų porų (skylutės kameros modelis)."""
    return F_PX * HEADLIGHT_GAP / max(pair_px, 1e-3)


@dataclass
class BeamController:
    fps: float = 20.0
    window_s: float = 0.3         # balsavimo langas
    vote_ratio: float = 0.6       # kokia lango dalis turi turėti automobilį
    hold_s: float = 2.0           # tiek sekundžių be automobilio -> tolimosios
    min_conf: float = 0.6         # tikimybės riba naujam automobiliui
    lit_thr: float = 35.0         # (tik senajai sąsajai) viso kadro šviesumo riba
    sky_frac: float = 0.35        # viršutinė kadro dalis, laikoma „dangumi“
    sky_thr: float = 30.0         # dangaus šviesumo riba apšviestam keliui
    lamps_min: float = 2.0        # kiek gatvės žibintų vidutiniškai turi matytis
    lit_window_s: float = 1.0     # apšvietimas vertinamas per šį laiką
    track: bool = True            # ar sekti patvirtintą automobilį
    track_conf: float = 0.35      # žemesnė riba jau sekamam automobiliui
    track_radius_px: float = 40.0 # kiek pikselių per kadrą gali pasislinkti sekamas objektas

    def __post_init__(self):
        self.high = True
        self.window = deque(maxlen=max(2, int(round(self.window_s * self.fps))))
        self.unseen = 0
        self.switches = 0
        self.tracks = []          # sekamų automobilių padėtys (x, y)
        self.lit_hist = deque(maxlen=max(2, int(round(self.lit_window_s * self.fps))))

    def _lit(self, frame_or_mean, dets):
        """
        Ar kelias apšviestas (KET 95.1). v4: vertinamas ne viso kadro šviesumas
        (snieguotą kelią mūsų pačių žibintai padaro šviesų), o per paskutinę
        sekundę: dangaus srities šviesumas IR gatvės žibintų skaičius.
        Senoji sąsaja (perduotas skaičius) – viso kadro vidurkis > lit_thr.
        """
        if np.isscalar(frame_or_mean):
            return frame_or_mean > self.lit_thr
        f = frame_or_mean
        top = f[: int(self.sky_frac * f.shape[0])]
        sky = float((cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) if top.ndim == 3 else top).mean())
        lamps = sum(1 for d in (dets or []) if d.get("cls") == 2)
        self.lit_hist.append((sky, lamps))
        skies, lamp_counts = zip(*self.lit_hist)
        return np.median(skies) > self.sky_thr and np.mean(lamp_counts) >= self.lamps_min

    def _follow(self, dets):
        """Sekamas automobilis matomas, jei šalia jo yra dėmė su P >= track_conf."""
        new, seen = [], False
        for tx, ty in self.tracks:
            near = [d for d in dets if d["prob"] >= self.track_conf
                    and np.hypot(d["c"][0] - tx, d["c"][1] - ty) <= self.track_radius_px]
            if near:
                best = max(near, key=lambda d: d["prob"])
                new.append(best["c"])
                seen = True
        return new, seen

    def update(self, vehicle_probs, frame_mean, dets=None):
        """
        vehicle_probs – kiekvienos dėmės tikimybė būti automobilio žibintu.
        frame_mean    – kadras (BGR masyvas) arba, senajai sąsajai, jo vidutinis šviesumas;
                        naudojamas apšviestam keliui aptikti (KET 95.1).
        dets          – (nebūtina) aptikimai su padėtimis {"c": (x, y), "prob": p};
                        reikalingi objektų sekimui.
        Grąžina True, jei turi degti tolimosios.
        """
        vehicle = bool(len(vehicle_probs)) and float(np.max(vehicle_probs)) >= self.min_conf
        lit = self._lit(frame_mean, dets)

        tracked = False
        if self.track and dets is not None:
            strong = [d["c"] for d in dets if d["prob"] >= self.min_conf]
            if not self.high:
                self.tracks, tracked = self._follow(dets)
            # nauji stiprūs aptikimai tampa sekamais objektais
            for c in strong:
                if all(np.hypot(c[0] - x, c[1] - y) > self.track_radius_px for x, y in self.tracks):
                    self.tracks.append(c)

        self.window.append(vehicle or lit)
        confirmed = sum(self.window) >= self.vote_ratio * self.window.maxlen
        present = vehicle or lit or tracked
        self.unseen = 0 if present else self.unseen + 1

        prev = self.high
        if self.high and confirmed:
            self.high = False
        elif not self.high and self.unseen >= self.hold_s * self.fps:
            self.high = True
            self.tracks = []
        if self.high and not confirmed:
            self.tracks = []      # kol tolimosios – sekame tik patvirtintus
        if prev != self.high:
            self.switches += 1
        return self.high
