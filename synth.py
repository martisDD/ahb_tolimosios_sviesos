"""
Sintetinių naktinio kelio kadrų generatorius (v2).

Kiekvienas kadras turi tikrąsias žymes (ground truth): kiekvienas šviesos šaltinis
nupiešiamas ir į vaizdą, ir į žymių žemėlapį (label map), todėl iš jo galima
automatiškai sugeneruoti mokymo duomenis klasifikatoriui.

Klasės:
    0 - priekinis žibintas (priešpriešinis automobilis)
    1 - galinis žibintas (priekyje važiuojantis automobilis)
    2 - gatvės žibintas
    3 - atšvaitas / kelio ženklas (atspindi mūsų pačių šviesą)
    4 - triukšmas (fonas, apšviestas asfaltas, atsitiktiniai blyksniai)

v2 pakeitimai (po testo su tikru vaizdo įrašu):
  * kelias gali būti posūkyje – objektai atsiduria bet kurioje kadro vietoje;
  * priekiniai žibintai: LED (melsvai balti), halogeniniai (gelsvi), ksenoniniai;
    tolimi žibintai dažnai nepersotinti, kartais matomas tik vienas;
  * kelio ženklai bet kurioje kelio pusėje, geltoni arba balti;
  * atšvaitai balti, gintariniai arba raudoni, kartais tankiai ant atitvaros;
  * mūsų tolimosios apšviečia asfaltą (didelė šviesi sritis kadro apačioje);
  * atsitiktinė kameros ekspozicija (kadras tamsesnis ar šviesesnis);
  * atšvaitų ir ženklų ryškumas krenta su atstumu (jie tik atspindi mūsų šviesą)
    ir paprastai yra mažesnis nei žibintų;
  * ryškios vietos „išbąla“ kaip tikroje kameroje (spalva lieka tik krašte);
  * kelio linijos (geltona vidurio ir baltos kraštų), ryškios mūsų šviesoje;
  * vertikalūs atšvaitų stulpeliai ir įvairių proporcijų ženklai;
  * v4: europietiški balti kelio stulpeliai su juoda juosta ir atšvaitu,
    balta punktyrinė vidurio linija ir drebančios kameros suliejimas;
  * žibintai „išsilieja“ (bloom) – net toli matomi kaip kelių pikselių dėmė;
    arti jie netaisyklingos formos, su atspindžiu šlapiame asfalte ir dažniausiai
    nepersotinti (sumažinus kadrą iki 640 px šviesumas suvidurkinamas).

Kamera modeliuojama kaip skylutės (pinhole) kamera; posūkis – parabolinis poslinkis:
    u = cx + f * (X + k * Z^2 / 2) / Z
    v = horizon + f * (cam_h - Y) / Z
"""
from dataclasses import dataclass, field
import numpy as np
import cv2

W, H = 640, 360
F_PX = 700.0          # židinio nuotolis pikseliais
CX = W / 2
HORIZON = 150
CAM_H = 1.3           # kameros aukštis, m
HEADLIGHT_GAP = 1.5   # atstumas tarp žibintų, m
CLASS_NAMES = ["headlight", "taillight", "streetlight", "reflector", "noise"]
VEHICLE_CLASSES = (0, 1)

HEADLIGHT_COLORS = [(240, 245, 255), (170, 215, 255), (255, 240, 225)]   # LED, halogen, xenon
REFLECTOR_COLORS = [(235, 235, 235), (40, 170, 245), (40, 40, 220)]      # balti, gintariniai, raudoni
SIGN_COLORS = [(20, 200, 235), (225, 225, 220)]                          # geltoni, balti


def project(X, Y, Z, curve=0.0):
    return CX + F_PX * (X + curve * Z * Z / 2) / Z, HORIZON + F_PX * (CAM_H - Y) / Z


@dataclass
class Light:
    X: float
    Y: float
    Z: float
    kind: int
    color: tuple            # BGR
    radius_m: float
    intensity: float        # 0..1, centro šviesumas
    halo: float             # švytėjimo stiprumas (0 = be švytėjimo)
    rect: bool = False      # stačiakampė forma (kelio ženklai)
    saturate: bool = False  # persotinta (balta) šerdis
    bloom_px: float = 0.0   # mažiausias matomas spindulys px (kameros „išsiliejimas“)
    wet: bool = False       # atspindys šlapiame asfalte po žibintu
    aspect: float = 1.3     # stačiakampio pločio ir aukščio santykis


@dataclass
class Scene:
    """Vieno kadro scena: šviesos šaltiniai + tikrasis „ar reikia artimųjų“."""
    lights: list = field(default_factory=list)
    oncoming_dist: float | None = None
    preceding_dist: float | None = None
    lit_road: bool = False
    curve: float = 0.0      # kelio kreivumas, 1/m (+ į dešinę)
    beam: float = 0.0       # kiek mūsų šviesos apšviečia asfaltą (0..1)
    gain: float = 1.0       # kameros ekspozicija
    markings: float = 0.0   # kelio linijų ryškumas (0 = nėra)
    center: str = "yellow"  # vidurio linija: "yellow" (dviguba) arba "dashed" (balta punktyrinė)
    dash_phase: float = 0.0
    motion_blur: float = 0.0  # kameros drebėjimo suliejimas, px


def car_lights(dist, lateral, oncoming, rng, style=None):
    """Du žibintai: priekiniai (įvairių tipų) arba raudoni galiniai."""
    style = style or {}
    lights = []
    if oncoming:
        color = style.get("hl_color", HEADLIGHT_COLORS[0])
        inten = style.get("hl_int", 1.0)
        halo = style.get("hl_halo", 1.0)
        sat = style.get("hl_sat", True)
        sides = (-0.5, 0.5) if not style.get("single") else (0.5,)
        for side in sides:
            lights.append(Light(lateral + side * HEADLIGHT_GAP, 0.65, dist, 0, color,
                                0.11, inten, halo, saturate=sat,
                                bloom_px=style.get("bloom", 1.0), wet=style.get("wet", False)))
    else:
        for side in (-0.5, 0.5):
            lights.append(Light(lateral + side * 1.3, 0.85, dist, 1, (40, 40, 230), 0.09,
                                style.get("tl_int", 0.75), 0.45))
    return lights


def random_car_style(rng):
    return {"hl_color": HEADLIGHT_COLORS[rng.integers(3)],
            "hl_int": rng.uniform(0.85, 1.0),
            "hl_halo": rng.uniform(0.2, 1.1),
            "bloom": rng.uniform(1.0, 3.5),
            "hl_sat": rng.random() < 0.25,
            "wet": rng.random() < 0.5,
            "single": rng.random() < 0.1,
            "tl_int": rng.uniform(0.5, 0.9)}


def random_road_style(rng):
    return {"seed": int(rng.integers(1 << 30)),
            "post_spacing": rng.uniform(25, 50),
            "post_color": int(rng.integers(3)),
            "post_lat": rng.uniform(3.5, 6.0),
            "guardrail": rng.random() < 0.25,
            "reach": rng.uniform(50, 130),
            "tall_posts": rng.random() < 0.3,
            "euro_posts": rng.random() < 0.35,
            "markings": rng.random() < 0.8,
            "signs_per_km": rng.uniform(5, 25)}


DEFAULT_ROAD = {"seed": 11, "post_spacing": 50, "post_color": 1, "post_lat": 4.2,
                "guardrail": True, "signs_per_km": 12, "reach": 90,
                "tall_posts": False, "euro_posts": False, "markings": False}


def _obj_rng(style, kind, k):
    """Kiekvienam objektui pastovūs parametrai (kad judant nesikeistų)."""
    return np.random.default_rng((style["seed"], kind, int(k)))


def retro(z, reach):
    """
    Atšvaito ryškumas pagal atstumą: jis šviečia tik atspindėdamas mūsų šviesą,
    todėl toli (už reach metrų) greitai blanksta, o automobilio žibintas – ne.
    """
    return float(min(1.0, (reach / z) ** 2))


def euro_post(x, z, o, reach):
    """
    Europietiškas kelio stulpelis (v4): baltas ~1 m aukščio korpusas su juoda
    juosta viršuje ir atšvaitu joje. Mūsų tolimosiose baltas korpusas tampa ryškus
    ir bespalvis, o juoda juosta jį suskaido į kelias dėmes viena virš kitos.
    """
    k = retro(z, 0.8 * reach)
    body = o.uniform(0.5, 0.95) * k
    refl = REFLECTOR_COLORS[o.choice([0, 0, 1, 2])]
    w = 0.12
    halo = o.uniform(0.15, 0.6)                          # suliejimas / švytėjimas kameroje
    parts = [
        # (aukštis Y, pusė aukščio m, spalva, ryškumas, klasė, švytėjimas)
        (0.35, 0.35, (220, 222, 218), body, 3, halo),          # apatinė balta dalis
        (0.89, 0.11, (12, 12, 12), 1.0, 4, 0.0),               # juoda juosta
        (0.89, 0.05, refl, min(1.0, body * 1.3), 3, halo),     # atšvaitas juostoje
        (1.05, 0.045, (220, 222, 218), body, 3, halo),         # viršutinė balta dalis
    ]
    out = []
    for y, half_h, col, inten, kind, hl in parts:
        out.append(Light(x, y, z, kind, col, half_h, inten, hl, rect=True,
                         aspect=(w / half_h) if half_h > 0.1 else 1.2))
    return out


def roadside(rng, offset_m, lit_road, style=None):
    """
    Atšvaitų stulpeliai, atitvaros atšvaitai, kelio ženklai ir (gyvenvietėje) gatvės
    žibintai. Atšvaitai ir ženklai apšviesti mūsų tolimųjų šviesų, todėl gali būti
    labai ryškūs – tai pagrindinis klaidingų aptikimų šaltinis.
    """
    style = style or DEFAULT_ROAD
    lights = []
    sp = style["post_spacing"]
    k0 = int(np.floor(offset_m / sp))
    for k in range(k0, k0 + int(600 / sp) + 2):
        z = k * sp - offset_m + 10
        if z < 6:
            continue
        o = _obj_rng(style, 1, k)
        col = REFLECTOR_COLORS[style["post_color"]] if o.random() < 0.85 else REFLECTOR_COLORS[o.integers(3)]
        tall = style.get("tall_posts", False)
        euro = style.get("euro_posts", False)
        for side in (-1, 1):
            if euro:
                lights += euro_post(side * style["post_lat"], z, o, style["reach"])
            elif tall:    # vertikali atšvaito juosta ant stulpelio
                lights.append(Light(side * style["post_lat"], 0.9, z, 3, col, o.uniform(0.08, 0.14),
                                    o.uniform(0.45, 1.0) * retro(z, style["reach"]), o.uniform(0.0, 0.35),
                                    rect=True, aspect=o.uniform(0.25, 0.5)))
            else:
                lights.append(Light(side * style["post_lat"], 0.8, z, 3, col, o.uniform(0.04, 0.08),
                                    o.uniform(0.35, 0.9) * retro(z, style["reach"]), o.uniform(0.0, 0.35)))
    if style["guardrail"]:      # tankūs atšvaitai ant atitvaros kairėje, 4 m žingsniu
        k0 = int(np.floor(offset_m / 4))
        for k in range(k0, k0 + 40):
            z = k * 4 - offset_m + 8
            o = _obj_rng(style, 2, k)
            if z > 6 and o.random() < 0.6:
                lights.append(Light(-4.8, 0.7, z, 3, REFLECTOR_COLORS[0], 0.05,
                                    o.uniform(0.35, 0.9) * retro(z, style["reach"]), o.uniform(0.0, 0.3)))
    step = 1000 / style["signs_per_km"]
    k0 = int(np.floor(offset_m / step))
    for k in range(k0, k0 + int(500 / step) + 2):
        o = _obj_rng(style, 3, k)
        z = k * step - offset_m + o.uniform(0, step * 0.5)
        if z > 8:
            lat = o.choice([-1, 1]) * o.uniform(4.5, 7.5)
            lights.append(Light(lat, o.uniform(1.5, 2.6), z, 3, SIGN_COLORS[o.integers(2)],
                                o.uniform(0.28, 0.45), o.uniform(0.4, 0.95) * retro(z, 1.8 * style["reach"]),
                                o.uniform(0.0, 0.25), rect=True, aspect=o.uniform(1.0, 3.0)))
    if lit_road:
        for i in range(8):
            z = 15 + 35 * i - offset_m % 35
            if z > 6:
                lights.append(Light(-6.0, 8.0, z, 2, (90, 190, 255), 0.25, 1.0, 1.2))
    return lights


def _marking(img, labels, curve, x, width, color, beam, gain, z0=5.0, z1=150.0, dash=None):
    """
    Kelio linija: ryški tik ten, kur ją apšviečia mūsų šviesa (atšvaitiniai dažai).
    dash = (brūkšnio ilgis, tarpas, fazė) metrais – punktyrinė linija.
    """
    if dash:
        d, g, ph = dash
        starts = np.arange(z0 - ph % (d + g), z1, d + g)
        segs = [(max(z0, a), a + d) for a in starts if a + d > z0]
    else:
        zs = np.geomspace(z0, z1, 30)
        segs = list(zip(zs[:-1], zs[1:]))
    for za, zb in segs:
        k = beam * min(1.0, (35.0 / za) ** 1.5) * gain
        if k < 0.05:
            break
        pts = [project(x - width / 2, 0, za, curve), project(x + width / 2, 0, za, curve),
               project(x + width / 2, 0, zb, curve), project(x - width / 2, 0, zb, curve)]
        poly = np.array(pts, np.int32)
        cv2.fillPoly(img, [poly], (np.array(color, np.float32) * k).tolist())
        cv2.fillPoly(labels, [poly], 3)


def _road_poly(curve, half=3.6, z0=4.0, z1=800.0):
    zs = np.geomspace(z0, z1, 40)
    left = [project(-half, 0, z, curve) for z in zs]
    right = [project(half, 0, z, curve) for z in zs][::-1]
    return np.array(left + right, np.int32), zs


def render(scene: Scene, rng):
    """Grąžina (BGR kadras, žymių žemėlapis)."""
    img = np.zeros((H, W, 3), np.float32)
    img[:HORIZON] = (14, 8, 6)
    poly, _ = _road_poly(scene.curve, half=6.0)
    cv2.fillPoly(img, [poly], (18, 18, 18))
    if scene.beam > 0:            # mūsų pačių šviesa ant asfalto
        beam = np.zeros((H, W), np.float32)
        near, _ = _road_poly(scene.curve, half=4.0, z0=4.0, z1=70.0)
        cv2.fillPoly(beam, [near], 1.0)
        beam = cv2.GaussianBlur(beam, (0, 0), 25)
        ramp = np.clip((np.arange(H)[:, None] - HORIZON) / (H - HORIZON), 0, 1) ** 0.6
        img += (beam * ramp * 175 * scene.beam)[..., None] * np.array([0.95, 0.97, 1.0], np.float32)
    if scene.lit_road:
        img += 25
    labels = np.full((H, W), 4, np.uint8)
    if scene.markings and scene.beam > 0:
        mb = scene.beam * scene.markings
        if scene.center == "yellow":                  # dviguba geltona vidurio linija (JAV)
            for x in (-0.12, 0.12):
                _marking(img, labels, scene.curve, x, 0.1, (40, 175, 205), mb, 1.0)
        else:                                         # balta punktyrinė (Europa): 3 m / 9 m
            _marking(img, labels, scene.curve, 0.0, 0.12, (215, 215, 215), mb, 1.0,
                     dash=(3.0, 9.0, scene.dash_phase))
        for x in (-3.6, 3.6):                         # baltos kraštų linijos
            _marking(img, labels, scene.curve, x, 0.12, (200, 200, 200), mb, 1.0)
    glow = np.zeros_like(img)

    for L in sorted(scene.lights, key=lambda l: -l.Z):
        u, v = project(L.X, L.Y, L.Z, scene.curve)
        r = max(1.0, L.bloom_px, F_PX * L.radius_m / L.Z)
        if not (-20 < u < W + 20 and -20 < v < H + 20):
            continue
        c = np.array(L.color, np.float32) / 255.0
        core = c * 255 * L.intensity
        if L.rect:
            w, h = max(1, int(r * L.aspect)), max(1, int(r))
            cv2.rectangle(img, (int(u - w), int(v - h)), (int(u + w), int(v + h)), core.tolist(), -1)
            cv2.rectangle(labels, (int(u - w), int(v - h)), (int(u + w), int(v + h)), L.kind, -1)
        else:
            cv2.circle(img, (int(u), int(v)), int(round(r)), core.tolist(), -1, lineType=cv2.LINE_AA)
            cv2.circle(labels, (int(u), int(v)), int(round(r)), L.kind, -1)
            if L.saturate:  # persotinta šerdis – kamera „perdega“
                cv2.circle(img, (int(u), int(v)), max(1, int(r * 0.6)), (255, 255, 255), -1)
            if L.kind == 0 and r >= 1.5:
                # netaisyklinga dėmė: objektyvo atspindžiai šalia žibinto
                for _ in range(2):
                    du, dv = rng.uniform(-r, r, 2)
                    rr = max(1, int(r * rng.uniform(0.3, 0.6)))
                    p = (int(u + du), int(v + dv))
                    cv2.circle(img, p, rr, (core * rng.uniform(0.7, 1.0)).tolist(), -1)
                    cv2.circle(labels, p, rr, 0, -1)
                if L.wet:   # vertikalus atspindys šlapiame asfalte
                    ax = (max(1, int(r * 0.6)), max(2, int(r * rng.uniform(1.5, 3.0))))
                    ctr = (int(u), int(v + r + ax[1]))
                    cv2.ellipse(img, ctr, ax, 0, 0, 360, (core * rng.uniform(0.45, 0.75)).tolist(), -1)
                    cv2.ellipse(labels, ctr, ax, 0, 0, 360, 0, -1)
        if L.halo > 0:
            if L.rect:   # stačiakampiui – švytėjimas aplink jo kontūrą, ne didžiulis apskritimas
                w, h = max(1, int(r * L.aspect)), max(1, int(r))
                cv2.rectangle(glow, (int(u - w - 3), int(v - h - 3)), (int(u + w + 3), int(v + h + 3)),
                              (c * 120 * L.halo).tolist(), -1)
            else:
                cv2.circle(glow, (int(u), int(v)), int(r * 3 + 2), (c * 120 * L.halo).tolist(), -1)

    glow = cv2.GaussianBlur(glow, (0, 0), 4)
    img = (img + glow) * scene.gain
    # kamera „išbalina“ ryškias vietas: spalva lieka tik dėmės krašte
    lum = img.max(axis=2, keepdims=True)
    wgt = np.clip((lum - 185) / 70, 0, 1) * 0.85
    img = img * (1 - wgt) + np.minimum(lum, 255) * wgt
    for _ in range(rng.integers(0, 4)):
        p = (int(rng.uniform(0, W)), int(rng.uniform(HORIZON, H)))
        cv2.circle(img, p, 1, (120, 120, 120), -1)
    if scene.motion_blur >= 1.5:                      # drebanti kamera
        n = int(round(scene.motion_blur)) | 1
        k = np.zeros((n, n), np.float32)
        ang = rng.uniform(0, np.pi)
        c = n // 2
        for i in range(n):
            t = i - c
            k[int(round(c + t * np.sin(ang))), int(round(c + t * np.cos(ang)))] = 1
        img = cv2.filter2D(img, -1, k / k.sum())
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    img += rng.normal(0, 4, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), labels


def random_scene(rng):
    """Atsitiktinė scena mokymo duomenims."""
    s = Scene(lit_road=rng.random() < 0.2,
              curve=rng.normal(0, 0.0012) if rng.random() < 0.7 else 0.0,
              beam=rng.uniform(0.0, 0.9) if rng.random() < 0.7 else 0.0,
              gain=rng.uniform(0.7, 1.15))
    road = random_road_style(rng)
    s.markings = rng.uniform(0.6, 1.15) if road["markings"] else 0.0
    s.center = "dashed" if rng.random() < 0.5 else "yellow"
    s.dash_phase = rng.uniform(0, 12)
    s.motion_blur = rng.uniform(1.5, 7) if rng.random() < 0.35 else 0.0
    s.lights += roadside(rng, rng.uniform(0, 2000), s.lit_road, road)
    for _ in range(rng.integers(0, 3)):
        d = rng.uniform(15, 700)
        s.lights += car_lights(d, rng.uniform(-5.5, -2.5), True, rng, random_car_style(rng))
        s.oncoming_dist = d if s.oncoming_dist is None else min(s.oncoming_dist, d)
    if rng.random() < 0.4:
        d = rng.uniform(10, 300)
        s.lights += car_lights(d, rng.uniform(-0.5, 0.5), False, rng, random_car_style(rng))
        s.preceding_dist = d
    return s


def scenario(fps=20):
    """
    Deterministinis 60 s vertinimo scenarijus (užmiesčio kelias, 90 km/h):
      0–10 s   tuščias kelias su atšvaitais ir ženklais
      10–22 s  priešpriešinis automobilis 700 m -> 0 m (artėjimo greitis 50 m/s)
      26–34 s  priekyje važiuojantis automobilis 60–120 m
      38–46 s  apšviesta gyvenvietė
      48–60 s  du priešpriešiniai automobiliai vienas po kito
    Grąžina Scene sąrašą.
    """
    rng = np.random.default_rng(7)
    frames = []
    n = 60 * fps
    car = {"hl_color": HEADLIGHT_COLORS[0], "hl_int": 1.0, "hl_halo": 1.0, "hl_sat": True}
    for i in range(n):
        t = i / fps
        s = Scene(lit_road=38 <= t < 46)
        s.lights += roadside(rng, t * 25.0, s.lit_road)
        if 10 <= t < 22:
            d = 700 - (t - 10) * 50 + 15
            s.lights += car_lights(d, -3.6, True, rng, car)
            s.oncoming_dist = d
        if 26 <= t < 34:
            d = 90 + 30 * np.sin((t - 26) / 8 * np.pi)
            s.lights += car_lights(d, 0.0, False, rng)
            s.preceding_dist = d
        if 48 <= t < 60:
            for k in (0, 1):
                d = 600 - (t - 48) * 50 + k * 60
                if d > 12:
                    s.lights += car_lights(d, -3.6, True, rng, car)
                    s.oncoming_dist = d if s.oncoming_dist is None else min(s.oncoming_dist, d)
        frames.append(s)
    return frames
