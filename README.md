# Automatinis tolimųjų šviesų perjungimas (DI prototipas)

Kamera + kompiuterinė rega + mašininis mokymasis: sistema atpažįsta priešpriešinių
automobilių priekinius ir priekyje važiuojančių automobilių galinius žibintus ir
automatiškai perjungia tolimąsias šviesas į artimąsias (KET 95 p.).

Palyginti du atpažinimo metodai su ta pačia sprendimų logika:
* **dėmių modelis** – šviesos dėmių išskyrimas + 17 požymių + atsitiktinis miškas,
  mokytas sintetiniais kadrais (pagrindinė sistema, v4);
* **YOLO11n** – konvoliucinis objektų detektorius (iš anksto išmokytas COCO arba
  apmokytas naktiniais vaizdais Google Colab aplinkoje).

## Failai

| Failas | Paskirtis |
| --- | --- |
| `synth.py` | Sintetinių naktinio kelio kadrų su žymėmis generatorius ir 60 s scenarijus |
| `features.py` | Šviesos dėmių išskyrimas (3 kaukės) ir 17 požymių |
| `train_classifier.py` | Dėmių modelio mokymas (Random Forest ir MLP, geresnis išsaugomas) |
| `train_hardneg.py` | v5 bandymas: mokymas su tikrais neigiamais pavyzdžiais (neigiamas rezultatas) |
| `detector.py` | `BlobDetector` (dėmių modelis), `OnnxYoloDetector` (YOLO per OpenCV), `RuleDetector` (be DI) |
| `controller.py` | Sprendimų logika: balsavimas 0,3 s lange, objektų sekimas, apšviesto kelio taisyklė, histerezė |
| `main.py` | Paleidimas su kamera ar vaizdo įrašu, rezultato vaizdo įrašas |
| `demo_compare.py` | Gynimui: dėmių modelis ir YOLO greta tame pačiame vaizdo įraše |
| `evaluate.py` | Vertinimas sintetiniame scenarijuje |
| `evaluate_real.py` | Vertinimas tikru vaizdo įrašu pagal rankiniu būdu pažymėtus automobilius |
| `real/*_cars.json` | Tikrų įrašų žymės: kada automobilis matomas ir kada prasilenkia |
| `real/*.json` | Visų bandytų metodų rezultatai (žr. lentelę žemiau) |
| `blob_classifier.joblib` | Dėmių modelis v4 (dabartinis, suspaustas) |
| `requirements.txt` | Reikalingos Python bibliotekos |
| `blob_classifier_v1.joblib` | Pradinis v1 modelis palyginimui |
| `yolo11n.onnx` | YOLO11n, iš anksto išmokytas COCO duomenimis |
| `best_night.onnx` | YOLO11n, apmokytas naktiniais vaizdais („Vehicle Detection Night time“) |
| `best_headlight.onnx` | YOLO11n, apmokytas žibintų rinkiniu (215 vaizdų; nepasiteisino) |
| `yolo_colab.ipynb`, `yolo_headlight_colab.ipynb` | YOLO mokymas Google Colab (T4 GPU) |
| `train_yolo.py` | YOLO mokymas su ultralytics vietiniame kompiuteryje (reikia GPU) |
| `seeds/` | Stabilumo patikrinimas: modeliai su skirtingais seed |

## Paleidimas („Windows“ cmd lange, `ahb` aplanke)

```
py -m pip install -r requirements.txt
py train_classifier.py
```

`train_classifier.py` iš naujo išmoko dėmių modelį su jūsų bibliotekų versijomis
(~3–4 min). Pateiktas `blob_classifier.joblib` išmokytas su scikit-learn 1.9.1.

## Vaizdo įrašai (testavimo duomenys)

Įrašai nėra saugykloje: jie atsisiunčiami iš [Pexels](https://www.pexels.com/license/)
(licencija leidžia nemokamai naudoti ir redaguoti, bet neleidžia platinti nepakeistų
įrašų). Atsisiųskite ir pervadinkite, tada įdėkite į aplanką `real`:

| Failas | Pexels įrašas | Autorius | Trukmė, raiška |
| --- | --- | --- | --- |
| `night.mp4` | [Night Drive on a Dark, Curvy Road](https://www.pexels.com/video/night-drive-on-a-dark-curvy-road-33650283/) | Kim Dodge | 1:55, 1920 × 1080 |
| `vingis.mp4` | [Driving On A Road At Night](https://www.pexels.com/video/driving-on-a-road-at-night-2876665/) | Jack Kazanjyan | 0:49, 3840 × 2160 |
| `naktis.mp4` | [Night Drive on Snowy Winter Road](https://www.pexels.com/video/night-drive-on-snowy-winter-road-35180062/) | Kim Dodge | 0:40, 1920 × 1080 |

Žymės (`real/*_cars.json`), kada automobilis matomas ir kada prasilenkia, sudarytos rankiniu
būdu peržiūrint kadrus.
Komentarų, prasidedančių `#`, į komandinę eilutę nekopijuokite.

### Vaizdo įrašai gynimui

```
py main.py --source real/vingis.mp4 --yolo best_night.onnx --yolo-classes all+filtras --save real/yolo_vingis.mp4
py main.py --source real/night.mp4  --yolo best_night.onnx --yolo-classes all+filtras --save real/yolo_night.mp4
py main.py --source real/naktis.mp4 --yolo best_night.onnx --yolo-classes all+filtras --save real/yolo_naktis.mp4
py demo_compare.py --video real/vingis.mp4 --out real/palyginimas_vingis.mp4
py demo_compare.py --video real/night.mp4  --out real/palyginimas_night.mp4
```

* `main.py` rodo langą (Esc – stabdyti); su `--no-show` tik išsaugo, greičiau.
* Numatytasis `--rezimas patarimas`: sistema tik **siūlo** vairuotojui perjungti šviesas
  (ekrane „SIULOMA: …“). `--rezimas automatinis` imituoja CAN komandą. Prototipas
  realių žibintų nevaldo nė viename režime.
* `demo_compare.py` – kairėje dėmių modelis, dešinėje naktinis YOLO, apačioje tikroji būsena.
* Dėmių modelis vietoj YOLO: `py main.py --source real/vingis.mp4`.

### Vertinimas

```
py evaluate_real.py --video real/vingis.mp4 --labels real/vingis_cars.json
py evaluate_real.py --video real/vingis.mp4 --labels real/vingis_cars.json --model best_night.onnx --yolo-classes all+filtras --tag ynight_f
```

`--yolo-classes`: `coco` (COCO transporto klasės), `all` (visos klasės), `1,6` (nurodytos);
`+filtras` – atmesti rėmelius, liečiančius kadro apačią (kapotas, skydelio atspindžiai)
ir platesnius nei 40 % kadro.

## Rezultatai tikruose įrašuose (akinimas / bereikalingos artimosios)

| Metodas | night | vingis | naktis | Akinimas iš viso |
| --- | --- | --- | --- | --- |
| Dėmių modelis v4 (+ sekimas) | 4,0 / 5,0 s | 0 / 37,1 s | 0,2 / 0 s | **4,1 s** |
| YOLO11n COCO + filtras | 20,9 / 0 s | 9,6 / 2,1 s | 5,5 / 0 s | 36,0 s |
| Naktinis YOLO11n + filtras | 20,1 / 0 s | 0,4 / 5,3 s | 0,2 / 0 s | 20,6 s |
| Žibintų YOLO11n + filtras | 23,0 / 0 s | 9,1 / 0 s | 32,9 / 0 s | 65,0 s |

* Dėmių modelis vienintelis mato tolimus žibintus tamsiame kelyje, bet baltus kelio
  stulpelius laiko automobiliais.
* Naktinis YOLO puikiai atskiria stulpelius ir artimus automobilius, bet nemato tolimų žibintų.
* `night` įrašu dėmių modelis buvo tobulinamas, todėl jo rezultatas ten optimistinis;
  `vingis` ir `naktis` – nepriklausomi testai.
