#!/bin/bash
# Kiekvieną seed modelį įvertina tikru įrašu su sekimu ir be jo
cd /home/claude/ahb
for s in 1 2 3 4 5; do
  if [ $s = 1 ]; then m=blob_classifier.joblib; else m=seeds/rf_seed$s.joblib; fi
  python3 evaluate_real.py --video real/night.mp4 --labels real/night_cars.json --model $m --tag seed${s}_notrack --no-track > /dev/null
  python3 evaluate_real.py --video real/night.mp4 --labels real/night_cars.json --model $m --tag seed${s}_track > /dev/null
done
