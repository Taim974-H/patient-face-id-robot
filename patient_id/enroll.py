"""Register patients from the command line.

    python -m patient_id.enroll --from-reference-images
    python -m patient_id.enroll --id P-1001 --name "Jane Doe" --qr CODE123 photo1.jpg photo2.jpg
"""
import argparse
import os
import re
from pathlib import Path

import cv2

from . import config
from .engine import FaceEngine
from .store import PatientStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--id")
    ap.add_argument("--name")
    ap.add_argument("--qr")
    ap.add_argument("--from-reference-images", action="store_true",
                    help="enrol every reference_images/*_base<N>_<name>.jpg (id = name, lower case)")
    args = ap.parse_args()

    s = config.load()
    engine = FaceEngine(s.models_dir, s.det_score, s.max_image_side, s.auto_brighten)
    store = PatientStore(os.path.join(s.data_dir, "patients.json"), s.max_patients)

    jobs = []  # (patient_id, name, image path)
    if args.from_reference_images:
        for path in sorted((config.ROOT / "reference_images").glob("*.[jp][pn]g")):
            m = re.search(r"base\d+_(.+)$", path.stem, re.IGNORECASE)
            name = (m.group(1) if m else path.stem).replace("_", " ").title()
            jobs.append((name.lower().replace(" ", "-"), name, path))
    elif args.id and args.images:
        jobs = [(args.id, args.name or args.id, Path(p)) for p in args.images]
    else:
        ap.error("give --from-reference-images, or --id with image files")

    for pid, name, path in jobs:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[skip] cannot read {path}")
            continue
        try:
            embs = engine.enrol_embeddings(img)
        except ValueError as exc:
            print(f"[skip] {path.name}: {exc}")
            continue
        evicted = store.enrol(pid, name, embs, args.qr)
        print(f"enrolled {pid} ({name}) from {path.name}" + (f"; evicted {evicted}" if evicted else ""))


if __name__ == "__main__":
    main()
