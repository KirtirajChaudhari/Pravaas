"""Prepare the requested Kaggle archive without extracting untrusted paths.

Retain official splits, remove exact decoded-image duplicates across splits
(test > validation > train), and record every retained image's provenance.
"""
import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


def prepare(archive, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, seen, skipped = [], set(), []
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png')) and '__MACOSX' not in n)
        def split(n):
            parts = n.lower().split('/')
            if any(p in ('test', 'testing') for p in parts): return 'test'
            if any(p in ('validation', 'valid', 'val') for p in parts): return 'val'
            if any(p in ('train', 'training') for p in parts): return 'train'
            return None
        names.sort(key=lambda n: {'test': 0, 'val': 1, 'train': 2, None: 3}[split(n)])
        for name in names:
            low = name.lower().replace('_', ' ').replace('-', ' ')
            label = 1 if any(x in low for x in ('non defective', 'nondefective', 'non defect')) else 0 if 'defect' in low else None
            if label is None:
                raise ValueError(f'Unknown class folder: {name}')
            raw = z.read(name)
            with Image.open(io.BytesIO(raw)) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                # Pixel hash catches the same image under different filenames.
                digest = hashlib.sha256(im.size.__repr__().encode() + im.tobytes()).hexdigest()
                if digest in seen:
                    skipped.append(name)
                    continue
                seen.add(digest)
                im.resize((224, 224), Image.Resampling.BILINEAR).save(output / f'{digest}.png')
            records.append({'file': f'{digest}.png', 'sha256_pixels': digest, 'source': name, 'label': label, 'split': split(name)})
    if not all(r['split'] for r in records) or set(r['split'] for r in records) != {'train', 'val', 'test'}:
        train, held = train_test_split(records, test_size=.3, stratify=[r['label'] for r in records], random_state=42)
        val, test = train_test_split(held, test_size=.5, stratify=[r['label'] for r in held], random_state=42)
        for group, tag in [(train,'train'),(val,'val'),(test,'test')]:
            for r in group: r['split'] = tag
        policy = 'Stratified 70/15/15 split, seed 42, after exact pixel deduplication'
    else:
        policy = 'Original train/validation/test splits; duplicate priority test > validation > train'
    meta = {'source': 'https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection', 'version': 2,
            'archive_sha256': hashlib.file_digest(open(archive, 'rb'), 'sha256').hexdigest(),
            'classes': ['Defective', 'Non-defective'], 'split_policy': policy,
            'counts': dict(Counter(r['split'] for r in records)), 'duplicates_removed': len(skipped),
            'limitation': 'Exact pixel deduplication does not exclude near-duplicates or shared capture locations.',
            'records': records}
    (output / 'manifest.json').write_text(json.dumps(meta, indent=2))
    print({k:v for k,v in meta.items() if k != 'records'}, flush=True)
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive')
    p.add_argument('--output', default='data/classification')
    a = p.parse_args()
    prepare(a.archive, a.output)
