"""Download the user's requested public Kaggle dataset, without credentials in code."""
import argparse
import os
import urllib.request
import zipfile
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/kaggle.zip');a=p.parse_args()
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.partial')
    url='https://www.kaggle.com/api/v1/datasets/download/salmaneunus/railway-track-fault-detection?datasetVersionNumber=2'
    try:
        urllib.request.urlretrieve(url,temp)
        if not zipfile.is_zipfile(temp):raise ValueError('Kaggle did not return a ZIP. Download version 2 manually after signing in.')
        os.replace(temp,target)
        print(target)
    finally:
        if temp.exists():temp.unlink()
