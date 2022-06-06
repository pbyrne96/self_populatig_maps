import pandas as pd
import os
from typing import Dict
from collections import defaultdict
from utils.numpy_utils import normalize_points
import numpy as np
from scipy import stats


FILE_LOCATION = '/locations'
UNWANTED_COLS = 'location'

strip_lat_lon = r'(?P<d>[\d\.]+).*?(?P<m>[\d\.]+).*?(?P<s>[\d\.]+)'

def read_return_df(filename) -> pd.DataFrame:
    with open(filename) as f:
        lines = f.readlines()
        columns = lines[0].replace('\n','').lower().split('\t')
        all_data = defaultdict(list)

        line: str
        for _,line in enumerate(lines[1:]):
            lat_long = tuple(line.replace('\n','').split('\t'))
            if not(lat_long) or len(lat_long) < 3:
                continue
            for index, col_name in enumerate(columns):
                all_data[col_name].append(lat_long[index])
        return pd.DataFrame.from_dict(all_data)

def convert_lat_lon(df: pd.DataFrame, col:str):
    dms = df[col].str.extract(strip_lat_lon).astype(float)
    _dms = dms['d'] + dms['m'].div(60) + dms['s'].div(3600)
    return _dms


def read_files() -> Dict[str, pd.DataFrame]:
    full_path = os.path.dirname(os.path.abspath(__file__)) + FILE_LOCATION
    all_files = list(map(lambda f: full_path + '/' +f ,os.listdir(full_path)))
    all_data: Dict[str, pd.DataFrame] = {}

    _file: str
    for _file in all_files:
        filename = _file.split('/')[-1].replace('.txt','')
        df = read_return_df(_file)
        cols_to_change = list(df.columns)[1:]
        for c in cols_to_change:
            df[c] = convert_lat_lon(df,c)
        df = df[cols_to_change].dropna()
        normalized_df = normalize_points(df)
        normalized_df = normalized_df[(np.abs(stats.zscore(normalized_df)) < 3).all(axis=1)]
        all_data[filename] = df.dropna()
    return all_data

import cv2
import glob

def image_to_video():
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    (h, w) = cv2.imread(glob.glob("diagrams/*.png")[0]).shape[:2]

    out = cv2.VideoWriter('video.mp4', fourcc, 2, (w, h), isColor=True)

    for img in sorted(glob.glob("diagrams/*.png")):
        img = cv2.imread(img)
        img = cv2.resize(img, (w, h))
        out.write(img)
    out.release()


if __name__ == '__main__':
    image_to_video()
