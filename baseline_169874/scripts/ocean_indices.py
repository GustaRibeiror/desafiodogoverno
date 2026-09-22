"""Parse NOAA files; monthly lags use observation month, not release vintage."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

FILES = {'soi': 'soi.txt', 'nino12': 'nina1.anom.data.txt',
         'nino3': 'nina3.anom.data.txt', 'nino4': 'nina4.anom.data.txt'}

def read_indices():
    output = {}
    for name, filename in FILES.items():
        text = (Path(__file__).parent / 'data' / 'indices' / filename).read_text()
        if name == 'soi':
            text = text.split('STANDARDIZED    DATA')[1]
        values = {}
        for line in text.splitlines():
            tokens = re.findall(r'-?\d+(?:\.\d+)?', line)
            if len(tokens) != 13 or not re.match(r'^\s*\d{4}', line):
                continue
            year = int(tokens[0])
            for month, value in enumerate(tokens[1:], 1):
                value = float(value)
                key = pd.Timestamp(year, month, 1)
                if key in values:
                    raise ValueError(f'Duplicate {name} {key}')
                values[key] = value if value > -90 else np.nan
        output[name] = pd.Series(values).sort_index()
        if not values:
            raise ValueError(f'Empty {filename}')
    return output

def add_indices(frame, dates, npoints, series, interactions=False):
    for name, data in series.items():
        for lag in (1, 2, 3):
            source = dates - pd.DateOffset(months=lag)
            assert (source < dates).all()
            values = data.reindex(source).to_numpy(dtype='float32')
            if not np.isfinite(values).all():
                raise ValueError(f'Missing historical values: {name} lag {lag}')
            frame[f'{name}_lag{lag}'] = np.repeat(values, npoints)
            if interactions and lag == 1:
                repeated = np.repeat(values, npoints)
                frame[f'{name}_lat'] = repeated * frame['lat'].to_numpy()
                frame[f'{name}_lon'] = repeated * frame['lon'].to_numpy()
                frame[f'{name}_month_sin'] = repeated * frame['month_sin'].to_numpy()
                frame[f'{name}_month_cos'] = repeated * frame['month_cos'].to_numpy()
    return frame
