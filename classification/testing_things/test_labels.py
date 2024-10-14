import pandas as pd
import torch

conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Spinal Canal Stenosis", "Left Subarticular Stenosis", "Right Subarticular Stenosis"]
LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}
LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}

df = pd.read_csv('../train.csv')


for COND in conditions:

    print('-' * 50)
    print('-' * 50)
    print(COND)
    print('-' * 50)
    print('-' * 50)

    values = {
        'l1_l2': [0, 0, 0],
        'l2_l3': [0, 0, 0],
        'l3_l4': [0, 0, 0],
        'l4_l5': [0, 0, 0],
        'l5_s1': [0, 0, 0],
    }

    _cond = COND.lower().replace(' ', '_')
    for k, row in df.iterrows():

        for level in LEVELS:
            _level = level.lower().replace('/', '_')
            col = f"{_cond}_{_level}"
            v = LABELS[row[col]] if row[col] in LABELS else None
            if v is not None:
                values[_level][v] += 1

    print(values)

    ns, ms, ss = 0, 0, 0

    for level in LEVELS:
        _level = level.lower().replace('/', '_')

        n = values[_level][0] / sum(values[_level])
        m = values[_level][1] / sum(values[_level])
        s = values[_level][2] / sum(values[_level])

        ns += n / 5
        ms += m / 5
        ss += s / 5

        print(_level, 'normal:', n, 'moderate:', m, 'severe:', s)

    print("Overall:", ns, ms, ss)

    na = []
    ma = []
    sa = []

    for level in LEVELS:
        _level = level.lower().replace('/', '_')

        na.append(values[_level][0])
        ma.append(values[_level][1])
        sa.append(values[_level][2])

    print(na, ma, sa)
    tt = torch.tensor([sum(na), sum(ma), sum(sa)]).float()
    print(tt)
    print(torch.softmax(tt, dim = 0))

    import numpy as np

    # Existing counts
    normal_count = sum(na)
    moderate_count = sum(ma)
    severe_count = sum(sa)

    # Desired ratio is 1:2:4 (normal : moderate : severe)
    # Severe is 470, so moderate should be 2x470, and normal should be 1x470.

    target_severe_count = severe_count
    target_moderate_count = min(moderate_count, 2 * target_severe_count)  # Around 940, based on 2x severe
    target_normal_count = min(normal_count, 4 * target_severe_count)       # Around 1880, based on 4x severe

    # Print new target values
    print("Target Normal Count:", target_normal_count)
    print("Target Moderate Count:", target_moderate_count)
    print("Target Severe Count:", target_severe_count)