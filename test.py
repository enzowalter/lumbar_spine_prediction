import pandas as pd
import numpy as np
import torch

DF_LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
DF_LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}

def get_study_labels(study_id, df_study_labels, condition, levels, labels):
    label_name = condition.lower().replace(" ", "_")
    all_labels = df_study_labels[df_study_labels['study_id'] == study_id]
    columns_of_interest = [col for col in all_labels.columns if label_name in col]
    filtered_labels = all_labels[columns_of_interest].to_dict('records')[0]
    final_labels = np.zeros(len(levels))

    for level in levels:
        level_name = level.lower().replace('/', '_')
        for item in filtered_labels:
            if level_name in item:
                try:
                    final_labels[levels[level]] = labels[filtered_labels[item]]
                except Exception as e: # sometimes labels are NaN
                    print("Error label", e)
                    return None

    return final_labels

df_valid = pd.read_csv("classification/validate.csv").to_dict("records")
df_pipeline = pd.read_csv("spinal_preds.csv").to_dict("records")
df_labels = pd.read_csv("../REFAIT/train.csv")

LEVELS = {'l1_l2':0, 'l2_l3':1, 'l3_l4':2, 'l4_l5':3, 'l5_s1':4}

class Prediction:
    study_id:int
    level: int
    normal:float
    moderate: float
    severe:float
    gt_label: int

    def get_pred(self):
        return [self.normal, self.moderate, self.severe]

    def __str__(self):
        return (f"Prediction(study_id={self.study_id}, "
                f"level={self.level}, "
                f"normal={self.normal}, "
                f"moderate={self.moderate}, "
                f"severe={self.severe}), "
                f"label={self.gt_label}")

    def __eq__(self, other):
        if not isinstance(other, Prediction):
            return False
        return (self.study_id == other.study_id and
                self.level == other.level and
                self.normal == other.normal and
                self.moderate == other.moderate and
                self.severe == other.severe)

predictions_valid = list()
for item in df_valid:
    p = Prediction()
    p.study_id = item['study_id']
    p.level = item['level']
    p.normal = item['normal']
    p.moderate = item['moderate']
    p.severe = item['severe']
    p.gt_label = get_study_labels(p.study_id, df_labels, "Spinal Canal Stenosis", DF_LEVELS, DF_LABELS)[p.level]
    predictions_valid.append(p)

predictions_pipeline = list()
for item in df_pipeline:
    p = Prediction()
    p.normal = item['normal_mild']
    p.moderate = item['moderate']
    p.severe = item['severe']

    study_id = int(item['row_id'].split('_')[0])
    level = LEVELS["_".join(item['row_id'].split('_')[-2:])]

    p.level = level
    p.study_id = study_id
    p.gt_label = get_study_labels(p.study_id, df_labels, "Spinal Canal Stenosis", DF_LEVELS, DF_LABELS)[p.level]
    predictions_pipeline.append(p)

PREDS = list()
LABELS = list()
for p in predictions_valid:
    PREDS.append(p.get_pred())
    LABELS.append(p.gt_label)

tpreds = torch.tensor(PREDS).float()
tlabels = torch.tensor(LABELS).long()

criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1/7, 2/7, 4/7]).float())
loss = criterion(tpreds, tlabels)
print("Loss concat:", loss.item())

loss = 0
for p in predictions_valid:
    tpreds = torch.tensor(p.get_pred()).float()
    tlabels = torch.tensor(p.gt_label).long()
    loss += torch.nn.CrossEntropyLoss(weight=torch.tensor([1/7, 2/7, 4/7]).float())(tpreds, tlabels)
loss /= len(predictions_valid)
print("Loss mean", loss)