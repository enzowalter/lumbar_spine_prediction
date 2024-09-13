import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

from lumbar_pipeline import compute_pipeline

def compute_scores(conf_matrix):
    weights = np.array([1, 2, 4])
    TP = np.diag(conf_matrix)
    FP = np.sum(conf_matrix, axis=0) - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    weighted_precision = np.sum(weights * precision) / np.sum(weights)
    weighted_recall = np.sum(weights * recall) / np.sum(weights)
    weighted_f1_score = np.sum(weights * f1_score) / np.sum(weights)
    return {
        "prec": weighted_precision, 
        "rec": weighted_recall, 
        "f1": weighted_f1_score
    }

if __name__ == "__main__":
    
    description_file = "../REFAIT/train_series_descriptions.csv"
    input_images_folder = "../REFAIT/train_images/"

    #predictions: pd.DataFrame = compute_pipeline(input_images_folder, description_file, nb_studies_id=190)
    #predictions.to_csv("predictions_pipeline.csv", index=False)

    predictions = pd.read_csv("spinal_preds.csv")
    
    labels = pd.read_csv("../REFAIT/train.csv")
    LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}

    column_names = [
        #"spinal_canal_stenosis",
        #"left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing",
        #"left_subarticular_stenosis",
        #"right_subarticular_stenosis"
    ]

    matrixes = np.zeros((5, 3, 3))
    all_predictions = list()
    all_labels = list()

    for k, pred_row in predictions.iterrows():

        row_id = pred_row['row_id']
        study_id = int(row_id.split('_')[0])
        column = "_".join(row_id.split('_')[1:])
        
        disease_index = None
        for j, name in enumerate(column_names):
            if name in column:
                disease_index = j
                break
        
        if disease_index is None:
            continue

        _labels = labels[labels['study_id'] == study_id]
        try:
            label_int = LABELS[_labels[column].values[0]]
        except:
            label_int = None
        if label_int is not None:
            s, m, n = pred_row['severe'], pred_row['moderate'], pred_row['normal_mild']
            index_max = np.argmax(np.array([n, m, s]))
            matrixes[j, label_int, index_max] += 1

            all_predictions.append([n, m, s])

            l = [0, 0, 0]
            l[label_int] = 1
            all_labels.append(l)

    for model in range(5):
        print(matrixes[model])
        score = compute_scores(matrixes[model])
        print("MODEL", model)
        print(score)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    class_weights = np.array([1, 2, 4])
    class_indices = np.argmax(all_labels, axis=1)
    sample_weights = class_weights[class_indices]
    loss = log_loss(all_labels, all_predictions, sample_weight=sample_weights)

    print(f'Weighted Log Loss: {loss}')
    