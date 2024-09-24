import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torch
import math

from models_axial import AxialSliceClassificationModel

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def generate_dataset(input_dir, condition, description):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4, "IDK": 5}

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Generates slices hotmap"):

        series_id = df_study_descriptions[(df_study_descriptions['study_id'] == study_id)
                                        & (df_study_descriptions['series_description'] == description)]['series_id'].to_list()
        
        for s_id in series_id:
            coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                                & (df_study_coordinates['condition'] == condition)
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            # add to dataset only if all vertebraes in series
            if len(coordinates_dict) == 5:

                instances = [c['instance_number'] for c in coordinates_dict]
                d_instance = [abs(instances[c] - instances[c+1]) for c in range(len(instances) - 1)]
                mean_instance = sum(d_instance) / len(d_instance)

                study_id = study_id
                series_id = s_id
                slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))

                indexes = list()

                # add known items
                for coordinate in coordinates_dict:
                    dataset_item = dict()
                    dataset_item['slice_path'] = f"{input_dir}/train_images/{study_id}/{s_id}/{coordinate['instance_number']}.dcm"
                    indexes.append(slices_path.index(dataset_item['slice_path']))
                    dataset_item['gt_label'] = LEVELS[coordinate['level']]
                    dataset.append(dataset_item)

                mini_index = min(indexes) - math.ceil(mean_instance)
                maxi_index = max(indexes) + math.ceil(mean_instance)

                # add unknown items
                while mini_index >= 0:
                    dataset_item = dict()
                    dataset_item['slice_path'] = slices_path[mini_index]
                    dataset_item['gt_label'] = LEVELS['IDK']
                    dataset.append(dataset_item)
                    mini_index -= 1

                while maxi_index < len(slices_path):
                    dataset_item = dict()
                    dataset_item['slice_path'] = slices_path[maxi_index]
                    dataset_item['gt_label'] = LEVELS['IDK']
                    dataset.append(dataset_item)
                    maxi_index += 1

    return dataset

def generate_test_set(input_dir, condition, description):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4, "IDK": 5}

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id[:180], desc="Generates test set"):

        series_id = df_study_descriptions[(df_study_descriptions['study_id'] == study_id)
                                        & (df_study_descriptions['series_description'] == description)]['series_id'].to_list()
        
        for s_id in series_id:
            coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                                & (df_study_coordinates['condition'] == condition)
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            # add to dataset only if all vertebraes in series
            if len(coordinates_dict) == 5:
                instances = [0] * 5
                for coordinate in coordinates_dict:
                    instances[LEVELS[coordinate['level']]] = coordinate['instance_number']
                slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))

                dataset.append({
                    "instances": instances,
                    "slices": slices_path
                })

    return dataset

class SlicePredicterDataset(Dataset):
    def __init__(self, infos):
        self.datas = infos

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        # load slice
        slice_path = data['slice_path']
        pixel_array = pydicom.dcmread(slice_path).pixel_array
        image = cv2.resize(pixel_array, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = (image - image.min()) / (image.max() - image.min())
        image = torch.tensor(image).unsqueeze(0).expand(3, 224, 224).float()

        # load label
        label = torch.tensor(data['gt_label']).long()
        return image, label

def calculate_f1(matrix):
    precision = np.zeros(matrix.shape[0])
    recall = np.zeros(matrix.shape[0])
    f1_scores = np.zeros(matrix.shape[0])

    for i in range(matrix.shape[0]):
        TP = matrix[i, i]
        FP = matrix[:, i].sum() - TP  # All predictions for class i minus TP
        FN = matrix[i, :].sum() - TP  # All actuals for class i minus TP

        if (TP + FP) > 0:
            precision[i] = TP / (TP + FP)
        else:
            precision[i] = 0.0
            
        if (TP + FN) > 0:
            recall[i] = TP / (TP + FN)
        else:
            recall[i] = 0.0
            
        if (precision[i] + recall[i]) > 0:
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_scores[i] = 0.0
            
    return np.mean(f1_scores)

def get_max_consecutive(preds, gt_indice, n):
    max_sum = -float('inf')
    max_idx = -1
    
    for i in range(len(preds) - n + 1):
        current_sum = preds[i:i + n].sum().item()
        if current_sum > max_sum:
            max_sum = current_sum
            max_idx = i
    
    max_consecutive_indices = list(range(max_idx, max_idx + n))
    is_gt_in_max = gt_indice in max_consecutive_indices
    
    return is_gt_in_max

def test_model(model, dataset_test, device):

    tops = {
        1: 0,
        3: 0,
        5: 0,
        10: 0,
    }

    for item in tqdm.tqdm(dataset_test, desc="Test model"):
        gt_instances = item['instances']
        slices_path = item['slices']

        images = np.zeros((len(slices_path), 1, 224, 224))
        for k, sp in enumerate(slices_path):
            pixel_array = pydicom.dcmread(sp).pixel_array
            image = cv2.resize(pixel_array, (224, 224), interpolation=cv2.INTER_LINEAR)
            image = (image - image.min()) / (image.max() - image.min())
            images[k, 0] = image

        images = torch.tensor(images).expand(len(slices_path), 3, 224, 224).to(device).float()
        with torch.no_grad():
            outputs = model(images)
        outputs = outputs.permute(1, 0)
        for level in range(5):
            preds = outputs[level, :]
            for top in tops: 
                tops[top] += 1 if get_max_consecutive(preds, gt_instances[level], top) else 0

    metrics = {t: tops[t] / (len(dataset_test) * 5) for t in tops}
    return metrics

def validate(model, loader, criterion, device):
    model.eval()

    valid_loss = 0
    matrix = np.zeros((6, 6))

    for images, labels in tqdm.tqdm(loader, desc="Valid"):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            _, i = torch.topk(predictions, 1)

            matrix[labels.item(), i.item()] += 1

            valid_loss += loss

    print(matrix)
    valid_loss /= len(loader)
    return valid_loss, calculate_f1(matrix)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for images, labels in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device).long()

        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)
    return epoch_loss

def train_model(input_dir, condition, description, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset = generate_dataset(input_dir, condition, description)
    dataset_test = generate_test_set(input_dir, condition, description)

    nb_valid = int(len(dataset) * 0.1)
    train_dataset = SlicePredicterDataset(dataset[nb_valid:])
    valid_dataset = SlicePredicterDataset(dataset[:nb_valid])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = AxialSliceClassificationModel()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    best_metrics = None
    best = -1
    for epoch in range(20):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device)
        loss_valid, _f1_score = validate(model, valid_loader, criterion, device)
        metrics = test_model(model, dataset_test, device)
        print("Epoch", epoch, "train_loss=", loss_train, "valid_loss=", loss_valid, "f1=", _f1_score, "metrics:", metrics)
        if _f1_score > best:
            print("New best model !", model_name)
            best = _f1_score
            scripted_model = torch.jit.script(model)
            scripted_model.save(model_name)

        print("-" * 55)
    
    print("-" * 55)
    print("-" * 55)
    print("-" * 55)
    print("DONE !")
    print("Model saved:", model_name)
    print("Best metrics:", best_metrics)
    print("-" * 55)
    print("-" * 55)
    print("-" * 55)
    return best_metrics

if __name__ == "__main__":

    train_model("../", "Right Subarticular Stenosis", "Axial T2", "model_slice_classification_axt2.ts")
