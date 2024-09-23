import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torch
import pickle
import pathlib

from models import SagittalSliceSelecterModel
from convert_to_ts import convert_to_ts

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def generate_scores(num_images, index):
    scores = np.zeros(num_images)
    for i in range(num_images):
        distance = abs(i - index)
        score = max(0, 1 - (distance ** 2) * 0.1)
        scores[i] = score
    return scores

def generate_dataset(input_dir, condition, description):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}

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
            if len(coordinates_dict) == len(LEVELS):
                dataset_item = dict()
                dataset_item['study_id'] = study_id
                dataset_item['series_id'] = s_id
                dataset_item['slices_path'] = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))

                if len(dataset_item['slices_path']) > 50:
                    continue # gpu explodes
                
                dataset_item['nb_slices'] = len(dataset_item['slices_path'])
                dataset_item['labels'] = np.zeros((len(LEVELS), len(dataset_item['slices_path'])))
                dataset_item['gt_indices'] = np.zeros(len(LEVELS))

                # Preprocess images
                output_folder = "_tmp2_"
                pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)
                images_preprocessed = np.zeros((len(dataset_item['slices_path']), 1, 224, 224))
                for k, path in enumerate(dataset_item['slices_path']):
                    im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                                (224, 224),
                                                interpolation=cv2.INTER_LINEAR)
                    im = (im - im.min()) / (im.max() - im.min() + 1e-9)
                    images_preprocessed[k, 0, ...] = im

                preprocess_name = f"{output_folder}/{study_id}_{s_id}.npy"
                np.save(preprocess_name, images_preprocessed)
                dataset_item["preprocess_name"] = preprocess_name

                # Fill GT
                for coordinate in coordinates_dict:
                    idx_level = LEVELS[coordinate['level']]
                    instance = coordinate['instance_number']
                    instance_index = None
                    for i, slice_path in enumerate(dataset_item['slices_path']):
                        if get_instance(slice_path) == instance:
                            instance_index = i
                            break
                    if instance_index is None:
                        raise Exception("Error in GT !")
                    dataset_item['gt_indices'][idx_level] = instance_index
                    dataset_item['labels'][idx_level] = generate_scores(dataset_item['nb_slices'], instance_index)

                dataset.append(dataset_item)
    return dataset

class SlicePredicterDataset(Dataset):
    def __init__(self, infos):
        self.datas = infos

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        labels = data['labels']
        nb_slices = data['nb_slices']
        images = np.load(data['preprocess_name'])

        images = torch.tensor(images).expand(nb_slices, 3, 224, 224).float()
        labels = torch.tensor(labels).float()
        return images, labels, np.array(data['gt_indices'])  

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

def validate(model, loader, criterion, device):
    model.eval()

    valid_loss = 0
    tops = {
        1: 0,
        3: 0,
        5: 0,
        10: 0,
    }
    for images, labels, gt_indices in tqdm.tqdm(loader, desc="Valid"):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            for level in range(5):
                preds = predictions[:, level]
                gt_indice = gt_indices[:, level].item()

                for top in tops: 
                    tops[top] += 1 if get_max_consecutive(preds.squeeze(0), gt_indice, top) else 0

            valid_loss += loss.item() / len(loader)
    return valid_loss, {t: tops[t] / (len(loader) * 5) for t in tops}

def train_epoch(model, loader, criterion, optimizer, accumulation_step, device):
    model.train()

    epoch_loss = 0
    optimizer.zero_grad()
    for step, (images, labels, gt_indices) in tqdm.tqdm(enumerate(loader), desc="Training", total=len(loader)):
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)

        #loss = criterion(predictions, labels)
        loss = criterion(predictions, labels) / accumulation_step
        loss.backward()
        epoch_loss += loss.item() / len(loader)

        if (step + 1) % accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    if (step + 1) % accumulation_step != 0:
        optimizer.step()
        optimizer.zero_grad()

    return epoch_loss

def train_model(input_dir, condition, description, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = generate_dataset(input_dir, condition, description)

    nb_valid = int(len(dataset) * 0.1)
    train_dataset = SlicePredicterDataset(dataset[nb_valid:])
    valid_dataset = SlicePredicterDataset(dataset[:nb_valid])

    # batch size = 1 because no padding on sequences
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = SagittalSliceSelecterModel()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss()

    best_metrics = None
    best = -1
    for epoch in range(10):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, 32, device)
        loss_valid, instance_accuracy = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "valid_loss=", loss_valid, "instance_accuracy=", instance_accuracy)
        if instance_accuracy[3] > best:
            print("New best model !", model_name)
            best = instance_accuracy[3]
            best_metrics = instance_accuracy
            with open(model_name, 'wb') as f:
                pickle.dump(model, f)

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
    train_model("../", "Right Subarticular Stenosis", "Axial T2", "model_slice_selection_axt2_right_left.pkl")
    convert_to_ts("model_slice_selection_axt2_right.pkl", "model_slice_selection_axt2_right_scripted.ts")
    
    train_model("../", "Left Subarticular Stenosis", "Axial T2", "model_slice_selection_axt2_left.pkl")
    convert_to_ts("model_slice_selection_axt2_left.pkl", "model_slice_selection_axt2_left_scripted.ts")

    # train_model("../", "Left Neural Foraminal Narrowing", "Sagittal T1", "model_slice_selection_st1_left.pkl")

    # train_model("../", "Right Neural Foraminal Narrowing", "Sagittal T1", "model_slice_selection_st1_right.pkl")

    # train_model("../", "Spinal Canal Stenosis", "Sagittal T2/STIR", "model_slice_selection_st2.pkl")
