import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import os

import glob
import torch
import warnings
warnings.filterwarnings("ignore") # warning on lstm

from models import *

############################################################
############################################################
#          GEN DATASET UTILS
############################################################
############################################################

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

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

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_dataset(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Generates classification dataset"):

        series_id = df_study_descriptions[(df_study_descriptions['study_id'] == study_id)
                                        & (df_study_descriptions['series_description'] == crop_description)]['series_id'].to_list()
        
        for s_id in series_id:
            coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                                & (df_study_coordinates['condition'] == crop_condition)
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            # add to dataset only if all vertebraes in gt
            if len(coordinates_dict) == len(LEVELS):
                dataset_item = dict()
                all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))
                dataset_item['all_slices'] = all_slices_path

                coordinates_dict = sorted(coordinates_dict, key = lambda x : LEVELS[x['level']])

                positions = list()
                slices_index = list()
                gt_labels = get_study_labels(study_id, df_study_labels, label_condition, LEVELS, LABELS)
                for coordinate in coordinates_dict:

                    x, y, = coordinate['x'], coordinate['y']
                    instance_number = coordinate['instance_number']
                    for k, sp in enumerate(all_slices_path):
                        if get_instance(sp) == instance_number:
                            original_shape = pydicom.dcmread(sp).pixel_array.shape
                            slice_index = k
                            break

                    x = (x / original_shape[1]) * 224
                    y = (y / original_shape[0]) * 224
                
                    positions.append([x, y])
                    slices_index.append(slice_index)

                dataset_item['labels'] = gt_labels
                dataset_item['positions'] = positions
                dataset_item['slices_index'] = slices_index
                dataset.append(dataset_item)
    return dataset


############################################################
############################################################
#          DATALOADER
############################################################
############################################################

class CropClassifierDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        slices_to_use = data['all_slices']
        input_images = np.zeros((len(slices_to_use), 224, 224))
        for i in range(len(slices_to_use)):
            p = pydicom.dcmread(slices_to_use[i]).pixel_array.astype(np.float32)
            p = cv2.resize(p, (224, 224))
            p = (p - p.min()) / (p.max() - p.min())
            input_images[i] = p
        
        input_images = torch.tensor(input_images).float().unsqueeze(0)
        input_images = input_images.expand(3, len(slices_to_use), 224, 224)

        return input_images, data['labels'], np.array(data['positions']), np.array(data['slices_index'])

############################################################
############################################################
#          TRAINING
############################################################
############################################################

def calculate_log_loss(predictions, labels, weights, epsilon=1e-9):
    probabilities = torch.clamp(F.softmax(predictions, dim=1), min=epsilon, max=1 - epsilon)
    one_hot_labels = F.one_hot(labels, num_classes=3).float()
    log_loss_batch = -torch.sum(one_hot_labels * torch.log(probabilities), dim=1)
    weights_per_sample = weights[labels.cpu().numpy()]
    weighted_log_loss_batch = log_loss_batch * torch.tensor(weights_per_sample, device=labels.device)
    return weighted_log_loss_batch

def validate(model, loader, criterion, device):
    model.eval()
    slice_loss = nn.MSELoss()
    coord_loss = nn.MSELoss()
    severity_loss = torch.nn.CrossEntropyLoss(weight = torch.tensor([1/7, 2/7, 4/7]).to(device))

    slice_total = 0
    coord_total = 0
    severity_total = 0
    
    with torch.no_grad():
        for images, labels, positions, slices_index in tqdm.tqdm(loader, desc="Valid"):
            images = images.to(device).float()
            labels = labels.to(device).long()
            positions = positions.to(device).float()
            slices_index = slices_index.to(device).float()

            slices, coords, severities = model(images.to(device))

            total_slice_loss = sum([slice_loss(slices[i], slices_index[:, i]) for i in range(5)]) / 5
            total_coord_loss = sum([coord_loss(coords[i], positions[:, i]) for i in range(5)]) / 5
            total_severity_loss = sum([severity_loss(severities[i], labels[:, i]) for i in range(5)])

            slice_total += total_slice_loss
            coord_total += total_coord_loss
            severity_total += total_severity_loss
            
    slice_total /= len(loader)
    coord_total /= len(loader)
    severity_total /= len(loader)
    print("Loss slice", slice_total, "loss coord", coord_total, "loss severity", severity_total)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0

    slice_loss = nn.MSELoss()
    coord_loss = nn.MSELoss()
    severity_loss = torch.nn.CrossEntropyLoss(weight = torch.tensor([1/7, 2/7, 4/7]).to(device))

    for images, labels, positions, slices_index in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()
        images = images.to(device).float()
        labels = labels.to(device).long()
        positions = positions.to(device).float()
        slices_index = slices_index.to(device).float()

        slices, coords, severities = model(images.to(device))

        total_slice_loss = sum([slice_loss(slices[i], slices_index[:, i]) for i in range(5)]) / 5
        total_coord_loss = sum([coord_loss(coords[i], positions[:, i]) for i in range(5)]) / 5
        total_severity_loss = sum([severity_loss(severities[i], labels[:, i]) for i in range(5)])

        loss = total_slice_loss + total_coord_loss + total_severity_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss


def train_submodel(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize):

    model_name = "__tmp__.pth"

    # get dataset
    dataset = generate_dataset(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:])
    valid_dataset = CropClassifierDataset(dataset[:nb_valid])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # get model
    """
    model = FoldModelClassifier(
        n_classes=3,
        n_fold_classifier=2,
        backbones=['ecaresnet26t.ra2_in1k', "seresnet50.a1_in1k", "resnet26t.ra2_in1k", "mobilenetv3_small_100.lamb_in1k", "efficientnet_b0.ra_in1k"],
        features_size=256,
    )
    """
    
    model = LumbarClassifierMultiLevel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # train
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1/7, 2/7, 4/7]).to(device))

    for epoch in range(10):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device)
        print("Epoch", epoch, "train_loss=", loss_train)
        validate(model, valid_loader, criterion, device)

def train_all_submodels(input_dir, label_condition):
    models = list()
    conditions = [
            "Left Neural Foraminal Narrowing", 
            "Right Neural Foraminal Narrowing", 
            "Spinal Canal Stenosis", 
            "Left Subarticular Stenosis", 
            "Right Subarticular Stenosis"
        ]
    descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR", "Axial T2", "Axial T2"]
    crop_sizes = [(96, 128), (96, 128), (96, 128), (164, 164), (164, 164)]
    image_resizes = [(640, 640), (640, 640), (640, 640), (800, 800), (800, 800)]
    for cc, cd, cs, ir in zip(conditions, descriptions, crop_sizes, image_resizes):
        print("-" * 50)
        print("-" * 50)
        print("TRAINING", cc, cd)
        print("-" * 50)
        print("-" * 50)
        m = train_submodel(
                    input_dir=input_dir,
                    crop_condition=cc,
                    label_condition=label_condition,
                    crop_description=cd,
                    crop_size=cs,
                    image_resize=ir,
        )
        models.append(m)
    return models

if __name__ == "__main__":
    submodels = train_all_submodels(input_dir="../../REFAIT", label_condition="Left Neural Foraminal Narrowing")
    for model in submodels:
        torch.save(submodels[model].state_dict(), f"test_submodel_{model}.pth")