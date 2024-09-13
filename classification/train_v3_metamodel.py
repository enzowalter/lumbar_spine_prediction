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

def get_condition_index(condition):
    conds = {
                'Left Neural Foraminal Narrowing' : 0,
                'Right Neural Foraminal Narrowing': 1,
                'Spinal Canal Stenosis': 2,
                'Left Subarticular Stenosis': 3,
                'Right Subarticular Stenosis' : 4,
             }
    return conds[condition]

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

def extract_centered_square_with_padding(array, center_x, center_y, sizeX, sizeY):
    square = np.zeros((sizeX, sizeY), dtype=array.dtype)
    start_x = max(center_x - (sizeX // 2), 0)
    end_x = min(center_x + (sizeX // 2), array.shape[0])
    start_y = max(center_y - (sizeY // 2), 0)
    end_y = min(center_y + (sizeY // 2), array.shape[1])
    out_start_x = (sizeX // 2) - (center_x - start_x)
    out_end_x = out_start_x + (end_x - start_x)
    out_start_y = (sizeY // 2) - (center_y - start_y)
    out_end_y = out_start_y + (end_y - start_y)
    square[out_start_x:out_end_x, out_start_y:out_end_y] = array[start_x:end_x, start_y:end_y]
    return square

def cut_crops(slices_path, x, y, crop_size, image_resize):
    output_crops = np.zeros((len(slices_path), 224, 224))
    for k, slice_path in enumerate(slices_path):
        pixel_array = pydicom.dcmread(slice_path).pixel_array.astype(np.float32)
        pixel_array = cv2.resize(pixel_array, image_resize, interpolation=cv2.INTER_LINEAR)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-9)
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # x y reversed in array
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_dataset(input_dir, label_condition):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}
    CROP_SIZES = [(96, 128), (96, 128), (96, 128), (164, 164), (164, 164)]
    IMAGE_RESIZE = [(640, 640), (640, 640), (640, 640), (800, 800), (800, 800)]

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Generates classification dataset"):

        for level in LEVELS:
            coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                                & (df_study_coordinates['level'] == level)].to_dict('records')
            coordinates_dict = sorted(coordinates_dict, key = lambda x : get_condition_index(x['condition']))
            gt_labels = get_study_labels(study_id, df_study_labels, label_condition, LEVELS, LABELS)
            if gt_labels is None:
                continue
            if len(coordinates_dict) != 5: # all scans
                continue

            dataset_item = dict()
            dataset_item['crops'] = list()
            dataset_item['label'] = gt_labels[LEVELS[level]]

            for coordinate_index, coordinate in enumerate(coordinates_dict):
                all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{coordinate['series_id']}/*.dcm"), key = lambda x : get_instance(x))
                crop_info = dict()
                slices_path = list()

                # get original slices
                for idx, sp in enumerate(all_slices_path):
                    if get_instance(sp) == coordinate['instance_number']:
                        if idx == 0:
                            slices_path.append(all_slices_path[idx])        
                            slices_path.append(all_slices_path[idx + 1])        
                            slices_path.append(all_slices_path[idx + 2])
                        elif idx == len(all_slices_path) - 1:
                            slices_path.append(all_slices_path[idx - 2])        
                            slices_path.append(all_slices_path[idx - 1])        
                            slices_path.append(all_slices_path[idx])
                        else:
                            slices_path.append(all_slices_path[idx - 1])        
                            slices_path.append(all_slices_path[idx])        
                            slices_path.append(all_slices_path[idx + 1])

                # get crop position
                x, y = coordinate['x'], coordinate['y']
                original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{coordinate['series_id']}/{coordinate['instance_number']}.dcm").pixel_array.shape
                x = x / original_shape[1] 
                y = y / original_shape[0]

                # add to dataset
                crop_info['condition'] = coordinate['condition']
                crop_info['position'] = (x, y)
                crop_info["slices"] = slices_path
                crop_info['crop_size'] = CROP_SIZES[coordinate_index]
                crop_info['image_resize'] = IMAGE_RESIZE[coordinate_index]
                dataset_item['crops'].append(crop_info)
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
        all_crops = np.zeros((5, 3, 224, 224))
        for k, crop_info in enumerate(data['crops']):
            slices_to_use = crop_info['slices']
            x, y = crop_info['position']
            x = int(x * crop_info['image_resize'][1])
            y = int(y * crop_info['image_resize'][0])

            all_crops[k, ...] = cut_crops(slices_to_use, x, y, crop_info['crop_size'], crop_info['image_resize'])
        all_crops = torch.tensor(all_crops).float()
        return all_crops, data['label']

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
    classification_loss_sum = 0
    total_log_loss = 0
    total_samples = 0
    weights = np.array([1, 2, 4])
    matrix = np.zeros((3, 3))
    with torch.no_grad():
        for images, labels_gt in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()
            final_output = model(images.to(device))

            i = torch.argmax(final_output, dim = 1)
            matrix[labels_gt.item(), i.item()] += 1

            loss = criterion(final_output, labels_gt)
            classification_loss_sum += loss.item()

            log_loss_batch = calculate_log_loss(final_output, labels_gt, weights)
            total_log_loss += log_loss_batch.sum().item()
            total_samples += labels_gt.size(0)

        avg_classification_loss = classification_loss_sum / len(loader)
        avg_log_loss = total_log_loss / total_samples

    avg_classification_loss = classification_loss_sum / len(loader)

    weights = np.array([1, 2, 4])
    TP = np.diag(matrix)
    FP = np.sum(matrix, axis=0) - TP
    FN = np.sum(matrix, axis=1) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    weighted_precision = np.sum(weights * precision) / np.sum(weights)
    weighted_recall = np.sum(weights * recall) / np.sum(weights)
    weighted_f1_score = np.sum(weights * f1_score) / np.sum(weights)

    print(matrix)
    print("Precision:", weighted_precision, "Recall:", weighted_recall, "F1:", weighted_f1_score)
    return {"loss": avg_classification_loss, "logloss": avg_log_loss}

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, labels in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device).long()

        predictions = model(images.to(device))
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss

def get_model(model_name):
    model = FoldModelClassifier(
        n_classes=3,
        n_fold_classifier=2,
        backbones=['ecaresnet26t.ra2_in1k', "seresnet50.a1_in1k", "resnet26t.ra2_in1k", "mobilenetv3_small_100.lamb_in1k", "efficientnet_b0.ra_in1k"],
        features_size=256,
    )
    model.load_state_dict(torch.load(model_name))
    # freeze model
    for param in model.parameters():
        param.requires_grad = False
    return model


def train_metamodel(input_dir, label_condition, submodels):

    print("TRAIN METAMODEL", label_condition)

    # get dataset
    dataset = generate_dataset(input_dir, label_condition)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:])
    valid_dataset = CropClassifierDataset(dataset[:nb_valid])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    conditions = [
            'Spinal Canal Stenosis',
            'Right Neural Foraminal Narrowing',
            'Left Neural Foraminal Narrowing',
            'Left Subarticular Stenosis',
            'Right Subarticular Stenosis',
        ]
    models = list()
    for cond in conditions:
        _model = get_model(f"test_submodel_{cond}.pth")
        models.append(_model)
    """
    metamodel = CropClassifierMetaModel(submodels)

    # freeze encoders
    for m in metamodel.encoder_models:
        for param in m.parameters():
            param.requires_grad = False

    metamodel = metamodel.to(device)

    # train
    optimizer = torch.optim.AdamW(metamodel.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1/7, 2/7, 4/7]).to(device))
    best = 123456
    for epoch in range(5):
        loss_train = train_epoch(metamodel, train_loader, criterion, optimizer, device)
        metrics = validate(metamodel, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['logloss'] < best:
            print("New best model !")
            best = metrics["logloss"]
            torch.save(metamodel.state_dict(), f"metamodel_{label_condition}.pth")
        
        scheduler.step()
        print("LR=", scheduler.get_last_lr())
        print("Encoder weights", metamodel.weights_encoders)
        print("-" * 55)
    
    print("Best metrics:", best)
    print("-" * 55)
    torch.cuda.empty_cache()
    return best

if __name__ == "__main__":

    models = [
        FoldModelClassifier(
            n_classes=3,
            n_fold_classifier=2,
            backbones=['ecaresnet26t.ra2_in1k', "seresnet50.a1_in1k", "resnet26t.ra2_in1k", "mobilenetv3_small_100.lamb_in1k", "efficientnet_b0.ra_in1k"],
            features_size=256,
        )
        for _ in range(5)
    ]
    
    metamodel = CropClassifierMetaModel(models)
    metamodel.load_state_dict(torch.load("save_spinal.pth"), strict=False)

    submodels = metamodel.encoder_models

    train_metamodel(input_dir="../../REFAIT", label_condition="Spinal Canal Stenosis", submodels=submodels)