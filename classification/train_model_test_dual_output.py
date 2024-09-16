import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as FT
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
                all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))
                gt_labels = get_study_labels(study_id, df_study_labels, label_condition, LEVELS, LABELS)
                
                if gt_labels is not None:
                    dataset_item = dict()
                    dataset_item['gt_label'] = gt_labels
                    dataset_item["masks"] = np.zeros((5, 96, 96))
                    dataset_item["slices"] = list()

                    for coordinate in coordinates_dict:
                        try:
                            idx_level = LEVELS[coordinate['level']]
                            x, y = coordinate['x'], coordinate['y']

                            original_slice_path = f"{input_dir}/train_images/{study_id}/{s_id}/{coordinate['instance_number']}.dcm"
                            if original_slice_path not in dataset_item['slices']:
                                dataset_item['slices'].append(original_slice_path)

                            original_shape = pydicom.dcmread(original_slice_path).pixel_array.shape
                            x = int((x / original_shape[1]) * 96) 
                            y = int((y / original_shape[0]) * 96)

                            dataset_item['masks'][idx_level, y-3:y+3, x-3:x+3] = 1
                        except Exception as e:
                            print("Error add item", e)
                            continue

                    # ensure 3 slices
                    dataset_item["slices"] = sorted(dataset_item["slices"], key = lambda x : get_instance(x))
                    dataset_item["slices"] = dataset_item["slices"][:3]

                    # add arround
                    while len(dataset_item["slices"]) < 3:
                        gt_instance = [get_instance(x) for x in dataset_item['slices']]
                        to_get = (min(gt_instance) - 1, max(gt_instance) + 1)
                        for t in to_get:
                            for sp in all_slices_path:
                                if get_instance(sp) == t:
                                    dataset_item['slices'].append(sp)
                                    break
                            if len(dataset_item["slices"]) >= 3:
                                break

                    dataset_item["slices"] = sorted(dataset_item["slices"], key = lambda x : get_instance(x))
                    dataset.append(dataset_item)   

    return dataset


############################################################
############################################################
#          DATALOADER
############################################################
############################################################

def tensor_augmentations(tensor_image):
    angle = random.uniform(-8, 8)
    tensor_image = FT.rotate(tensor_image, angle)

    #tensor_image = FT.adjust_brightness(tensor_image, brightness_factor=random.uniform(0.9, 1.1))
    #tensor_image = FT.adjust_contrast(tensor_image, contrast_factor=random.uniform(0.9, 1.1))
    #tensor_image = FT.adjust_saturation(tensor_image, saturation_factor=random.uniform(0.9, 1.1))
    #tensor_image = FT.adjust_hue(tensor_image, hue_factor=random.uniform(-0.05, 0.05))

    noise = torch.randn(tensor_image.size()) * 0.01
    tensor_image = tensor_image + noise

    return tensor_image

def z_score_normalization(image):
    mean = image.mean()
    std = image.std()
    return (image - mean) / std

class CropClassifierDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        slices_to_use = data['slices']

        slices = np.zeros((3, 224, 224))
        for k, sp in enumerate(slices_to_use):
            p = pydicom.dcmread(sp).pixel_array.astype(np.float32)
            p = cv2.resize(p, (224, 224), interpolation=cv2.INTER_LINEAR)
            p = z_score_normalization(p)
            slices[k] = p

        slices = torch.tensor(slices).float()

        if self.is_train:
            slices = tensor_augmentations(slices)

        masks = torch.tensor(data['masks']).float()

        return slices, masks, data['gt_label']

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

def validate(model, loader, criterion_labels, criterion_mask, device):
    model.eval()
    total_loss_sum = 0
    seg_loss = 0
    cla_loss = 0

    total_log_loss = 0
    total_samples = 0
    weights = np.array([1, 2, 4])
    matrix = np.zeros((3, 3))
    with torch.no_grad():
        for images, masks_gt, labels_gt in tqdm.tqdm(loader, desc="Validation", total=len(loader)):
            images = images.to(device)
            labels_gt = labels_gt.to(device).long()
            masks_gt = masks_gt.to(device)

            pred_labels, pred_masks = model(images.to(device))
            pred_labels = pred_labels.reshape(pred_labels.size(0), 5, 3)

            loss_label = 0
            for i in range(5):
                loss_label += criterion_labels(pred_labels[:, i], labels_gt[:, i])
            loss_label /= 5
            cla_loss += loss_label

            loss_mask = criterion_mask(pred_masks, masks_gt)
            seg_loss += loss_mask

            loss = loss_label * 0.7 + loss_mask * 0.3
            total_loss_sum += loss.item()

            for i in range(5):
                log_loss_batch = calculate_log_loss(pred_labels[:, i], labels_gt[:, i], weights)
                total_log_loss += log_loss_batch.sum().item()
                total_samples += labels_gt.size(0)

                m = torch.argmax(pred_labels[:, i], dim = 1)
                matrix[labels_gt[:, i].item(), m.item()] += 1

        avg_classification_loss = total_loss_sum / len(loader)
        avg_log_loss = total_log_loss / total_samples

    avg_classification_loss = total_loss_sum / len(loader)

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
    print("Classification loss:", cla_loss / len(loader), "Segmentation loss:", seg_loss / len(loader))
    return {"loss": avg_classification_loss, "logloss": avg_log_loss}

def train_epoch(model, loader, criterion_labels, criterion_mask, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, masks_gt, labels_gt in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels_gt = labels_gt.to(device).long()
        masks_gt = masks_gt.to(device)

        pred_labels, pred_masks = model(images.to(device))
        pred_labels = pred_labels.reshape(pred_labels.size(0), 5, 3)

        loss_label = 0
        for i in range(5):
            loss_label += criterion_labels(pred_labels[:, i], labels_gt[:, i])
        loss_label /= 5

        loss_mask = criterion_mask(pred_masks, masks_gt)

        loss = loss_label * 0.7 + loss_mask * 0.3
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss

def train_submodel_test_dual_output(input_dir, model_name, crop_description, crop_condition, label_condition, crop_size, image_resize):

    # get dataset
    dataset = generate_dataset(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # get model
    model = MobileNetDualOutput()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # train with folding
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)
    criterion_labels = torch.nn.CrossEntropyLoss(weight = torch.tensor([1/7, 2/7, 4/7]).to(device))
    criterion_mask = torch.nn.BCELoss()
    best = 123456
    for epoch in range(15):
        loss_train = train_epoch(model, train_loader, criterion_labels, criterion_mask, optimizer, device)
        metrics = validate(model, valid_loader, criterion_labels, criterion_mask, device)
        print("Epoch folding", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['logloss'] < best:
            print("New best model !", model_name)
            best = metrics["logloss"]

        scheduler.step()
    return best

if __name__ == "__main__":
    best_logloss = train_submodel_test_dual_output(
                    input_dir="../../REFAIT",
                    model_name="classification_left_neural_foraminal_narrowing.pth",
                    crop_condition="Left Neural Foraminal Narrowing",
                    label_condition="Left Neural Foraminal Narrowing",
                    crop_description="Sagittal T1",
                    crop_size=(64, 96),
                    image_resize=(640, 640),
    )
