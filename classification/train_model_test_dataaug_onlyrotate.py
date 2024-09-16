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
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # x y reversed in array
        crop = (crop - crop.min()) / (crop.max() - crop.min() + 1e-9)
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_dataset(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize, slice_offset):
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
                    for coordinate in coordinates_dict:
                        try:
                            idx_level = LEVELS[coordinate['level']]
                            x, y = coordinate['x'], coordinate['y']
                            dataset_item = dict()

                            # get original slices
                            for idx, sp in enumerate(all_slices_path):
                                if get_instance(sp) == coordinate['instance_number']:
                                    idx_to_use = idx + slice_offset
                                    break

                            if idx_to_use >= 0 and idx_to_use < len(all_slices_path):

                                idx_level = LEVELS[coordinate['level']]
                                x, y = coordinate['x'], coordinate['y']
                                original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{s_id}/{coordinate['instance_number']}.dcm").pixel_array.shape
                                x = int((x / original_shape[1]) * image_resize[1]) 
                                y = int((y / original_shape[0]) * image_resize[0])

                                dataset_item['slice_path'] = all_slices_path[idx_to_use]
                                dataset_item['study_id'] = study_id
                                dataset_item['series_id'] = s_id
                                dataset_item['position'] = (x, y)
                                dataset_item['gt_label'] = gt_labels[idx_level]
                                dataset_item['crop_size'] = crop_size
                                dataset_item['image_resize'] = image_resize
                                dataset.append(dataset_item)

                        except Exception as e:
                            print("Error add item", e)
                            continue
    return dataset


############################################################
############################################################
#          DATALOADER
############################################################
############################################################

def tensor_augmentations(tensor_image):
    angle = random.uniform(-15, 15)
    tensor_image = FT.rotate(tensor_image, angle)

    tensor_image = FT.adjust_brightness(tensor_image, brightness_factor=random.uniform(0.9, 1.1))
    tensor_image = FT.adjust_contrast(tensor_image, contrast_factor=random.uniform(0.9, 1.1))
    tensor_image = FT.adjust_saturation(tensor_image, saturation_factor=random.uniform(0.9, 1.1))
    tensor_image = FT.adjust_hue(tensor_image, hue_factor=random.uniform(-0.05, 0.05))

    noise = torch.randn(tensor_image.size()) * 0.01  # Small noise
    tensor_image = tensor_image + noise

    return tensor_image


class CropClassifierDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        slices_to_use = [data['slice_path']]
        x, y = data['position']
        crops = cut_crops(slices_to_use, x, y, data['crop_size'], data['image_resize'])
        crops = torch.tensor(crops).float()
        crops = crops.expand(3, 224, 224)
        if self.is_train:
            crops = tensor_augmentations(crops)
        return crops, data['gt_label']

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

def validate(model, type, loader, criterion, device):
    model.eval()
    classification_loss_sum = 0
    total_log_loss = 0
    total_samples = 0
    weights = np.array([1, 2, 4])
    matrix = np.zeros((3, 3))
    with torch.no_grad():
        for images, labels_gt in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()

            if type == "fold":
                final_output = model.forward_fold(images.to(device), mode="valid")
            else:
                final_output = model.forward_inference(images.to(device))

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

def train_epoch(model, type, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, labels in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device).long()

        if type == "fold":
            predictions = model.forward_fold(images.to(device), mode="train")
        else:
            predictions = model.forward_inference(images.to(device))

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss


def train_submodel_test_slice(input_dir, model_name, crop_description, crop_condition, label_condition, crop_size, image_resize, slice_offset):

    # get dataset
    dataset = generate_dataset(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize, slice_offset)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # get model
    model = FoldModelClassifier(
        n_classes=3,
        n_fold_classifier=5,
        backbones=['ecaresnet26t.ra2_in1k', "seresnet50.a1_in1k", "resnet26t.ra2_in1k", "mobilenetv3_small_100.lamb_in1k", "efficientnet_b0.ra_in1k"],
        features_size=256,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # train with folding
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1/7, 2/7, 4/7]).to(device))
    best = 123456
    for epoch in range(20):
        loss_train = train_epoch(model, "fold", train_loader, criterion, optimizer, device)
        metrics = validate(model, "fold", valid_loader, criterion, device)
        print("Epoch folding", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['logloss'] < best:
            print("New best model !", model_name)
            best = metrics["logloss"]
            #torch.save(model.state_dict(), model_name)

        scheduler.step()
    return best

if __name__ == "__main__":
        
    crops_sizes = [
        (64, 96),
    ]

    for crop_size in crops_sizes:
        print("Testing:", crop_size)
        best_logloss = train_submodel_test_slice(
                        input_dir="../../REFAIT",
                        model_name="classification_left_neural_foraminal_narrowing.pth",
                        crop_condition="Left Neural Foraminal Narrowing",
                        label_condition="Left Neural Foraminal Narrowing",
                        crop_description="Sagittal T1",
                        crop_size=crop_size,
                        image_resize=(640, 640),
                        slice_offset=0
        )
        print("Best metric:", best_logloss)
        print("-" * 50)
