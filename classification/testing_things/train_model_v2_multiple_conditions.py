import numpy as np
import pandas as pd
import torch.utils
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter, affine_transform, map_coordinates
import random
import os

import glob
import torch
import warnings
warnings.filterwarnings("ignore") # warning on lstm

from models_v2_police import *

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

def clahe_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    image = clahe.apply(np.uint8(image))
    image = image.astype(np.float32) / 255.
    return image

def cut_crops(slices_path, x, y, crop_size, image_resize):
    output_crops = np.zeros((len(slices_path), 128, 128))
    for k, slice_path in enumerate(slices_path):
        pixel_array = pydicom.dcmread(slice_path).pixel_array.astype(np.float32)
        pixel_array = cv2.resize(pixel_array, image_resize, interpolation=cv2.INTER_LINEAR)
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # x y reversed in array
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_dataset(input_dir, crop_description, crop_condition, crop_size, image_resize):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")

    datasets = list()
    nb_label_none = 0
    nb_bad_pos = 0
    
    for condition in crop_condition:
        all_coordinates = df_study_coordinates[df_study_coordinates['condition'] == condition].to_dict('records')
        dataset = list()
        for coordinate in tqdm.tqdm(all_coordinates, desc=f"Parsing coordinates for {condition}..."):

            s_id = coordinate['series_id']
            study_id = coordinate['study_id']
            level_str = coordinate['level'].lower().replace('/', '_')
            column_name = condition.lower().replace(' ', '_') + "_" + level_str
            label_str = df_study_labels[
                (df_study_labels['study_id'] == study_id)
            ][column_name].item()
            label_int = LABELS[label_str] if label_str in LABELS else None

            if label_int is None:
                nb_label_none += 1
                continue

            all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))

            original_idx = None
            for idx, sp in enumerate(all_slices_path):
                if get_instance(sp) == coordinate['instance_number']:
                    original_idx = idx
                    break

            x, y = coordinate['x'], coordinate['y']
            original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{s_id}/{coordinate['instance_number']}.dcm").pixel_array.shape
            x = int((x / original_shape[1]) * image_resize[1]) 
            y = int((y / original_shape[0]) * image_resize[0])

            dataset_item = dict()
            dataset_item['study_id'] = study_id
            dataset_item['condition'] = condition
            dataset_item['all_slices'] = all_slices_path
            dataset_item['original_index'] = original_idx
            dataset_item['series_id'] = s_id
            dataset_item['position'] = (x, y)
            dataset_item['gt_label'] = label_int
            dataset_item['crop_size'] = crop_size
            dataset_item['image_resize'] = image_resize

            dataset.append(dataset_item)
        datasets.append(dataset)

    print(nb_bad_pos, "items with bad index position", nb_label_none, "items with label none")
    return datasets


############################################################
############################################################
#          DATALOADER
############################################################
############################################################

def create_soft_labels(hard_labels, radius=2, sigma=1.0, max_value=1.0):
    soft_labels = np.zeros_like(hard_labels, dtype=float)
    for idx, label in enumerate(hard_labels):
        if label == 1:
            for i in range(-radius, radius + 1):
                if 0 <= idx + i < len(hard_labels):
                    soft_labels[idx + i] = max_value * np.exp(-0.5 * (i / sigma) ** 2)
    return soft_labels

def z_score_normalize_all_slices(scan):
    scan_flattened = scan.flatten()
    mean = np.mean(scan_flattened)
    std = np.std(scan_flattened)
    z_normalized_scan = (scan - mean) / std
    return z_normalized_scan

class CropClassifierDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        x, y = data['position']
        original_idx = data['original_index']
        all_slices_path = data['all_slices']

        index_offsets = [-2, -1, 0, 1, 2]
        weights = [0.025, 0.175, 0.6, 0.175, 0.025]

        valid_offsets = []
        valid_weights = []
        for offset, weight in zip(index_offsets, weights):
            offset_index = original_idx + offset
            if 0 <= offset_index - 2 and offset_index + 2 < len(all_slices_path):
                valid_offsets.append(offset)
                valid_weights.append(weight)

        if len(valid_offsets) == 0:
            raise ValueError(f"No valid offsets available for original index {original_idx} "
                            f"with slice count {len(all_slices_path)}.")

        valid_weights = np.array(valid_weights) / np.sum(valid_weights)
        offset = np.random.choice(valid_offsets, p=valid_weights)
        offset_index = original_idx + offset

        slices_path = [
            all_slices_path[offset_index - 2],
            all_slices_path[offset_index - 1],
            all_slices_path[offset_index],
            all_slices_path[offset_index + 1],
            all_slices_path[offset_index + 2]
        ]

        if self.is_train:
            x += random.randint(-8, 9)
            y += random.randint(-8, 9)

        good_slice_idx = slices_path.index(all_slices_path[original_idx])
        crops = cut_crops(slices_path, x, y, data['crop_size'], data['image_resize'])
        crops = z_score_normalize_all_slices(crops)
        crops = torch.tensor(crops).float()
        crops = crops.unsqueeze(1).expand(5, 3, 128, 128)

        weights_crops = np.array([0, 0, 0, 0, 0])
        weights_crops[good_slice_idx] = 1
        weights_crops = create_soft_labels(weights_crops, radius=1, sigma=0.75)
        weights_crops = torch.softmax(torch.tensor(weights_crops), dim = 0)

        return crops, data['gt_label'], torch.tensor(good_slice_idx), weights_crops

############################################################
############################################################
#          TRAINING
############################################################
############################################################

def validate(model, loader, criterion, device):
    print("-" * 50)
    model.eval()
    classification_loss_sum = 0
    all_predictions = []
    all_labels = []
    good_crops_selection = 0
    total_loss_weight = 0

    with torch.no_grad():
        for images, labels_gt, good_slice_idx, gt_weights in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()
            good_slice_idx = good_slice_idx.to(device).long()
            gt_weights = gt_weights.to(device).float()
            
            outputs, crops_weight, selected_crops = model(images.to(device), mode = "inference")
            
            weights_loss = 0
            good_crops_selection += (selected_crops == good_slice_idx).float().sum()
            weights_loss = nn.BCELoss()(crops_weight, gt_weights)
            
            total_loss_weight += weights_loss.mean().item()

            all_predictions.append(outputs.cpu())
            all_labels.append(labels_gt.cpu())
            
            loss = criterion(outputs, labels_gt)
            classification_loss_sum += loss.item()

    all_predictions = torch.cat(all_predictions, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    
    concat_loss = criterion(all_predictions, all_labels).item()
    avg_classification_loss = classification_loss_sum / len(loader)
    print("Accuracy on crop selection:", good_crops_selection / (len(loader) * model.nb_encoders))
    print("Loss on weights:", total_loss_weight / len(loader))
    return {"concat_loss": concat_loss, "mean_loss": avg_classification_loss}

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    print("-" * 50)
    if (epoch + 1) % 4 == 0:
        print("Training weights !")
    model.train()
    epoch_loss = 0
    for images, labels, good_slice_idx, gt_weights in tqdm.tqdm(loader, desc="Training", total=len(loader)):

        images = images.to(device).float()
        labels = labels.to(device).long()
        good_slice_idx = good_slice_idx.to(device).long()
        gt_weights = gt_weights.to(device).float()

        if (epoch + 1) % 4 == 0:
            outputs, crops_weights, _ = model(images, mode = "train_gate")
            weights_loss = 0
        else:
            outputs, crops_weights, _ = model(images, mode = "train")
            weights_loss = nn.BCELoss()(crops_weights, gt_weights)
        
        classification_loss = criterion(outputs, labels)
        total_loss = classification_loss + weights_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item() / len(loader)

    if (epoch + 1) % 4 == 0:
        print("Trained weights:", model.weights)

    return epoch_loss

def train_submodel(input_dir, model_name, crop_description, crop_condition, crop_size, image_resize):

    # get dataset
    datasets = generate_dataset(input_dir, crop_description, crop_condition, crop_size, image_resize)

    train_datasets = []
    valid_datasets = []

    for dataset in datasets:
        nb_valid = int(len(dataset) * 0.1)
        valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False)
        valid_datasets.append(valid_dataset)
        train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True)
        train_datasets.append(train_dataset)
    
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=2)

    # backbones = ['cspresnet50.ra_in1k', 'ese_vovnet39b.ra_in1k', 'dm_nfnet_f0.dm_in1k', 'ecaresnet26t.ra2_in1k', 'resnet34.a1_in1k']
    # backbones = ['focalnet_small_lrf.ms_in1k', 'convnextv2_tiny.fcmae_ft_in1k', 'dm_nfnet_f0.dm_in1k', 'cspresnet50.ra_in1k', 'ecaresnet26t.ra2_in1k']
    # backbones = ['focalnet_small_lrf.ms_in1k', 'convnext_base.fb_in22k_ft_in1k', 'dm_nfnet_f0.dm_in1k', 'ecaresnet26t.ra2_in1k', 'resnet34.a1_in1k']
    backbones = ['focalnet_small_lrf.ms_in1k', 'convnext_base.fb_in22k_ft_in1k', 'dm_nfnet_f0.dm_in1k']

    model = REM(
        n_classes=3,
        # n_classifiers=3,
        unification_size=768,
        backbones=backbones,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1, 2, 4]).float().to(device))
    best = 123456

    for epoch in range(30):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['concat_loss'] < best:
            print("New best model !")
            best = metrics["concat_loss"]
            # torch.save(model.state_dict(), model_name)
        print('-' * 50)
        scheduler.step()
        print(scheduler.get_last_lr())
    return best

if __name__ == "__main__":

    best = train_submodel(
        input_dir="../",
        model_name="classification_sagittal.pth",
        crop_condition=["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing"],
        crop_description=["Sagittal T1"],
        crop_size=(80, 120),
        image_resize=(640, 640),
    )

    # conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Spinal Canal Stenosis", "Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    # descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR", "Axial T2", "Axial T2"]
    # crop_sizes = [(80, 120), (80, 120), (80, 120), (128, 128), (128, 128)]
    # out_name = ["classification_st1_left.pth", "classification_st1_right.pth", "classification_st2.pth", "classification_ax_left.pth", "classification_ax_right.pth"]

    # metrics = dict()
    # for cond, desc, csize, out in zip(conditions, descriptions, crop_sizes, out_name):
    #     if not "Left Neural Foraminal Narrowing" in cond:
    #         continue

    #     print('-' * 50)
    #     print('-' * 50)
    #     print("Training:", cond)
    #     print('-' * 50)
    #     print('-' * 50)
    #     best = train_submodel(
    #         input_dir="../",
    #         model_name=out,
    #         crop_condition=cond,
    #         crop_description=desc,
    #         crop_size=csize,
    #         image_resize=(640, 640),
    #     )
    #     metrics[cond] = best

    # print("Done !")
    # print(metrics)