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
import shutil
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter, affine_transform, map_coordinates
import random
import concurrent.futures

from models import *

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def create_soft_labels(hard_labels, radius=2, sigma=1.0, max_value=1.0):
    soft_labels = np.zeros_like(hard_labels, dtype=float)
    for idx, label in enumerate(hard_labels):
        if label == 1:
            for i in range(-radius, radius + 1):
                if 0 <= idx + i < len(hard_labels):
                    soft_labels[idx + i] = max_value * np.exp(-0.5 * (i / sigma) ** 2)
    return soft_labels

def generate_scores(num_images, index):
    scores = np.zeros(num_images)
    scores[index] = 1
    return create_soft_labels(scores)

import pydicom
import numpy as np
from scipy.ndimage import zoom
import cv2

def load_dicom_series(dicom_files):
    dicom_files = sorted(dicom_files, key = lambda x : get_instance(x))
    slices = [pydicom.dcmread(f) for f in dicom_files]
    images = np.stack([s.pixel_array for s in slices], axis=-1)  # shape: [height, width, num_slices]
    return images

def resample_volume(volume, slice_thickness, pixel_spacing, new_spacing=[1.0, 1.0, 1.0]):
    current_spacing = [slice_thickness] + pixel_spacing
    resize_factor = np.array(current_spacing) / np.array(new_spacing)
    resampled_volume = zoom(volume, resize_factor, mode='nearest')
    return resampled_volume

def resize_slices_to_224(volume):
    num_slices = volume.shape[-1]
    resized_slices = []
    for i in range(num_slices):
        slice_ = volume[:, :, i]
        resized_slice = cv2.resize(slice_, (224, 224), interpolation=cv2.INTER_LINEAR)
        resized_slices.append(resized_slice)
    resized_volume = np.stack(resized_slices, axis=-1)
    return resized_volume

def z_score_normalize_all_slices(scan):
    scan_flattened = scan.flatten()
    mean = np.mean(scan_flattened)
    std = np.std(scan_flattened)
    z_normalized_scan = (scan - mean) / std
    return z_normalized_scan

def generate_dataset(input_dir, condition, description, scores_fct):

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

                dataset_item['nb_slices'] = len(dataset_item['slices_path'])
                dataset_item['labels'] = np.zeros((len(LEVELS), len(dataset_item['slices_path'])))
                dataset_item['gt_indices'] = np.zeros(len(LEVELS))

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
                    dataset_item['labels'][idx_level] = scores_fct(dataset_item['nb_slices'], instance_index)

                dataset.append(dataset_item)
    return dataset

class SlicePredicterDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        labels = data['labels']
        slices = data['slices_path']
        volume = load_dicom_series(slices)
        resized_volume = resize_slices_to_224(volume)
        normalised_volume = z_score_normalize_all_slices(resized_volume)

        normalised_volume = normalised_volume.transpose(2, 0, 1)
        images = torch.tensor(normalised_volume).unsqueeze(1).float()
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

        # predictions = model(images)
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = torch.nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def train_model(input_dir, condition, description, scores_fct, step):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = generate_dataset(input_dir, condition, description, scores_fct)

    # select validation and training set
    nb_valid = len(dataset) // 4
    valid_indices = list(range(step * nb_valid, (step + 1) * nb_valid))
    all_indices = list(range(len(dataset)))
    train_indices = list(set(all_indices) - set(valid_indices))
    
    train_dataset = SlicePredicterDataset([dataset[i] for i in train_indices], is_train=True)
    valid_dataset = SlicePredicterDataset([dataset[i] for i in valid_indices])

    # batch size = 1 because no padding on sequences
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=3, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=3)

    model = SliceSelecterModelTransformer()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = FocalLoss(alpha=1, gamma=2)

    best_metrics = None

    current_name = f"trained_models/{condition.lower().replace(' ', '_')}_step_{step}.pth"

    best = -1
    for epoch in range(10):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, 8, device)
        loss_valid, instance_accuracy = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "step", step, "train_loss=", loss_train, "valid_loss=", loss_valid, "instance_accuracy=", instance_accuracy)
        if instance_accuracy[3] > best:
            print("New best model !")
            best = instance_accuracy[3]
            best_metrics = instance_accuracy
            torch.save(model.state_dict(), current_name)
        print("-" * 55)

    return best_metrics

if __name__ == "__main__":

    conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Spinal Canal Stenosis", "Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR", "Axial T2", "Axial T2"]
    out_name = ["slice_selector_st1_left_bs16.ts", "slice_selector_st1_right_bs16.ts", "slice_selector_st2_bs16.ts", "slice_selector_ax_left_bs16.ts", "slice_selector_ax_right_bs16.ts"]
    scores_fct = [generate_scores, generate_scores, generate_scores, generate_scores, generate_scores]

    metrics = dict()
    def train_and_collect_metrics(cond, desc, fct, step):
        print('-' * 50)
        print(f"Training: {cond} cross step {step}")
        best = train_model(
            input_dir="../",
            condition=cond,
            description=desc,
            scores_fct=fct,
            step=step,
        )
        return step, best

    for cond, desc, fct, out_name in zip(conditions, descriptions, scores_fct, out_name):
        metrics[cond] = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_step = {executor.submit(train_and_collect_metrics, cond, desc, fct, step): step for step in range(4)}
            
            for future in concurrent.futures.as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    step_num, best = future.result()
                    metrics[cond].append((step_num, best))
                    print(f"Metrics for {cond}, step {step_num}: {best}")
                except Exception as exc:
                    print(f"{cond}, step {step} generated an exception: {exc}")

    print("Done!")
    print("All metrics:", metrics)
