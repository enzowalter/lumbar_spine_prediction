import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torch
from scipy.ndimage import label, center_of_mass
import warnings
warnings.filterwarnings("ignore") # warning on lstm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet34(weights="DEFAULT")

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

class Upsampler(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(0.021),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.LeakyReLU(0.021),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.LeakyReLU(0.021),
            nn.Conv2d(in_channels // 8, in_channels // 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
        )
        self.lastconv = nn.Conv2d(in_channels // 8, 5, 1, 1, 0)

    def forward(self, x):
        x = self.inconv(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = F.interpolate(x, size = (384, 384), mode="bilinear", align_corners=True)
        x = self.lastconv(x)
        return x

class PositionHead(nn.Module):
    def __init__(self, features_size):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(features_size, features_size // 2),
            nn.ReLU(),
            nn.Linear(features_size // 2, 10),
        )

    def forward(self, x):
        x = self.pooler(x)
        x = x.flatten(start_dim = 1)
        x = self.fc(x)
        return x
        
class LumbarSegmentationModelSagittal(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.upper_head = Upsampler(in_channels = 512)
        self.position_head = PositionHead(features_size = 512)

    def forward(self, images):
        encoded = self.image_encoder(images)
        mask = self.upper_head(encoded)
        positions = self.position_head(encoded)
        return mask.sigmoid(), positions.sigmoid()
    

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def generate_dataset(input_dir, condition, description):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    MASK_SIZE = (384, 384)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Generates segmentation masks"):

        series_id = df_study_descriptions[(df_study_descriptions['study_id'] == study_id)
                                        & (df_study_descriptions['series_description'] == description)]['series_id'].to_list()
        
        for s_id in series_id:
            coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                                & (df_study_coordinates['condition'] == condition)
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            if len(coordinates_dict) != 5:
                continue

            all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))

            # get slices to use
            original_idx = None
            for coordinate in coordinates_dict:
                if coordinate['level'] != "L3/L4":
                    continue
                for idx, sp in enumerate(all_slices_path):
                    if get_instance(sp) == coordinate['instance_number']:
                        original_idx = idx
                        break

            if original_idx > 2 and original_idx < len(all_slices_path) - 3:
                
                dataset_item = dict()
                dataset_item['all_slices'] = all_slices_path
                dataset_item['original_idx'] = original_idx
                dataset_item['output_mask'] = np.zeros((5, 384, 384))
                dataset_item['labels_positions'] = [[None, None]] * len(LEVELS)
                dataset_item['gt_positions'] = [[None, None]] * len(LEVELS)

                for coordinate in coordinates_dict:
                    idx_level = LEVELS[coordinate['level']]
                    instance = coordinate['instance_number']
                    x, y = coordinate['x'], coordinate['y']
                    original_slice = f"{input_dir}/train_images/{study_id}/{s_id}/{instance}.dcm"
                    original_shape = pydicom.dcmread(original_slice).pixel_array.shape
                    x = int((x / original_shape[1]) * MASK_SIZE[1])
                    y = int((y /original_shape[0]) * MASK_SIZE[0])
                    dataset_item['gt_positions'][idx_level] = [x, y]
                    dataset_item['labels_positions'][idx_level] = [x / MASK_SIZE[1], y / MASK_SIZE[0]]
                    dataset_item['output_mask'][idx_level][y-17:y+17,x-17:x+17] = 1
                dataset.append(dataset_item)
    return dataset


def load_dicom_series(dicom_files):
    dicom_files = sorted(dicom_files, key = lambda x : get_instance(x))
    slices = [pydicom.dcmread(f) for f in dicom_files]
    images = np.stack([s.pixel_array for s in slices], axis=-1)  # shape: [height, width, num_slices]
    return images

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

class SegmentationDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        
        original_idx = data['original_idx']
        all_slices_path = data['all_slices']

        index_offets = [-2, -1, 0, 1, 2]
        weights = [0.025, 0.175, 0.6, 0.175, 0.025]

        offset = np.random.choice(index_offets, p=weights)
        original_idx += offset

        slices_path = list()

        if original_idx == 0:
            slices_path.append(all_slices_path[original_idx])        
            slices_path.append(all_slices_path[original_idx + 1])        
            slices_path.append(all_slices_path[original_idx + 2])
        elif original_idx == len(all_slices_path) - 1:
            slices_path.append(all_slices_path[original_idx - 2])        
            slices_path.append(all_slices_path[original_idx - 1])        
            slices_path.append(all_slices_path[original_idx])
        else:
            slices_path.append(all_slices_path[original_idx - 1])        
            slices_path.append(all_slices_path[original_idx])        
            slices_path.append(all_slices_path[original_idx + 1])
    
        volume = load_dicom_series(slices_path)
        resized_volume = resize_slices_to_224(volume)
        normalised_volume = z_score_normalize_all_slices(resized_volume)

        normalised_volume = normalised_volume.transpose(2, 0, 1)
        images = torch.tensor(normalised_volume).float()
        return images, torch.tensor(data['output_mask']).float(), torch.tensor(data['gt_positions']), torch.tensor(data['labels_positions']).float()


def find_center_of_largest_activation(mask):
    mask = (mask > 0.5).float().detach().cpu().numpy()
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return None
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0
    largest_component_center = center_of_mass(labeled_mask == np.argmax(sizes))
    center_coords = tuple(map(int, largest_component_center))
    return (center_coords[1], center_coords[0]) # x, y

def validate(model, loader, criterion_mask, criterion_position, device, threshold=0.5):
    model.eval()
    total_mask_loss = 0
    total_pos_loss = 0
    total_metrics = {
        'loss': 0,
        "pos_loss": 0
    }

    metrics_position = [{
        'mdx': list(),
        'mdy': list(),
        'pdx': list(),
        'pdy': list(),
        'adx': list(),
        'ady': list(),
    } for _ in range(5)]
    nb_errors = 0

    with torch.no_grad():
        for images, gt_mask, mask_position, gt_position in tqdm.tqdm(loader, desc="Training", total=len(loader)):

            images = images.to(device)
            gt_mask = gt_mask.to(device)
            gt_position = gt_position.to(device)
            pred_mask, pred_position = model(images)
            pred_position = pred_position.reshape(pred_position.shape[0], 5, 2)

            loss_mask = criterion_mask(pred_mask, gt_mask)
            loss_position = criterion_position(pred_position, gt_position)

            total_mask_loss += loss_mask.item()
            total_pos_loss += loss_position.item()
            
            for level in range(5):
                
                _pred_positionX, _pred_positionY = pred_position[:, level].squeeze()
                
                _pred_positionX = _pred_positionX * 384
                _pred_positionY = _pred_positionY * 384
                
                mask_ = pred_mask[:, level]
                mask_ = mask_.squeeze()
                pos_pred = find_center_of_largest_activation(mask_)
                if pos_pred is not None:
                    predx, predy = pos_pred
                else:
                    predx, predy = 1234, 1234
                    nb_errors += 1
                gt_posx, gt_posy = mask_position[:, level].squeeze()
                metrics_position[level]['mdx'].append(abs(predx - gt_posx.item()))
                metrics_position[level]['mdy'].append(abs(predy - gt_posy.item()))
                
                metrics_position[level]['pdx'].append(abs(_pred_positionX - gt_posx.item()))
                metrics_position[level]['pdy'].append(abs(_pred_positionY - gt_posy.item()))
                
                avx = (_pred_positionX.item() + predx) / 2
                avy = (_pred_positionY.item() + predy) / 2
                metrics_position[level]['adx'].append(abs(avx - gt_posx.item()))
                metrics_position[level]['ady'].append(abs(avy - gt_posy.item()))

    for level in range(5):
        metrics_position[level]['mdx'] = sum(metrics_position[level]['mdx']) / (len(metrics_position[level]['mdx']) + 1e-9)
        metrics_position[level]['mdy'] = sum(metrics_position[level]['mdy']) / (len(metrics_position[level]['mdy']) + 1e-9)
        metrics_position[level]['pdx'] = sum(metrics_position[level]['pdx']) / (len(metrics_position[level]['pdx']) + 1e-9)
        metrics_position[level]['pdy'] = sum(metrics_position[level]['pdy']) / (len(metrics_position[level]['pdy']) + 1e-9)
        metrics_position[level]['adx'] = sum(metrics_position[level]['adx']) / (len(metrics_position[level]['adx']) + 1e-9)
        metrics_position[level]['ady'] = sum(metrics_position[level]['ady']) / (len(metrics_position[level]['ady']) + 1e-9)
        metrics_position[level]['md_total'] = (metrics_position[level]['mdx'] + metrics_position[level]['mdy']) / 2
        metrics_position[level]['pd_total'] = (metrics_position[level]['pdx'] + metrics_position[level]['pdy']) / 2
        metrics_position[level]['ad_total'] = (metrics_position[level]['adx'] + metrics_position[level]['ady']) / 2

    
    for level in range(5):
        print("Level:", level, "segmentation:", metrics_position[level]['mdx'], metrics_position[level]['mdy'], "position:", metrics_position[level]['pdx'], metrics_position[level]['pdy'])
        
    avg_mask_loss = total_mask_loss / len(loader)
    avg_pos_loss = total_pos_loss / len(loader)
    avg_metrics = dict()
    avg_metrics['mask_loss'] = avg_mask_loss
    avg_metrics['pos_loss'] = avg_pos_loss
    avg_metrics['mdx'] = sum(metrics_position[i]['mdx'] for i in range(5)) / 5
    avg_metrics['mdy'] = sum(metrics_position[i]['mdy'] for i in range(5)) / 5
    avg_metrics['mean_md'] = avg_metrics['mdx'] * 0.5 + avg_metrics['mdy'] * 0.5
    avg_metrics['pdx'] = sum(metrics_position[i]['pdx'] for i in range(5)) / 5
    avg_metrics['pdy'] = sum(metrics_position[i]['pdy'] for i in range(5)) / 5
    avg_metrics['mean_pd'] = avg_metrics['pdx'] * 0.5 + avg_metrics['pdy'] * 0.5
    avg_metrics['adx'] = sum(metrics_position[i]['adx'] for i in range(5)) / 5
    avg_metrics['ady'] = sum(metrics_position[i]['ady'] for i in range(5)) / 5
    avg_metrics['mean_ad'] = avg_metrics['adx'] * 0.5 + avg_metrics['ady'] * 0.5
    avg_metrics['errors_percent'] = nb_errors / (len(loader) * 5)
    avg_metrics['nb_errors'] = nb_errors
    return avg_metrics


def train_epoch(model, loader, criterion_mask, criterion_position, optimizer, device):
    model.train()

    epoch_loss = 0
    for images, gt_mask, mask_position, gt_position in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()

        images = images.to(device)
        gt_mask = gt_mask.to(device)
        gt_position = gt_position.to(device)
        pred_mask, pred_position = model(images)
        pred_position = pred_position.reshape(pred_position.shape[0], 5, 2)

        loss_mask = criterion_mask(pred_mask, gt_mask)
        loss_position = criterion_position(pred_position, gt_position)
        
        loss = (loss_mask + loss_position) / 2
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss


def train_model_sagittal(input_dir, condition, description, out_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = generate_dataset(input_dir, condition, description)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = SegmentationDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = SegmentationDataset(dataset[:nb_valid])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = LumbarSegmentationModelSagittal()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma = 0.6)
    criterion_mask = torch.nn.BCELoss()
    criterion_position = torch.nn.MSELoss()

    best_metrics = None
    best = 123456
    for epoch in range(20):
        loss_train = train_epoch(model, train_loader, criterion_mask, criterion_position, optimizer, device)
        metrics = validate(model, valid_loader, criterion_mask, criterion_position, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['mean_md'] < best:
            print("New best model !", out_name)
            best = metrics["mean_md"]
            best_metrics = metrics
            scripted_model = torch.jit.script(model)
            scripted_model.save(out_name)

        scheduler.step()
        print(scheduler.get_last_lr())
        print("-" * 55)

    return best_metrics


conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Spinal Canal Stenosis"]
descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR"]
out_name = ["segmentation_st1_left.ts", "segmentation_st1_right.ts", "segmentation_st2.ts"]
metrics = dict()

for cond, desc, out in zip(conditions, descriptions, out_name):
    print('-' * 50)
    print('-' * 50)
    print('-' * 50)
    print("Training:", cond)
    print('-' * 50)
    print('-' * 50)
    print('-' * 50)
    best = train_model_sagittal(
        input_dir="/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification",
        out_name=out,
        condition=cond,
        description=desc,
    )
    metrics[cond] = best
    
print("Done !")
print(metrics)