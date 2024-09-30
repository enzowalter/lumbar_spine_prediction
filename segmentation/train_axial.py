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

import segmentation_models_pytorch as smp

from models_axial import LumbarSegmentationModel

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

##########################################################
#
#   SLICES SELECTION INFERENCE
#
##########################################################

def get_max_consecutive_N(preds, n):
    max_sum = -float('inf')
    max_idx = -1
    
    for i in range(len(preds) - n + 1):
        current_sum = preds[i:i + n].sum().item()
        if current_sum > max_sum:
            max_sum = current_sum
            max_idx = i
    
    max_consecutive_indices = list(range(max_idx, max_idx + n))
    values = preds[max_consecutive_indices]
    values = values.detach().cpu().numpy()
    return values, max_consecutive_indices

def get_best_slice_selection(input_size, device, slice_model, pathes):
    nb_slices = len(pathes)
    images = np.zeros((nb_slices, 1, *input_size))
    for k, path in enumerate(pathes):
        im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                        input_size,
                                        interpolation=cv2.INTER_LINEAR)
        mean = im.mean()
        std = im.std()
        im = (im - mean) / std
        images[k, 0, ...] = im

    images = torch.tensor(images).expand(nb_slices, 3, *input_size).float()
    with torch.no_grad():
        preds_model = slice_model(images.to(device).unsqueeze(0))
    preds_model = preds_model.squeeze()
    slices_by_level = [list() for _ in range(5)]
    for level in range(preds_model.shape[0]):
        pred_level = preds_model[level, :]
        _, max_indice = get_max_consecutive_N(pred_level, 3)
        slices_by_level[level] = [pathes[i] for i in max_indice]
    return slices_by_level

def generate_dataset(input_dir, condition, description, model_name):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    MASK_SIZE = (384, 384)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slice_model = torch.jit.load(model_name)
    slice_model = slice_model.eval().to(device)

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Generates segmentation masks"):

        print(study_id)

        series_id = df_study_descriptions[(df_study_descriptions['study_id'] == study_id)
                                        & (df_study_descriptions['series_description'] == description)]['series_id'].to_list()
        
        for s_id in series_id:
            coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                                & (df_study_coordinates['condition'] == condition)
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            # add to dataset only if all vertebraes in gt
            if len(coordinates_dict) == len(LEVELS):
                
                all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))
                best_slices = get_best_slice_selection((224, 224), device, slice_model, all_slices_path)

                for coordinate in coordinates_dict:
                    idx_level = LEVELS[coordinate['level']]
                    instance = coordinate['instance_number']
                    x, y = coordinate['x'], coordinate['y']
                    original_slice = f"{input_dir}/train_images/{study_id}/{s_id}/{instance}.dcm"
                    original_shape = pydicom.dcmread(original_slice).pixel_array.shape
                    x = int((x / original_shape[1]) * MASK_SIZE[1])
                    y = int((y /original_shape[0]) * MASK_SIZE[0])
                    dataset_item = dict()
                    dataset_item['slices'] = best_slices[idx_level]
                    dataset_item['gt_positions'] = [x, y]
                    dataset_item['output_mask'] = np.zeros((1, 384, 384))
                    dataset_item['output_mask'][0, y-17:y+17,x-17:x+17] = 1
                    dataset.append(dataset_item)
    return dataset

class SegmentationDataset(Dataset):
    def __init__(self, infos):
        self.datas = infos

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        slices_path = data['slices']
        nb_slices = len(slices_path)
        images = np.zeros((nb_slices, 384, 384))
        for k, path in enumerate(slices_path):
            im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                           (384, 384),
                                           interpolation=cv2.INTER_LINEAR)
            # z score
            mean = im.mean()
            std = im.std()
            im = (im - mean) / std
            images[k, ...] = im
        images = torch.tensor(images).float()
        return images, torch.tensor(data['output_mask']).float(), torch.tensor(data['gt_positions'])

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

def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    total_metrics = {
        'loss': 0,
        'precision': 0,
        'recall': 0,
        'iou': 0,
        'f1': 0
    }

    metrics_position = {
        'dx': list(),
        'dy': list(),
    }
    nb_errors = 0

    count = 0

    with torch.no_grad():
        for images, labels, gt_positions in tqdm.tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            gt_positions = gt_positions.to(device)

            predictions = model(images)

            loss = criterion(predictions, labels)
            total_loss += loss.item()

            mask_ = predictions
            mask_ = mask_.squeeze()
            pos_pred = find_center_of_largest_activation(mask_)
            if pos_pred is not None:
                predx, predy = pos_pred
            else:
                predx, predy = 1234, 1234
                nb_errors += 1
            gt_posx, gt_posy = gt_positions.squeeze()
            metrics_position['dx'].append(abs(predx - gt_posx.item()))
            metrics_position['dy'].append(abs(predy - gt_posy.item()))

            tp, fp, fn, tn = smp.metrics.get_stats(predictions.float(), labels.long(), mode='multilabel', threshold=0.5)

            total_metrics['precision'] += smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item()
            total_metrics['recall'] += smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item()
            total_metrics['iou'] += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            total_metrics['f1'] += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()

            count += 1

    avg_loss = total_loss / count
    avg_metrics = {key: total_metrics[key] / count for key in total_metrics}
    avg_metrics['loss'] = avg_loss
    avg_metrics['dx'] = sum(metrics_position['dx']) / len(metrics_position['dx'])
    avg_metrics['dy'] = sum(metrics_position['dy']) / len(metrics_position['dy'])
    avg_metrics['mean_d'] = avg_metrics['dx'] * 0.5 + avg_metrics['dy'] * 0.5
    avg_metrics['errors_percent'] = nb_errors / len(loader)
    avg_metrics['nb_errors'] = nb_errors
    return avg_metrics


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for images, labels, _ in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss

def train_model_sagittal(input_dir, condition, description, out_name, slice_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = generate_dataset(input_dir, condition, description, slice_name)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = SegmentationDataset(dataset[nb_valid:])
    valid_dataset = SegmentationDataset(dataset[:nb_valid])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = LumbarSegmentationModel()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.5)
    criterion = torch.nn.BCELoss()

    best_metrics = None
    best = 123456
    for epoch in range(30):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['mean_d'] < best:
            print("New best model !", out_name)
            best = metrics["mean_d"]
            best_metrics = metrics
            scripted_model = torch.jit.script(model)
            scripted_model.save(out_name)

        scheduler.step()
        print(scheduler.get_last_lr())
        print("-" * 55)

    return best_metrics

if __name__ == "__main__":


    conditions = ["Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    descriptions = ["Axial T2", "Axial T2"]
    out_name = ["segmentation_ax_left.ts", "segmentation_ax_right.ts"]
    slice_path = '../trained_models/v9/'
    slice_models = [f"{slice_path}/slice_selector_ax_left.ts", f"{slice_path}/slice_selector_ax_right.ts"]
    metrics = dict()

    for cond, desc, out, slice_name in zip(conditions, descriptions, out_name, slice_models):
        print('-' * 50)
        print('-' * 50)
        print('-' * 50)
        print("Training:", cond)
        print('-' * 50)
        print('-' * 50)
        print('-' * 50)
        best = train_model_sagittal(
            input_dir="../",
            out_name=out,
            condition=cond,
            description=desc,
            slice_name=slice_name,
        )
        metrics[cond] = best
    print("Done !")
    print(metrics)
