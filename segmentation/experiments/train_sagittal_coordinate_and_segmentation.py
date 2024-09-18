import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as FT
import glob
import torch
import random
import warnings
warnings.filterwarnings("ignore") # warning on lstm

from models_sagittal import *

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def get_max_consecutive3(predictions):
    index = 1
    max_index = predictions.size(0) - 1
    best_index = None
    best_sum = -1
    while index < max_index:
        current_sum = predictions[index - 1] + predictions[index] + predictions[index + 1]
        if current_sum > best_sum:
            best_sum = current_sum
            best_index = index
        index += 1
    
    indices = [best_index-1, best_index, best_index+1]
    values = [predictions[best_index-1], predictions[best_index], predictions[best_index+1]]
    return values, indices

def get_best_slice_selection(slice_model, pathes, device):
    """Return 3 best overall slices
    """
    nb_slices = len(pathes)
    images = np.zeros((nb_slices, 1, 224, 224))
    for k, path in enumerate(pathes):
        im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                        (224, 224),
                                        interpolation=cv2.INTER_LINEAR)
        im = (im - im.min()) / (im.max() - im.min() + 1e-9)
        images[k, 0, ...] = im
    images = torch.tensor(images).expand(nb_slices, 3, 224, 224).float()
    preds = slice_model(images.to(device).unsqueeze(0)).squeeze()
    preds = torch.sum(preds, dim=0)
    _, max_indices = get_max_consecutive3(preds)
    return [pathes[i] for i in max_indices]

def generate_dataset(input_dir, conditions, description, slice_model):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    MASK_SIZE = (96, 96)
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
                                & (df_study_coordinates['condition'].isin(conditions))
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            # add to dataset only if all vertebraes in gt
            if len(coordinates_dict) == len(LEVELS):
                dataset_item = dict()
                dataset_item['study_id'] = study_id
                dataset_item['series_id'] = s_id
                dataset_item['all_slices_path'] = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))
                dataset_item['slice_per_level'] = get_best_slice_selection(slice_model, dataset_item['all_slices_path'], device)
                dataset_item['gt_positions'] = [[None, None]] * len(LEVELS)
                dataset_item['original_shape'] = [[None, None]] * len(LEVELS)
                dataset_item['output_mask'] = np.zeros((len(LEVELS), *MASK_SIZE))
                
                for coordinate in coordinates_dict:
                    idx_level = LEVELS[coordinate['level']]
                    instance = coordinate['instance_number']
                    x, y = coordinate['x'], coordinate['y']

                    original_slice = f"{input_dir}/train_images/{study_id}/{s_id}/{instance}.dcm"
                    original_shape = pydicom.dcmread(original_slice).pixel_array.shape

                    x = x / original_shape[1]
                    y = y /original_shape[0]
                    dataset_item['gt_positions'][idx_level] = [x, y]
                    dataset_item['original_shape'][idx_level] = list(original_shape)

                    x = int(x * MASK_SIZE[1])
                    y = int(y * MASK_SIZE[0])
                    dataset_item['output_mask'][idx_level][y-5:y+5,x-5:x+5] = 1

                dataset.append(dataset_item)
    return dataset

def tensor_augmentations(tensor_image):
    noise = torch.randn(tensor_image.size()) * 0.01
    tensor_image = tensor_image + noise
    return tensor_image

class PositionDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        slices_path = data['slice_per_level']
        nb_slices = len(slices_path)
        images = np.zeros((nb_slices, 384, 384))
        for k, path in enumerate(slices_path):
            im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                           (384, 384),
                                           interpolation=cv2.INTER_LINEAR)
            im = (im - im.min()) / (im.max() - im.min() + 1e-9)
            images[k, ...] = im
        images = torch.tensor(images).float()
        if self.is_train:
            images = tensor_augmentations(images)
        return images, np.array(data['gt_positions']), torch.tensor(data['output_mask']).float(), np.array(data['original_shape'])

def validate(model, loader, criterion_position, criterion_mask, device):
    model.eval()

    valid_loss = 0
    metrics = [{
        'dx': list(),
        'dy': list(),
    } for _ in range(5)]

    for images, gt_positions, gt_mask, gt_shape in tqdm.tqdm(loader, desc="Valid"):
        with torch.no_grad():
            images = images.to(device)
            gt_positions = gt_positions.to(device).float()
            gt_mask = gt_mask.to(device).float()
            
            predictions, masks = model(images)
            loss_mask = criterion_mask(masks, gt_mask)
            predictions = predictions.reshape(predictions.size(0), 5, 2)
            loss_pos = 0
            for level in range(5):
                loss_pos += criterion_position(predictions[:, level], gt_positions[:, level])

                shape = gt_shape[:, level].squeeze()
                predx, predy = predictions[:, level].squeeze()
                predx *= shape[1]
                predy *= shape[0]

                gt_posx, gt_posy = gt_positions[:, level].squeeze()
                gt_posx *= shape[1]
                gt_posy *= shape[0]

                metrics[level]['dx'].append(abs(predx - gt_posx.item()))
                metrics[level]['dy'].append(abs(predy - gt_posy.item()))

            loss_pos /= 5
            loss = loss_pos * 0.6 + loss_mask * 0.4
            valid_loss += loss.item() / len(loader)

    for level in range(5):
        metrics[level]['dx'] = sum(metrics[level]['dx']) / (len(metrics[level]['dx']) + 1e-9)
        metrics[level]['dy'] = sum(metrics[level]['dy']) / (len(metrics[level]['dy']) + 1e-9)
        metrics[level]['d_total'] = (metrics[level]['dx'] + metrics[level]['dy']) / 2

    outputs = {
        "dx": sum(metrics[i]['dx'] for i in range(5)) / 5,
        "dy": sum(metrics[i]['dy'] for i in range(5)) / 5,
    }
    outputs['d_total'] = (outputs['dx'] + outputs['dy']) / 2
    
    print("-" * 50)
    print("VALIDATION")
    for level in range(5):
        print(f"\t{level}")
        print(f"\t{metrics[level]}")
    print("-" * 50)
    return valid_loss, outputs

def train_epoch(model, loader, criterion_position, criterion_mask, optimizer, device):
    model.train()

    epoch_loss = 0
    for images, gt_positions, gt_mask, gt_shape in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()

        images = images.to(device)
        gt_positions = gt_positions.to(device).float()
        gt_mask = gt_mask.to(device).float()

        predictions, masks = model(images)

        loss_mask = criterion_mask(masks, gt_mask)

        predictions = predictions.reshape(predictions.size(0), 5, 2)
        loss_pos = 0
        for level in range(5):
            loss_pos += criterion_position(predictions[:, level], gt_positions[:, level])
        loss_pos = loss_pos / 5

        loss = loss_pos * 0.6 + loss_mask * 0.4
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss

def train_model_sagittal(input_dir, conditions, description, slice_model_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slice_model = torch.jit.load(slice_model_path, map_location='cpu')
    slice_model = slice_model.eval()
    slice_model = slice_model.to(device)

    dataset = generate_dataset(input_dir, conditions, description, slice_model)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = PositionDataset(dataset[nb_valid:], is_train=False)
    valid_dataset = PositionDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = LumbarPositionAdnSegmentationModelSagittal()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion_position = torch.nn.L1Loss()
    criterion_mask = torch.nn.BCELoss()

    best_metrics = None
    best = 123456
    for epoch in range(10):
        loss_train = train_epoch(model, train_loader, criterion_position, criterion_mask, optimizer, device)
        loss_valid, metrics = validate(model, valid_loader, criterion_position, criterion_mask, device)
        print("Epoch", epoch, "train_loss=", loss_train, "valid_loss=", loss_valid, "metrics=", metrics)
        if metrics['d_total'] < best:
            print("New best model !", model_name)
            best = metrics["d_total"]
            best_metrics = metrics
            model_scripted = torch.jit.script(model)
            model_scripted.save(model_name)
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
    m1 = train_model_sagittal(
            "../../REFAIT",
            ["Spinal Canal Stenosis"],
            "Sagittal T2/STIR",
            "../trained_models/v0/model_slice_selection_st2.ts",
            "model_segmentation_st2_coordinate_and_seg.ts",
        )