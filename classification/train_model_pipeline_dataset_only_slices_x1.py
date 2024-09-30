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
from scipy.ndimage import label, center_of_mass

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
        crop = clahe_equalization(crop)
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

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
        _, max_indice = get_max_consecutive_N(pred_level, 1)
        slices_by_level[level] = [pathes[i] for i in max_indice]
    return slices_by_level

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_dataset(input_dir, crop_description, crop_condition, crop_size, image_resize, slice_model):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    device = torch.device("cuda")

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

                gt_labels = get_study_labels(study_id, df_study_labels, crop_condition, LEVELS, LABELS)

                if gt_labels is not None:
                    
                    all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))
                    
                    # slice selector input_size, device, slice_model, pathes
                    best_by_level = get_best_slice_selection((224, 224), device, slice_model, all_slices_path)

                    # for coordinate
                    for coordinate in coordinates_dict:
                        idx_level = LEVELS[coordinate['level']]
                        x, y = coordinate['x'], coordinate['y']
                        original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{s_id}/{coordinate['instance_number']}.dcm").pixel_array.shape
                        x = int((x / original_shape[1]) * image_resize[1]) 
                        y = int((y / original_shape[0]) * image_resize[0])

                        dataset_item = dict()
                        dataset_item['slices'] = best_by_level[idx_level]
                        dataset_item['position'] = (x, y)
                        dataset_item['gt_label'] = gt_labels[idx_level]
                        dataset_item['crop_size'] = crop_size
                        dataset_item['image_resize'] = image_resize

                        dataset.append(dataset_item)

    return dataset


############################################################
############################################################
#          DATALOADER
############################################################
############################################################

def tensor_augmentations(tensor_image):
    angle = random.uniform(-12, 12)
    tensor_image = FT.rotate(tensor_image, angle)
    return tensor_image

class CropClassifierDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        x, y = data['position']
        slices = data['slices']

        if self.is_train:
            x += random.randint(-6, 6)
            y += random.randint(-6, 6)

        crops = cut_crops(slices, x, y, data['crop_size'], data['image_resize'])
        crops = torch.tensor(crops).float().expand(3, 128, 128)
        if self.is_train:
            crops = tensor_augmentations(crops)
        return crops, data['gt_label']

############################################################
############################################################
#          TRAINING
############################################################
############################################################

def validate(model, loader, criterion, device):
    model.eval()
    classification_loss_sum = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels_gt in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()
            final_output = model(images.to(device), mode="inference")
            
            all_predictions.append(final_output.cpu())
            all_labels.append(labels_gt.cpu())
            
            loss = criterion(final_output, labels_gt)
            classification_loss_sum += loss.item()

    all_predictions = torch.cat(all_predictions, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    
    concat_loss = criterion(all_predictions, all_labels).item()
    avg_classification_loss = classification_loss_sum / len(loader)
    return {"concat_loss": concat_loss, "mean_loss": avg_classification_loss}

def train_epoch(model, loader, criterion, optimizer_encoders, device):
    model.train()
    epoch_loss = 0
    for images, labels in tqdm.tqdm(loader, desc="Training"):
        optimizer_encoders.zero_grad()
        images = images.to(device)
        labels = labels.to(device).long()

        predictions = model(images.to(device), mode="train")
        loss = criterion(predictions, labels)
        loss.backward()
        epoch_loss += loss.item() / len(loader)
        optimizer_encoders.step()

    return epoch_loss

def train_submodel(input_dir, model_name, crop_description, crop_condition, crop_size, image_resize, slice_model):

    # load models
    slice_model = torch.jit.load(slice_model).eval().cuda()

    # get dataset
    dataset = generate_dataset(input_dir, crop_description, crop_condition, crop_size, image_resize, slice_model)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    backbones_768 = ['focalnet_small_lrf.ms_in1k', 'cs3darknet_m.c2ns_in1k', 'convnextv2_tiny.fcmae_ft_in1k', 'twins_svt_base.in1k']
    backbones_1024=['focalnet_base_lrf.ms_in1k', 'densenet121.ra_in1k', 'convnext_base.fb_in1k', 'darknet53.c2ns_in1k', 'hgnetv2_b1.ssld_stage2_ft_in1k']

    model = REM(
        n_classes=3,
        n_fold_classifier=3,
        backbones=backbones_1024,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    parameters_encoder = list(model.encoders.parameters()) + list(model.classifiers.parameters())
    optimizer_encoders = torch.optim.AdamW(parameters_encoder, lr=0.0001)

    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1, 2, 4]).float().to(device))
    best = 123456
    for epoch in range(15):
        loss_train = train_epoch(model, train_loader, criterion, optimizer_encoders, device)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['concat_loss'] < best:
            print("New best model !", "Weights encoders", model.weights_encoders)
            best = metrics["concat_loss"]
            n_model = REM_Script(n_classes=3, n_fold_classifier=3, backbones=backbones_1024)
            n_model.load_state_dict(model.state_dict())
            scripted_model = torch.jit.script(n_model)
            scripted_model.save(model_name)

        print('-' * 50)

    return best

if __name__ == "__main__":

    conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Spinal Canal Stenosis" , "Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR", "Axial T2", "Axial T2"]
    crop_sizes = [(128, 128), (128, 128), (128, 128), (128, 128), (128, 128)]
    out_name = ["classification_st1_left_x1.ts", "classification_st1_right_x1.ts", "classification_st2_x1.ts", "classification_ax_left_x1.pth", "classification_ax_right_x1.pth"]

    slice_models = [
        '../trained_models/v9/slice_selector_st1_left.ts',
        '../trained_models/v9/slice_selector_st1_right.ts', 
        '../trained_models/v9/slice_selector_st2.ts',
        "../trained_models/v9/slice_selector_ax_left.ts",
        "../trained_models/v9/slice_selector_ax_right.ts",
    ]

    # conditions = ["Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    # descriptions = ["Axial T2", "Axial T2"]
    # out_name = ["classification_ax_left.ts", "classification_ax_right.ts"]
    # crop_sizes = [(128, 128), (128, 128)]
    # slice_path = '../trained_models/v9/'
    # slice_models = [f"{slice_path}/slice_selector_ax_left.ts", f"{slice_path}/slice_selector_ax_right.ts"]
    metrics = dict()

    for cond, desc, csize, out, slice_m in zip(conditions, descriptions, crop_sizes, out_name, slice_models):
        print('-' * 50)
        print('-' * 50)
        print('-' * 50)
        print("Training:", cond)
        print('-' * 50)
        print('-' * 50)
        print('-' * 50)
        best = train_submodel(
            input_dir="../",
            model_name=out,
            crop_condition=cond,
            crop_description=desc,
            crop_size=csize,
            image_resize=(640, 640),
            slice_model = slice_m,
        )
        metrics[cond] = best

    print("Done !")
    print(metrics)