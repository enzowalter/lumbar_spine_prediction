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

####################################################
#        CROP SELECTION
####################################################

class DynamicModelLoader:
    def __init__(self, model_name, hidden_size=None):
        self.model = timm.create_model(model_name, pretrained=False)
        self.model_name = model_name
        self.hidden_size = hidden_size
        if hidden_size is not None:
            self.model = self.modify_classifier()

    def modify_classifier(self):
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        elif hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.fc.in_features if hasattr(self.model.head, 'fc') else self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError("Unknown classifier structure")
        return self.model

class CropEncoder(nn.Module):
    def __init__(self, model_name, features_size):
        super().__init__()
        self.backbone = DynamicModelLoader(model_name, hidden_size=features_size).model

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = torch.mean(features, dim = (2, 3))
        return features

class CropSelecter(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.features_size = 1024
        self.encoder = CropEncoder(model_name, self.features_size)
        self.lstm = nn.LSTM(self.features_size, self.features_size // 4, bidirectional=True, batch_first=True, dropout=0.021, num_layers=2)
        self.classifier = nn.Linear(self.features_size // 2, 1)

    def forward(self, crops):
        b, s, c, h, w = crops.size()
        features = torch.empty(b, s, self.features_size).to(crops.device)
        for i in range(s):
            features[:, i] = self.encoder(crops[:, i])
        lstm_out, _ = self.lstm(features)
        scores = self.classifier(lstm_out)
        return scores.sigmoid().squeeze(-1)

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
    """
    Return best slices for each level.
    """

    nb_slices = len(pathes)
    images = np.zeros((nb_slices, 1, *input_size))
    for k, path in enumerate(pathes):
        im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                        input_size,
                                        interpolation=cv2.INTER_LINEAR)
        im = (im - im.min()) / (im.max() - im.min() + 1e-9)
        images[k, 0, ...] = im
    images = torch.tensor(images).expand(nb_slices, 3, *input_size).float()
    with torch.no_grad():
        preds_model = slice_model(images.to(device).unsqueeze(0))
    preds_model = preds_model.squeeze()
    preds_overall = torch.sum(preds_model, dim=0)

    # get best 3 overall (=> best after sum of each level)
    values, max_indices = get_max_consecutive_N(preds_overall, 3)
    best_slices_overall = dict()
    best_slices_overall['pathes'] = [pathes[i] for i in max_indices]
    best_slices_overall['values'] = [v for v in values]

    # get best 8 by level (for crop selection)
    slices8_by_level = [
        {"pathes": list(), "values": list()} for _ in range(preds_model.shape[0])
    ]
    for level in range(preds_model.shape[0]):
        pred_level = preds_model[level, :]
        values, max_indice = get_max_consecutive_N(pred_level, 8)
        slices8_by_level[level]['pathes'] = [pathes[i] for i in max_indice]
        slices8_by_level[level]['values'] = [v for v in values]

    return best_slices_overall, slices8_by_level


##########################################################
#
#   SEGMENTATION INFERENCE
#
##########################################################

def find_center_of_largest_activation(mask: torch.tensor) -> tuple:
    mask = (mask > 0.5).float().detach().cpu().numpy()
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return None
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0
    largest_component_center = center_of_mass(labeled_mask == np.argmax(sizes))
    center_coords = tuple(map(int, largest_component_center))
    return (center_coords[1] / mask.shape[1], center_coords[0] / mask.shape[0]) # x, y normalised

def get_segmentation_input(slices_to_use, mode, seg_input_size, device):
    if mode == "best_overall":
        images = np.zeros((3, *seg_input_size))
        _slices_path = slices_to_use
        for k, path in enumerate(_slices_path):
            im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                            seg_input_size,
                            interpolation=cv2.INTER_LINEAR
                        )
            im = (im - im.min()) / (im.max() - im.min() + 1e-9)
            images[k, ...] = im
        images = torch.tensor(images).float().to(device)
        return images
    
    elif mode == "best_by_level":
        images = np.zeros((5, 3, *seg_input_size))
        for level in range(5):
            _slices_to_use = slices_to_use
            _slices_to_use = sorted(_slices_to_use, key = lambda x: get_instance(x))
            for ch, slice_to_use in enumerate(_slices_to_use):
                im = cv2.resize(pydicom.dcmread(slice_to_use).pixel_array.astype(np.float32), 
                                seg_input_size,
                                interpolation=cv2.INTER_LINEAR
                            )
                im = (im - im.min()) / (im.max() - im.min() + 1e-9)
                images[level, ch, ...] = im
        images = torch.tensor(images).float().to(device)
        return images
    else:
        return None

def get_position_by_level(model, slices_to_use, mode, seg_input_size, device) -> dict:
    inputs = get_segmentation_input(slices_to_use, mode, seg_input_size, device)
    if mode == "best_overall":
        with torch.no_grad():
            masks = model(inputs.unsqueeze(0)) # model predict 5 levels
        masks = masks.squeeze()
        position_by_level = [find_center_of_largest_activation(masks[i]) for i in range(5)]
    else:
        with torch.no_grad():
            masks = model(inputs) # model predict 1 level, we put levels in batch dim
        masks = masks.squeeze(1)
        position_by_level = [find_center_of_largest_activation(masks[i]) for i in range(5)]
    return position_by_level


##########################################################
#
#   CROP SELECTION INFERENCE
#
##########################################################

def clahe_equalization_norm2(image, clip_limit=2.0, grid_size=(8, 8)):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    image = clahe.apply(np.uint8(image))
    image = image.astype(np.float32) / 255.
    return image

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
    output_crops = np.zeros((len(slices_path), 128, 128))
    for k, slice_path in enumerate(slices_path):
        pixel_array = pydicom.dcmread(slice_path).pixel_array.astype(np.float32)
        pixel_array = cv2.resize(pixel_array, image_resize, interpolation=cv2.INTER_LINEAR)
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # x y reversed in array
        crop = clahe_equalization_norm2(crop)
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

def get_8crops_by_level(slices_to_use, position_by_level, crop_selecter_crop_size, crop_selecter_resize_image, device):
    crops_output = np.zeros((5, 8, 128, 128))
    for level, (slices_, position) in enumerate(zip(slices_to_use, position_by_level)):
        if position is not None:
            slices = slices_
            px = int(position[0] * crop_selecter_resize_image[1])
            py = int(position[1] * crop_selecter_resize_image[0])
            crops_output[level, ...] = cut_crops(slices, px, py, crop_selecter_crop_size, crop_selecter_resize_image)
        else:
            print("No prediction on segmentation !", level)

    crops_output = torch.tensor(crops_output).float().to(device)
    crops_output = crops_output.unsqueeze(2).expand(5, 8, 3, 128, 128)
    return crops_output

def compute_best3_by_level(crops, slices_to_use, model):
    with torch.no_grad():
        preds = model(crops)
    slices_to_ret = list()
    for level in range(5):
        pred = preds[level]
        s_ = slices_to_use[level]
        values, indices = get_max_consecutive_N(pred, 3)
        t = list()
        for indice in indices:
            t.append(s_[indice])
        slices_to_ret.append(t)

    return slices_to_ret


############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_dataset(input_dir, crop_description, crop_condition, crop_size, image_resize, slice_model, segmentation_model, crop_model):
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
                    best_overall, best8_by_level = get_best_slice_selection((224, 224), device, slice_model, all_slices_path)
                    best8_by_level = [x['pathes'] for x in best8_by_level]

                    # segmentation model, slices_to_use, mode, seg_input_size, device
                    positions = get_position_by_level(segmentation_model, best_overall['pathes'], 'best_overall', (384, 384), device)

                    # crop selection slices_to_use, position_by_level, crop_selecter_crop_size, crop_selecter_resize_image, device
                    crops_by_level = get_8crops_by_level(best8_by_level, positions, (64, 96), (640, 640), device)
                    crops_to_use = compute_best3_by_level(crops_by_level, best8_by_level, crop_model)

                    for level in range(5):

                        _crops_to_use = crops_to_use[level]
                        _position = positions[level]
                        if _position is not None:
                            x, y = _position
                            x = int(x * image_resize[1]) 
                            y = int(y * image_resize[0])

                            dataset_item = dict()
                            dataset_item['study_id'] = study_id
                            dataset_item['all_slices'] = _crops_to_use
                            dataset_item['series_id'] = s_id
                            dataset_item['position'] = (x, y)
                            dataset_item['gt_label'] = gt_labels[level]
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
    angle = random.uniform(-10, 10)
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
        slices = data['all_slices']

        crops = cut_crops(slices, x, y, data['crop_size'], data['image_resize'])
        crops = torch.tensor(crops).float()
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

def train_epoch(model, loader, criterion, optimizer_encoders, device, accumulation_step):
    model.train()
    epoch_loss = 0
    optimizer_encoders.zero_grad()
    for step, (images, labels) in tqdm.tqdm(enumerate(loader), desc="Training", total=len(loader)):
        optimizer_encoders.zero_grad()
        images = images.to(device)
        labels = labels.to(device).long()

        predictions = model(images.to(device), mode="train")
        loss = criterion(predictions, labels) / accumulation_step
        loss.backward()
        epoch_loss += loss.item() / len(loader)

        if (step + 1) % accumulation_step == 0:
            optimizer_encoders.step()
            optimizer_encoders.zero_grad()
    
    if (step + 1) % accumulation_step != 0:
        optimizer_encoders.step()
        optimizer_encoders.zero_grad()

    return epoch_loss

def train_submodel(input_dir, model_name, crop_description, crop_condition, crop_size, image_resize, slice_model, segmentation_model, crop_model):

    # load models
    slice_model = torch.jit.load(slice_model).eval().cuda()
    segmentation_model = torch.jit.load(segmentation_model).eval().cuda()
    crop_model_ = CropSelecter('convnext_base.fb_in22k_ft_in1k')
    crop_model_.load_state_dict(torch.load(crop_model))
    crop_model = crop_model_.eval().cuda() 

    # get dataset
    dataset = generate_dataset(input_dir, crop_description, crop_condition, crop_size, image_resize, slice_model, segmentation_model, crop_model)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    backbones_768 = ['focalnet_small_lrf.ms_in1k', 'cs3darknet_m.c2ns_in1k', 'convnextv2_tiny.fcmae_ft_in1k', 'twins_svt_base.in1k']
    #backbones = ['cspresnet50.ra_in1k', 'convnext_base.fb_in22k_ft_in1k', 'ese_vovnet39b.ra_in1k', 'densenet161.tv_in1k', 'dm_nfnet_f0.dm_in1k']

    model = REM(
        n_classes=3,
        n_fold_classifier=3,
        backbones=backbones_768,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    parameters_encoder = list(model.encoders.parameters()) + list(model.classifiers.parameters())
    optimizer_encoders = torch.optim.AdamW(parameters_encoder, lr=0.0001)

    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1, 2, 4]).float().to(device))
    best = 123456
    for epoch in range(15):
        loss_train = train_epoch(model, train_loader, criterion, optimizer_encoders, device, accumulation_step = 16)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['concat_loss'] < best:
            print("New best model !", "Weights encoders", model.weights_encoders)
            best = metrics["concat_loss"]
            torch.save(model.state_dict(), model_name)
        print('-' * 50)

    return best

if __name__ == "__main__":

    conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Spinal Canal Stenosis"] # , "Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR"] #, "Axial T2", "Axial T2"]
    crop_sizes = [(80, 120), (80, 120), (80, 120)] #, (128, 128), (128, 128)]
    out_name = ["classification_st1_left.pth", "classification_st1_right.pth", "classification_st2.pth"] #, "classification_ax_left.pth", "classification_ax_right.pth"]

    slice_models = ['../trained_models/v6/slice_selector_st1_left.ts', '../trained_models/v6/slice_selector_st1_right.ts', '../trained_models/v6/slice_selector_st2.ts']
    segmentation_models = ['../trained_models/v6/model_segmentation_st1_left.ts', '../trained_models/v6/model_segmentation_st1_right.ts', '../trained_models/v6/model_segmentation_st2.ts']
    crop_models = ['../trained_models/v6/model_crop_selection_st1_left.pth', '../trained_models/v6/model_crop_selection_st1_right.pth', '../trained_models/v6/model_crop_selection_st2.pth']

    metrics = dict()

    for cond, desc, csize, out, slice_m, seg_m, crop_m in zip(conditions, descriptions, crop_sizes, out_name, slice_models, segmentation_models, crop_models):
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
            segmentation_model = seg_m,
            crop_model = crop_m,
        )
        metrics[cond] = best

    print("Done !")
    print(metrics)