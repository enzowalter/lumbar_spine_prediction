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
from scipy.ndimage import label, center_of_mass

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

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_best_slice_selection(model_selection, pathes):
    nb_slices = len(pathes)
    images = np.zeros((nb_slices, 1, 224, 224))
    for k, path in enumerate(pathes):
        im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                        (224, 224),
                                        interpolation=cv2.INTER_LINEAR)
        im = (im - im.min()) / (im.max() - im.min() + 1e-9)
        images[k, 0, ...] = im
    images = torch.tensor(images).expand(nb_slices, 3, 224, 224).float()
    with torch.no_grad():
        preds = model_selection(images.to(get_device()).unsqueeze(0)).squeeze()
    preds_overall = torch.sum(preds, dim=0)

    # get best by level
    slices_by_level = [
        {"pathes": list(), "values": list()} for _ in range(preds.shape[0])
    ]
    for level in range(preds.shape[0]):
        pred_level = preds[level, :]
        values, max_indice = get_max_consecutive3(pred_level)
        slices_by_level[level]['pathes'] = [pathes[i] for i in max_indice]
        slices_by_level[level]['values'] = [v for v in values]

    # get best overall (=> best after sum of each level)
    values, max_indices = get_max_consecutive3(preds_overall)
    best_slices_overall = dict()
    best_slices_overall['pathes'] = [pathes[i] for i in max_indices]
    best_slices_overall['values'] = [v for v in values]

    return slices_by_level, best_slices_overall


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

def clahe_equalization_norm2(image, clip_limit=2.0, grid_size=(8, 8)):
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
        crop = clahe_equalization_norm2(crop)
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

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

def get_segmentation_input(slices_to_use: list):
    images = np.zeros((3, 384, 384))
    _slices_path = slices_to_use[:3]
    for k, path in enumerate(_slices_path):
        im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                        (384, 384),
                        interpolation=cv2.INTER_LINEAR
                    )
        im = (im - im.min()) / (im.max() - im.min() + 1e-9)
        images[k, ...] = im
    images = torch.tensor(images).float().to(get_device())
    return images

def get_position_by_level(slices_to_use: list, model_segmentation) -> dict:
    inputs = get_segmentation_input(slices_to_use['pathes'])
    with torch.no_grad():
        masks = model_segmentation(inputs.unsqueeze(0)) # model predict 5 levels
    masks = masks.squeeze()
    position_by_level = [find_center_of_largest_activation(masks[i]) for i in range(5)]
    return position_by_level

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_dataset(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize, selection_path, segmentation_path):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}

    model_selection = torch.jit.load(selection_path).eval().to(get_device())
    model_segmentation = torch.jit.load(segmentation_path).eval().to(get_device())

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
                    best_per_level, best_overall = get_best_slice_selection(model_selection, all_slices_path)
                    position_by_level = get_position_by_level(best_overall, model_segmentation) 

                    for level in range(5):
                        if position_by_level[level] is None:
                            continue
                        dataset_item = dict()
                        dataset_item['study_id'] = study_id
                        dataset_item['level'] = level
                        dataset_item['slices_to_use'] = best_per_level[level]['pathes']
                        dataset_item['series_id'] = s_id
                        x, y = position_by_level[level]
                        dataset_item['position'] = (int(x * 640), int(y * 640))
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

class CropClassifierDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        slices_to_use = data['slices_to_use']
        x, y = data['position']
        crops = cut_crops(slices_to_use, x, y, data['crop_size'], data['image_resize'])
        crops = torch.tensor(crops).float()
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

def validate(model, loader, criterion, device):
    model.eval()
    classification_loss_sum = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels_gt in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()
            final_output = model.forward_fold(images.to(device), mode="valid")
            
            all_predictions.append(final_output.cpu())
            all_labels.append(labels_gt.cpu())
            
            loss = criterion(final_output, labels_gt)
            classification_loss_sum += loss.item()

    all_predictions = torch.cat(all_predictions, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    
    concat_loss = criterion(all_predictions, all_labels).item()
    avg_classification_loss = classification_loss_sum / len(loader)
    return {"concat_loss": concat_loss, "mean_loss": avg_classification_loss}

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, labels in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device).long()
        
        predictions = model.forward_fold(images.to(device), mode="train")
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss


def train_submodel(input_dir, model_name, crop_description, crop_condition, label_condition, crop_size, image_resize, select_path, seg_path):
    # get dataset
    dataset = generate_dataset(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize, select_path, seg_path)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # get model
    model = FoldModelClassifier(
        n_classes=3,
        n_fold_classifier=3,
        backbones=['densenet201.tv_in1k', 'seresnext101_32x4d.gluon_in1k', 'convnext_base.fb_in22k_ft_in1k', 'dm_nfnet_f0.dm_in1k', 'mobilenetv3_small_100.lamb_in1k'],
        features_size=384,
    )
    device = get_device()
    model = model.to(device)

    # train with folding
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1, 2, 4]).float().to(device))
    best = 123456
    for epoch in range(20):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['concat_loss'] < best:
            print("New best model !", model_name)
            best = metrics["concat_loss"]
            torch.save(model.state_dict(), model_name)
        scheduler.step()
        print('-' * 50)
    return best

if __name__ == "__main__":

    # best_logloss1 = train_submodel(
    #                 input_dir="../../REFAIT",
    #                 model_name="classification_spinal_canal_stenosis_pipeline_dataset.pth",
    #                 crop_condition="Spinal Canal Stenosis",
    #                 label_condition="Spinal Canal Stenosis",
    #                 crop_description="Sagittal T2/STIR",
    #                 crop_size=(80, 120),
    #                 image_resize=(640, 640),
    #                 select_path="../trained_models/v3/model_slice_selection_st2.ts",
    #                 seg_path="../trained_models/v3/model_segmentation_st2_384x384.ts",
    # )
    best_logloss2 = train_submodel(
                    input_dir="../../REFAIT",
                    model_name="classification_left_neural_foraminal_narrowing_pipeline_dataset.pth",
                    crop_condition="Left Neural Foraminal Narrowing",
                    label_condition="Left Neural Foraminal Narrowing",
                    crop_description="Sagittal T1",
                    crop_size=(80, 120),
                    image_resize=(640, 640),
                    select_path="../trained_models/v3/model_slice_selection_st1_left.ts",
                    seg_path="../trained_models/v3/model_segmentation_st1_left_384x384.ts",
    )
    '''
    best_logloss3 = train_submodel(
                    input_dir="../../REFAIT",
                    model_name="classification_right_neural_foraminal_narrowing_pipeline_dataset.pth",
                    crop_condition="Right Neural Foraminal Narrowing",
                    label_condition="Right Neural Foraminal Narrowing",
                    crop_description="Sagittal T1",
                    crop_size=(80, 120),
                    image_resize=(640, 640),
                    select_path="../trained_models/v3/model_slice_selection_st1_right.ts",
                    seg_path="../trained_models/v3/model_segmentation_st1_right_384x384.ts",
    )
    print("F>DDFQSSQDF>FDJK>DKJF>DJK")
    print("F>DDFQSSQDF>FDJK>DKJF>DJK")
    print(best_logloss1, best_logloss2, best_logloss3)

    best_logloss4 = train_submodel(
                    input_dir="../../REFAIT",
                    model_name="classification_right_subarticular_stenosis.pth",
                    crop_condition="Right Subarticular Stenosis",
                    label_condition="Right Subarticular Stenosis",
                    crop_description="Axial T2",
                    crop_size=(96, 96),
                    image_resize=(720, 720),
    )
    best_logloss5 = train_submodel(
                    input_dir="../../REFAIT",
                    model_name="classification_left_subarticular_stenosis.pth",
                    crop_condition="Left Subarticular Stenosis",
                    label_condition="Left Subarticular Stenosis",
                    crop_description="Axial T2",
                    crop_size=(96, 96),
                    image_resize=(720, 720),
    )
    print("F>DDFQSSQDF>FDJK>DKJF>DJK")
    print("F>DDFQSSQDF>FDJK>DKJF>DJK")
    print("F>DDFQSSQDF>FDJK>DKJF>DJK")
    print("F>DDFQSSQDF>FDJK>DKJF>DJK")
    print("F>DDFQSSQDF>FDJK>DKJF>DJK")
    print("F>DDFQSSQDF>FDJK>DKJF>DJK")
    print(best_logloss1, best_logloss2, best_logloss3, best_logloss4, best_logloss5)
    '''
