import numpy as np
import pandas as pd
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
                gt_labels = get_study_labels(study_id, df_study_labels, crop_condition, LEVELS, LABELS)

                if gt_labels is not None:
                    for coordinate in coordinates_dict:
                        try:
                            original_idx = None
                            for idx, sp in enumerate(all_slices_path):
                                if get_instance(sp) == coordinate['instance_number']:
                                    original_idx = idx
                                    break

                            if original_idx > 2 and original_idx < len(all_slices_path) - 3:
                                idx_level = LEVELS[coordinate['level']]
                                x, y = coordinate['x'], coordinate['y']
                                original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{s_id}/{coordinate['instance_number']}.dcm").pixel_array.shape
                                x = int((x / original_shape[1]) * image_resize[1]) 
                                y = int((y / original_shape[0]) * image_resize[0])

                                dataset_item = dict()
                                dataset_item['study_id'] = study_id
                                dataset_item['all_slices'] = all_slices_path
                                dataset_item['original_index'] = original_idx
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

class MedicalImageAugmentation:
    def __init__(self, rotation_range=(-5, 5), translation_range=(-10, 10), 
                 elastic_alpha=10, elastic_sigma=5, intensity_variation_range=(0.9, 1.1)):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.intensity_variation_range = intensity_variation_range

    def augment(self, image):
        image = self.random_rotation_translation(image)
        image = self.intensity_scaling(image)
        return image

    def random_rotation_translation(self, image):
        angle = random.uniform(*self.rotation_range)
        transform = T.Compose([
            T.ToPILImage(),
            T.RandomAffine(degrees=(angle, angle)),
            T.ToTensor()
        ])
        return transform(image.unsqueeze(0)).squeeze(0)

    def intensity_scaling(self, image):
        scale_factor = random.uniform(*self.intensity_variation_range)
        return image * scale_factor


class CropClassifierDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train
        self.augmenter = MedicalImageAugmentation()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        x, y = data['position']
        original_idx = data['original_index']
        all_slices_path = data['all_slices']

        if self.is_train:
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

        if self.is_train:
            x += random.randint(-8, 9)
            y += random.randint(-8, 9)

        crops = cut_crops(slices_path, x, y, data['crop_size'], data['image_resize'])
        crops = torch.tensor(crops).float()

        # if self.is_train:
        #     for i in range(crops.shape[0]):
        #         crops[i] = self.augmenter.augment(crops[i])
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

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    epoch_loss = 0
    for images, labels in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device).long()

        if (epoch + 1) % 4 == 0:
            predictions = model(images.to(device), mode="train_gate")
        else:
            predictions = model(images.to(device), mode="train")

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss

def train_submodel(input_dir, model_name, crop_description, crop_condition, crop_size, image_resize):

    # get dataset
    dataset = generate_dataset(input_dir, crop_description, crop_condition, crop_size, image_resize)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4)

    #backbones = ['cspresnet50.ra_in1k', 'convnext_base.fb_in22k_ft_in1k', 'ese_vovnet39b.ra_in1k', 'densenet161.tv_in1k', 'dm_nfnet_f0.dm_in1k']
    #backbones_1024=['focalnet_base_lrf.ms_in1k', 'densenet121.ra_in1k', 'convnext_base.fb_in1k', 'darknet53.c2ns_in1k', 'hgnetv2_b1.ssld_stage2_ft_in1k']
    backbones = ['focalnet_small_lrf.ms_in1k', 'convnext_base.fb_in22k_ft_in1k', 'dm_nfnet_f0.dm_in1k', 'ecaresnet26t.ra2_in1k', 'resnet34.a1_in1k']
    model = REM(
        n_classes=3,
        nb_classifiers=3,
        unification_features_size=1024,
        backbones=backbones,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.005)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1, 2, 4]).float().to(device))
    best = 123456
    for epoch in range(15):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['concat_loss'] < best:
            print("New best model !", "Weights encoders", model.weights_encoders_per_classifier)
            best = metrics["concat_loss"]
            # torch.save(model.state_dict(), model_name)
        print('-' * 50)

    return best

if __name__ == "__main__":

    conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Spinal Canal Stenosis", "Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR", "Axial T2", "Axial T2"]
    crop_sizes = [(80, 120), (80, 120), (80, 120), (128, 128), (128, 128)]
    out_name = ["classification_st1_left.pth", "classification_st1_right.pth", "classification_st2.pth", "classification_ax_left.pth", "classification_ax_right.pth"]

    metrics = dict()
    for cond, desc, csize, out in zip(conditions, descriptions, crop_sizes, out_name):
        print('-' * 50)
        print('-' * 50)
        print("Training:", cond)
        print('-' * 50)
        print('-' * 50)
        best = train_submodel(
            input_dir="../",
            model_name=out,
            crop_condition=cond,
            crop_description=desc,
            crop_size=csize,
            image_resize=(640, 640),
        )
        metrics[cond] = best
        break

    print("Done !")
    print(metrics)