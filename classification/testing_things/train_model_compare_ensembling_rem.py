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

from models_tmp import *

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
    output_crops = np.zeros((len(slices_path), 128, 128))
    for k, slice_path in enumerate(slices_path):
        pixel_array = pydicom.dcmread(slice_path).pixel_array.astype(np.float32)
        pixel_array = cv2.resize(pixel_array, image_resize, interpolation=cv2.INTER_LINEAR)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-9)
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # x y reversed in array
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_dataset(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize):
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
                                    idx_to_use = idx
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
        crops = crops.expand(3, 128, 128)
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

def validate_models(models, loader, criterion, device):
    for model in models:
        model.eval()

    classification_loss_sum = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels_gt in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()

            final_output = list()
            for model in models:
                final_output.append(model(images.to(device)))

            final_output = torch.stack(final_output, dim = 1)
            final_output = torch.mean(final_output, dim = 1)

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

        predictions = model(images.to(device), mode='train')

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss


def train_submodel_test_encoder(input_dir, model_name, crop_description, crop_condition, label_condition, crop_size, image_resize, encoder_name):

    # get dataset
    dataset = generate_dataset(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # get model
    # model = SimpleClassifier(encoder_name)
    model = REM(
        n_classes=3,
        n_fold_classifier=3,
        backbones=['focalnet_base_lrf.ms_in1k', 'densenet121.ra_in1k', 'convnext_base.fb_in1k', 'darknet53.c2ns_in1k', 'hgnetv2_b1.ssld_stage2_ft_in1k']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # train with folding
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1/7, 2/7, 4/7]).to(device))
    best = 123456
    for epoch in range(20):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['concat_loss'] < best:
            print("New best model !", model_name)
            best = metrics["concat_loss"]
            n_model = REM_Script(
                    n_classes=3,
                    n_fold_classifier=3,
                    backbones=['focalnet_base_lrf.ms_in1k', 'densenet121.ra_in1k', 'convnext_base.fb_in1k', 'darknet53.c2ns_in1k', 'hgnetv2_b1.ssld_stage2_ft_in1k']
                )
            n_model.load_state_dict(model.state_dict())
            scripted_model = torch.jit.script(n_model)
            scripted_model.save(model_name)

        scheduler.step()
    return best

if __name__ == "__main__":
        
    # Ensembling
    encoders = [
        'focalnet_base_lrf.ms_in1k', # 0.5247368812561035
        'densenet121.ra_in1k', # 0.5281888246536255
        # 'convnextv2_base.fcmae_ft_in1k', 0.9270282983779907
        'convnext_base.fb_in1k', # 0.5202980637550354
        # 'cs3darknet_l.c2ns_in1k', 0.5652085542678833
        # 'cspresnet50.ra_in1k', 0.5684816241264343
        'darknet53.c2ns_in1k', # 0.5463195443153381
        # 'dla60.in1k', 0.5571991205215454
        # 'gc_efficientnetv2_rw_t.agc_in1k', 0.5591642260551453
        'hgnetv2_b1.ssld_stage2_ft_in1k', # 0.5294161438941956
        # 'inception_next_base.sail_in1k', 0.5858572125434875
        # 'mobilenetv1_100.ra4_e3600_r224_in1k', 0.5529153943061829
        # 'pit_b_distilled_224.in1k', fail
        # 'selecsls42b.in1k', 0.549743115901947
    ]
    metrics = dict()
    for k, encoder in enumerate(encoders):
        print("Training:", encoder)
        best_logloss = train_submodel_test_encoder(
                        input_dir="../../",
                        model_name=f"ensemble_{k}.ts",
                        crop_condition="Left Neural Foraminal Narrowing",
                        label_condition="Left Neural Foraminal Narrowing",
                        crop_description="Sagittal T1",
                        crop_size=(80, 120),
                        image_resize=(640, 640),
                        encoder_name=encoder,
        )
        metrics[encoder] = best_logloss
    print(metrics)

    # GET RESULTS
    dataset = generate_dataset("../../", "Sagittal T1", "Left Neural Foraminal Narrowing", "Left Neural Foraminal Narrowing", (80, 120), (640, 640))
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = list()
    for k, encoder in enumerate(encoders):
        m = torch.jit.load(f"ensemble_{k}.ts")
        m = m.to(device)
        m = m.eval()
        models.append(m)

    metrics = validate_models(models, valid_loader, torch.nn.CrossEntropyLoss(weight = torch.tensor([1/7, 2/7, 4/7]).to(device)), device)

    print("Ensembling metrics=", metrics)