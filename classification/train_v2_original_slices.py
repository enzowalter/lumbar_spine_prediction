import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import copy

import glob
import torch
import warnings
warnings.filterwarnings("ignore") # warning on lstm

from models import *
from med3d import ResnetCropClassifier

############################################################
############################################################
#          GEN DATASET UTILS
############################################################
############################################################

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def get_best_slice_selection(slice_model, pathes, topk, device):
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
    slices_by_level = [list() for _ in range(preds.shape[0])]
    for level in range(preds.shape[0]):
        pred_level = preds[level, :]
        _, max_indice = torch.topk(pred_level, topk)
        slices_by_level[level] = sorted([pathes[i.item()] for i in max_indice], key = lambda x : get_instance(x))
    return slices_by_level

def get_study_labels(study_id, df_study_labels, condition, levels, labels):
    label_name = condition.lower().replace(" ", "_")
    label_name = "left_subarticular_stenosis"
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
    output_crops = np.zeros((len(slices_path), 224, 224))
    for k, slice_path in enumerate(slices_path):
        pixel_array = pydicom.dcmread(slice_path).pixel_array.astype(np.float32)
        pixel_array = cv2.resize(pixel_array, image_resize, interpolation=cv2.INTER_LINEAR)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-9)
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # x y reversed in array
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_dataset(input_dir, conditions, description, slice_model, crop_size, image_resize):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Generates classification dataset"):

        series_id = df_study_descriptions[(df_study_descriptions['study_id'] == study_id)
                                        & (df_study_descriptions['series_description'] == description)]['series_id'].to_list()
        
        for s_id in series_id:
            coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                                & (df_study_coordinates['condition'].isin(conditions))
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            # add to dataset only if all vertebraes in gt
            if len(coordinates_dict) == len(LEVELS):
                all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))
                gt_labels = get_study_labels(study_id, df_study_labels, conditions[0], LEVELS, LABELS)

                if gt_labels is not None:
                    for coordinate in coordinates_dict:
                        try:
                            idx_level = LEVELS[coordinate['level']]
                            x, y = coordinate['x'], coordinate['y']
                            dataset_item = dict()
                            dataset_item['slices_path'] = list()

                            # get original slices
                            for idx, sp in enumerate(all_slices_path):
                                if get_instance(sp) == coordinate['instance_number']:
                                    if idx == 0:
                                        dataset_item['slices_path'].append(all_slices_path[idx])        
                                        dataset_item['slices_path'].append(all_slices_path[idx + 1])        
                                        dataset_item['slices_path'].append(all_slices_path[idx + 2])
                                    elif idx == len(all_slices_path) - 1:
                                        dataset_item['slices_path'].append(all_slices_path[idx - 2])        
                                        dataset_item['slices_path'].append(all_slices_path[idx - 1])        
                                        dataset_item['slices_path'].append(all_slices_path[idx])
                                    else:
                                        dataset_item['slices_path'].append(all_slices_path[idx - 1])        
                                        dataset_item['slices_path'].append(all_slices_path[idx])        
                                        dataset_item['slices_path'].append(all_slices_path[idx + 1])

                            idx_level = LEVELS[coordinate['level']]
                            x, y = coordinate['x'], coordinate['y']
                            original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{s_id}/{coordinate['instance_number']}.dcm").pixel_array.shape
                            x = int((x / original_shape[1]) * image_resize[0]) 
                            y = int((y / original_shape[0]) * image_resize[1])

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

class CropClassifierDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        slices_to_use = data['slices_path']
        #slices_path, x, y, crop_size, image_resize
        x, y = data['position']
        #dx = torch.randint(-7, 8, (1,)).item()
        #dy = torch.randint(-7, 8, (1,)).item()

        #x += dx
        #y += dy

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
    total_log_loss = 0
    total_samples = 0
    weights = np.array([1, 2, 4])
    matrix = np.zeros((3, 3))
    with torch.no_grad():
        for images, labels_gt in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()
            final_output = model(images.to(device), mode="valid")

            i = torch.argmax(final_output, dim = 1)
            matrix[labels_gt.item(), i.item()] += 1

            loss = criterion(final_output, labels_gt)
            classification_loss_sum += loss.item()

            log_loss_batch = calculate_log_loss(final_output, labels_gt, weights)
            total_log_loss += log_loss_batch.sum().item()
            total_samples += labels_gt.size(0)

        avg_classification_loss = classification_loss_sum / len(loader)
        avg_log_loss = total_log_loss / total_samples

    avg_classification_loss = classification_loss_sum / len(loader)

    weights = np.array([1, 2, 4])
    TP = np.diag(matrix)
    FP = np.sum(matrix, axis=0) - TP
    FN = np.sum(matrix, axis=1) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    weighted_precision = np.sum(weights * precision) / np.sum(weights)
    weighted_recall = np.sum(weights * recall) / np.sum(weights)
    weighted_f1_score = np.sum(weights * f1_score) / np.sum(weights)

    print(matrix)
    print("Precision:", weighted_precision, "Recall:", weighted_recall, "F1:", weighted_f1_score)
    return {"loss": avg_classification_loss, "logloss": avg_log_loss}

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, labels in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device).long()

        predictions = model(images.to(device), mode="train")
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss

def train_model(input_dir, conditions, description, slice_model_path, crop_size, image_resize, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slice_model = torch.jit.load(slice_model_path, map_location='cpu')
    slice_model = slice_model.eval()
    slice_model = slice_model.to(device)

    dataset = generate_dataset(input_dir, conditions, description, slice_model, crop_size, image_resize)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:])
    valid_dataset = CropClassifierDataset(dataset[:nb_valid])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = FoldModelClassifier(
        n_classes=3,
        n_fold_classifier=3,
        backbones=['ecaresnet26t.ra2_in1k', "seresnet50.a1_in1k", "resnet26t.ra2_in1k", "mobilenetv3_small_100.lamb_in1k", "efficientnet_b0.ra_in1k"],
        features_size=256,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1/7, 2/7, 4/7]).to(device))

    best = 123456
    for epoch in range(10):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['logloss'] < best:
            print("New best model !", model_name)
            best = metrics["logloss"]
            torch.save(model.state_dict(), model_name)
            #model_scripted = torch.jit.script(model)
            #model_scripted.save(model_name)
        
        scheduler.step()
        print("LR=", scheduler.get_last_lr())
        print("-" * 55)
    
    print("-" * 55)
    print("-" * 55)
    print("DONE !")
    print("Model saved:", model_name)
    print("Best metrics:", best)
    print("-" * 55)
    print("-" * 55)

    torch.cuda.empty_cache()
    return best

if __name__ == "__main__":
    m1 = train_model(
                "../../REFAIT",
                ["Spinal Canal Stenosis"],
                "Sagittal T2/STIR",
                "../trained_models/v0/model_slice_selection_st2.ts",
                (96, 128),
                (640, 640),
                f"model_classification_st2_original_slices.ts",
            )
