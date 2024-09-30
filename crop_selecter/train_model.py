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

from models import *

############################################################
############################################################
#          GEN DATASET UTILS
############################################################
############################################################

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

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

def z_score_normalization(image):
    mean = image.mean()
    std = image.std()
    return (image - mean) / std

def clahe_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    image = np.uint8(image)
    return clahe.apply(image)

def clahe_equalization_norm2(image, clip_limit=2.0, grid_size=(8, 8)):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    image = clahe.apply(np.uint8(image))
    image = image.astype(np.float32) / 255.
    return image

def clahe_equalization_norm(image, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    image = np.uint8(image)
    image = clahe.apply(image)
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    return image

def compute_dahe(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = image * 255.
    image = image.astype(np.uint8)
    hist_original = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    img_eq = cv2.equalizeHist(image)
    hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256]).flatten()
    hist_diff = np.abs(hist_original - hist_eq)
    hist_diff = hist_diff.astype(np.float32)
    hist_diff_normalized = hist_diff / np.max(hist_diff)
    return hist_diff_normalized

def laplacian_norm_filter(image):
    image = image.astype(np.int16)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    image = laplacian.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min())
    return image

def laplacian_norm_filter2(image):
    image = image.astype(np.int16)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    image = laplacian.astype(np.float32)
    image = image / 255.
    return image

def laplacian_filter(image):
    image = image.astype(np.int16)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian

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

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def get_max_consecutive(preds, n):
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
        _, max_indice = get_max_consecutive_N(pred_level, 8)
        slices_by_level[level] = [pathes[i] for i in max_indice]
    return slices_by_level

def generate_scores(num_images, index):
    scores = np.zeros(num_images)
    scores[index] = 1
    if index - 1 >= 0:
        scores[index - 1] = 0.5
    if index + 1 <= num_images - 1:
        scores[index + 1] = 0.5
    return scores

def generate_dataset(input_dir, model_selection, crop_description, crop_condition, crop_size, image_resize):

    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
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
                if len(all_slices_path) < 10:
                    print("No enought slices !")
                    continue

                slices_by_level = get_best_slice_selection((224, 224), get_device(), model_selection, all_slices_path)

                for coordinate in coordinates_dict:
                    level = coordinate['level']
                    idx_level = LEVELS[level]
                    slices_to_use = slices_by_level[idx_level]
                    gt_instance = coordinate["instance_number"]

                    has_instance = False
                    for sp in slices_to_use:
                        if get_instance(sp) == gt_instance:
                            has_instance = True
                            break

                    if has_instance:
                        original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{coordinate['series_id']}/{gt_instance}.dcm").pixel_array.shape
                        x = int((coordinate['x'] / original_shape[1]) * image_resize[1])
                        y = int((coordinate['y'] / original_shape[0]) * image_resize[0])

                        dataset_item = dict()
                        dataset_item['all_slices_path'] = sorted(slices_to_use, key = lambda x : get_instance(x)) # in case of
                        dataset_item['position'] = (x, y)
                        dataset_item['gt_instance'] = gt_instance
                        dataset_item['crop_size'] = crop_size
                        dataset_item['image_resize'] = image_resize
                        dataset.append(dataset_item)
                    else:
                        print("Fail on slice selector !")

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

        all_slices_path = data['all_slices_path']
        gt_instance = data['gt_instance']
        x, y = data["position"]
        crop_size = data['crop_size']
        image_resize = data['image_resize']

        if self.is_train:
            x += random.randint(-8, 8)
            y += random.randint(-8, 8)

        crops = cut_crops(all_slices_path, x, y, crop_size, image_resize)

        gt_index = None
        for k, s in enumerate(all_slices_path):
            if get_instance(s) == gt_instance:
                gt_index = k
                break
        gt_scores = generate_scores(8, gt_index)

        crops = torch.tensor(crops).float().unsqueeze(1)
        crops = crops.expand(8, 3, 128, 128)
        return crops, torch.tensor(gt_scores).float(), gt_index

############################################################
############################################################
#          TRAINING
############################################################
############################################################

def validate(model, loader, criterion, device):
    model.eval()
    selection_loss_sum = 0

    good_item = 0
    top3_good = 0
    distances = list()
    with torch.no_grad():
        for images, labels_gt, gt_index in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device)
            final_output = model(images.to(device), mode="inference")
            loss = criterion(final_output, labels_gt)

            _, indice = torch.max(final_output, dim=1)
            if indice.item() == gt_index.item():
                good_item += 1
            else:
                distances.append(abs(indice.item() - gt_index.item()))

            consecutive_sums = final_output[:, :-2] + final_output[:, 1:-1] + final_output[:, 2:]
            _, max_consecutive_idx = torch.max(consecutive_sums, dim=1)
            max_consecutive_3 = (max_consecutive_idx.item(), max_consecutive_idx.item() + 1, max_consecutive_idx.item() + 2)
            if gt_index.item() in max_consecutive_3:
                top3_good += 1

            selection_loss_sum += loss.item()

    avg_classification_loss = selection_loss_sum / len(loader)
    return {
            "loss": avg_classification_loss, 
            "accuracy_top1": good_item / len(loader), 
            "accuracy_top3": top3_good / len(loader),
            "error_distance": sum(distances) / len(distances) if len(distances) > 0 else 0
            }

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    epoch_loss = 0
    for images, labels, _ in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images.to(device), mode="train")
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)
    return epoch_loss

def train_crop_selecter(input_dir, slice_model_name, model_name, crop_description, crop_condition, crop_size, image_resize):

    # get model
    model = torch.jit.load(slice_model_name)
    model = model.eval()
    model = model.to(get_device())

    # get dataset
    dataset = generate_dataset(input_dir, model, crop_description, crop_condition, crop_size, image_resize)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = REM_CropSelecterModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss().to(device)
    best = -1
    for epoch in range(15):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['accuracy_top3'] > best:
            print("New best model !", model_name)
            best = metrics["accuracy_top3"]
            scripted_model = REM_CropSelecterModel_Scripted()
            scripted_model.load_state_dict(model.state_dict())
            scripted_model = torch.jit.script(scripted_model)
            scripted_model.save(model_name)
        
        print('-' * 50)

    print('-' * 50)
    print('-' * 50)
    print('-' * 50)
    print("DONE")
    print("model", model_name)
    print("Best", best)
    print('-' * 50)
    print('-' * 50)
    print('-' * 50)

    return best

if __name__ == "__main__":


    # conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Spinal Canal Stenosis"]
    # descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR"]
    # out_name = ["crop_selection_st1_left.ts", "crop_selection_st1_right.ts", "crop_selection_st2.ts"]
    # slice_path = '../trained_models/v9/'
    # slice_models = [f"{slice_path}/slice_selector_st1_left.ts", f"{slice_path}/slice_selector_st1_right.ts", f"{slice_path}/slice_selector_st2.ts"]
    # metrics = dict()

    conditions = ["Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    descriptions = ["Axial T2", "Axial T2"]
    out_name = ["crop_selection_ax_left.ts", "crop_selection_ax_right.ts"]
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
        best = train_crop_selecter(
                        input_dir="../",
                        slice_model_name=slice_name,
                        model_name=out,
                        crop_condition=cond,
                        crop_description=desc,
                        crop_size=(128, 128),
                        image_resize=(640, 640),
        )

        metrics[cond] = best
    print("Done !")
    print(metrics)
