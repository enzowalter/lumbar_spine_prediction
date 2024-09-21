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

def cut_crops(slices_path, x, y, crop_size, image_resize, normalisation):
    output_crops = np.zeros((len(slices_path), 128, 128))
    for k, slice_path in enumerate(slices_path):
        pixel_array = pydicom.dcmread(slice_path).pixel_array.astype(np.float32)
        if normalisation == "min_max_slice":
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-9)
        
        pixel_array = cv2.resize(pixel_array, image_resize, interpolation=cv2.INTER_LINEAR)
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # x y reversed in array
        if normalisation == "min_max_crop":
            crop = (crop - crop.min()) / (crop.max() - crop.min() + 1e-9)
        elif normalisation == "laplacian":
            crop = laplacian_filter(crop)
        elif normalisation == "laplacian_norm":
            crop = laplacian_norm_filter(crop)
        elif normalisation == "laplacian_norm_2":
            crop = laplacian_norm_filter2(crop)
        elif normalisation == "clahe":
            crop = clahe_equalization(crop)
        elif normalisation == "clahe_norm":
            crop = clahe_equalization_norm(crop)
        elif normalisation == "clahe_norm_2":
            crop = clahe_equalization_norm2(crop)
        elif normalisation == "clahe_norm_gpt":
            crop = compute_dahe(crop)
        elif normalisation == "zscore":
            crop = z_score_normalization(crop)

        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_scores(num_images, index):
    scores = np.zeros(num_images)
    for i in range(num_images):
        distance = abs(i - index)
        score = max(0, 1 - (distance ** 2) * 0.1)
        scores[i] = score
    return scores

def generate_dataset(input_dir, crop_description, crop_condition, crop_size, image_resize):
    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Generates classification dataset"):

        coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                            & (df_study_coordinates['condition'] == crop_condition)
                        ].to_dict('records')

        for coordinate in coordinates_dict:
            all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{coordinate['series_id']}/*.dcm"), key = lambda x : get_instance(x))
            if len(all_slices_path) < 10:
                print("No enought slices !")
                continue

            dataset_item = dict()

            gt_instance = coordinate["instance_number"]
            original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{coordinate['series_id']}/{gt_instance}.dcm").pixel_array.shape
            x = int((coordinate['x'] / original_shape[1]) * image_resize[1])
            y = int((coordinate['y'] / original_shape[0]) * image_resize[0])

            dataset_item['all_slices_path'] = all_slices_path
            dataset_item['position'] = (x, y)
            dataset_item['gt_instance'] = gt_instance
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
    def __init__(self, infos, normalisation, is_train=False):
        self.datas = infos
        self.is_train = is_train
        self.normalisation = normalisation

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        all_slices_path = data['all_slices_path']
        gt_instance = data['gt_instance']
        x, y = data["position"]
        crop_size = data['crop_size']
        image_resize = data['image_resize']

        for idx, sp in enumerate(all_slices_path):
            if get_instance(sp) == gt_instance:
                start_index = -1
                while not (start_index <= idx < start_index + 8 and start_index >= 0 and start_index + 8 <= len(all_slices_path)):
                    start_index = random.randint(max(0, idx - 7), min(len(all_slices_path) - 8, idx))
                slices = all_slices_path[start_index:start_index+8]
                
                crops = cut_crops(slices, x, y, crop_size, image_resize, self.normalisation)
                gt_index = None
                for k, s in enumerate(slices):
                    if get_instance(s) == gt_instance:
                        gt_index = k
                        break
                gt_scores = generate_scores(8, gt_index)
                break

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
            final_output = model(images.to(device))
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

        predictions = model(images.to(device))
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)
    return epoch_loss

def train_crop_selecter(input_dir, model_name, crop_description, crop_condition, crop_size, image_resize, normalisation):

    # get dataset
    dataset = generate_dataset(input_dir, crop_description, crop_condition, crop_size, image_resize)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True, normalisation=normalisation)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False, normalisation=normalisation)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = CropSelecter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss().to(device)
    best = 123456
    for epoch in range(50):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['loss'] < best:
            print("New best model !", model_name)
            best = metrics["loss"]
        print('-' * 50)
    return best

if __name__ == "__main__":
    train_crop_selecter(
                    input_dir="../",
                    model_name="",
                    crop_condition="Left Neural Foraminal Narrowing",
                    crop_description="Sagittal T1",
                    crop_size=(80, 120),
                    image_resize=(640, 640),
                    normalisation="clahe_norm_2"
    )
