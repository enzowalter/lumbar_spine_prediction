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
                            dataset_item = dict()

                            # get original slices
                            dataset_item['slices_path'] = list()
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

                                    break

                            idx_level = LEVELS[coordinate['level']]
                            x, y = coordinate['x'], coordinate['y']
                            original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{s_id}/{coordinate['instance_number']}.dcm").pixel_array.shape
                            x = int((x / original_shape[1]) * image_resize[1]) 
                            y = int((y / original_shape[0]) * image_resize[0])

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
    angle = random.uniform(-10, 10)
    tensor_image = FT.rotate(tensor_image, angle)
    noise = torch.randn(tensor_image.size()) * 0.01
    tensor_image = tensor_image + noise
    return tensor_image

class CropClassifierDataset(Dataset):
    def __init__(self, infos, normalisation, is_train=False):
        self.datas = infos
        self.is_train = is_train
        self.normalisation = normalisation

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        slices_to_use = data['slices_path']
        x, y = data['position']

        if self.is_train:
            x += random.randint(-5, 6)
            y += random.randint(-5, 6)

        crops = cut_crops(slices_to_use, x, y, data['crop_size'], data['image_resize'], self.normalisation)
        crops = torch.tensor(crops).float()
        if self.is_train:
            crops = tensor_augmentations(crops)
        return crops, data['gt_label']

############################################################
############################################################
#          TRAINING
############################################################
############################################################

def validate_fold(models, loader, criterion, device):
    mm = list()
    for model in models:
        model.eval()
        model = model.cuda()
        mm.append(model)
    models = mm
    classification_loss_sum = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels_gt in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()
            
            final_output = torch.stack([model(images.to(device)) for model in models], dim=1)
            final_output = torch.mean(final_output, dim=1)

            all_predictions.append(final_output.cpu())
            all_labels.append(labels_gt.cpu())
            
            loss = criterion(final_output, labels_gt)
            classification_loss_sum += loss.item()

    all_predictions = torch.cat(all_predictions, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    
    concat_loss = criterion(all_predictions, all_labels).item()
    avg_classification_loss = classification_loss_sum / len(loader)
    return {"concat_loss": concat_loss, "mean_loss": avg_classification_loss}


def validate(model, loader, criterion, device):
    classification_loss_sum = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels_gt in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()


            final_output = model(images.to(device))
            
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
        
        predictions = model(images.to(device))
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss

def train_submodel(input_dir, model_name, crop_description, crop_condition, label_condition, crop_size, image_resize, normalisation):

    # get dataset
    dataset = generate_dataset(input_dir, crop_description, crop_condition, label_condition, crop_size, image_resize)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = CropClassifierDataset(dataset[nb_valid:], is_train=True, normalisation=normalisation)
    valid_dataset = CropClassifierDataset(dataset[:nb_valid], is_train=False, normalisation=normalisation)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # get model
    # model = FoldModelClassifier(
    #     n_classes=3,
    #     n_fold_classifier=3,
    #     backbones=['densenet201.tv_in1k', 'seresnext101_32x4d.gluon_in1k', 'convnext_base.fb_in22k_ft_in1k', 'dm_nfnet_f0.dm_in1k', 'mobilenetv3_small_100.lamb_in1k'],
    #     features_size=384,
    # )

    print("Test REM")

    """
    model = REM(
        n_classes=3,
        n_fold_classifier=3,
        backbones=['ese_vovnet39b.ra_in1k', 'cspresnet50.ra_in1k', 'mobilenetv3_small_100.lamb_in1k', 'ecaresnet26t.ra2_in1k', 'resnet26t.ra2_in1k'],
        features_size=256,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1, 2, 4]).float().to(device))
    best = 123456
    for epoch in range(10):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['concat_loss'] < best:
            print("New best model !", model_name)
            best = metrics["concat_loss"]
            #torch.save(model.state_dict(), model_name)
        print('-' * 50)
    """
        
    print("Test Folded")

    backbones=['ese_vovnet39b.ra_in1k', 'cspresnet50.ra_in1k', 'mobilenetv3_small_100.lamb_in1k', 'ecaresnet26t.ra2_in1k', 'resnet26t.ra2_in1k']
    models = list()
    for backbone in backbones:
        model = SimpleClassifier(
            encoder_name=backbone
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        tmp_name = "__tmp__.pth"

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1, 2, 4]).float().to(device))
        best = 123456
        for epoch in range(8):
            loss_train = train_epoch(model, train_loader, criterion, optimizer, device)
            metrics = validate(model, valid_loader, criterion, device)
            print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
            if metrics['concat_loss'] < best:
                print("New best model !", model_name)
                best = metrics["concat_loss"]
                torch.save(model.state_dict(), tmp_name)
            print('-' * 50)

        model = SimpleClassifier(
            encoder_name=backbone
        )
        model.load_state_dict(torch.load(tmp_name, map_location="cpu"))
        models.append(model)

    print("Testing with ensembling")
    metrics = validate_fold(models, valid_loader, criterion, device)
    print("ensembling metrics", metrics)


if __name__ == "__main__":

    best_logloss1 = train_submodel(
                    input_dir="../../REFAIT",
                    model_name="classification_spinal_canal_stenosis.pth",
                    crop_condition="Spinal Canal Stenosis",
                    label_condition="Spinal Canal Stenosis",
                    crop_description="Sagittal T2/STIR",
                    crop_size=(80, 120),
                    image_resize=(640, 640),
                    normalisation="clahe_norm_2"
    )

    """
    train_submodel(
                    input_dir="../../REFAIT",
                    model_name="classification_right_subarticular_stenosis_pipeline_dataset.pth",
                    crop_condition="Right Subarticular Stenosis",
                    label_condition="Right Subarticular Stenosis",
                    crop_description="Axial T2",
                    crop_size=(164, 164),
                    image_resize=(640, 640),
                    normalisation="clahe_norm_2"
    )
    train_submodel(
                    input_dir="../../REFAIT",
                    model_name="classification_left_subarticular_stenosis_pipeline_dataset.pth",
                    crop_condition="Left Subarticular Stenosis",
                    label_condition="Left Subarticular Stenosis",
                    crop_description="Axial T2",
                    crop_size=(164, 164),
                    image_resize=(640, 640),
                    normalisation="clahe_norm_2"
    )

    best_logloss1 = train_submodel(
                    input_dir="../../REFAIT",
                    model_name="classification_spinal_canal_stenosis.pth",
                    crop_condition="Spinal Canal Stenosis",
                    label_condition="Spinal Canal Stenosis",
                    crop_description="Sagittal T2/STIR",
                    crop_size=(80, 120),
                    image_resize=(640, 640),
    )

    best_logloss2 = train_submodel(
                    input_dir="../../REFAIT",
                    model_name="classification_left_neural_foraminal_narrowing.pth",
                    crop_condition="Left Neural Foraminal Narrowing",
                    label_condition="Left Neural Foraminal Narrowing",
                    crop_description="Sagittal T1",
                    crop_size=(64, 96),
                    image_resize=(640, 640),
    )
    best_logloss3 = train_submodel(
                    input_dir="../../REFAIT",
                    model_name="classification_right_neural_foraminal_narrowing.pth",
                    crop_condition="Right Neural Foraminal Narrowing",
                    label_condition="Right Neural Foraminal Narrowing",
                    crop_description="Sagittal T1",
                    crop_size=(64, 96),
                    image_resize=(640, 640),
    )
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
    """
