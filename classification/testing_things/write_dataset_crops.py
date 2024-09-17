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
    for study_id in tqdm.tqdm(studies_id[:10], desc="Generates classification dataset"):

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

                                crops = cut_crops([dataset_item['slice_path']], x, y, crop_size, image_resize)

                                crop_name = f"{study_id}_{idx_level}_{int(gt_labels[idx_level])}.jpg"
                                cv2.imwrite(crop_name, crops[0] * 255)

                                dataset.append(dataset_item)

                        except Exception as e:
                            print("Error add item", e)
                            continue
    return dataset


if __name__ == "__main__":
    generate_dataset(
            input_dir="../../REFAIT/",
            crop_condition="Left Neural Foraminal Narrowing",
            label_condition="Left Neural Foraminal Narrowing",
            crop_description="Sagittal T1",
            crop_size=(64, 96),
            image_resize=(640, 640),
        )
