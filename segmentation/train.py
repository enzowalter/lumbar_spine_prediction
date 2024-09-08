import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torch
from scipy.ndimage import label, center_of_mass
import warnings
warnings.filterwarnings("ignore") # warning on lstm

from models import LumbarSegmentationModel

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def get_best_slice_selection(slice_model, pathes, device):
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
    slices_to_ret = list()
    for level in range(preds.shape[0]):
        pred_level = preds[level, :]
        _, max_indice = torch.topk(pred_level, 1)
        slices_to_ret.append(pathes[max_indice.item()])
    return slices_to_ret

def find_center_of_largest_activation(mask):
    mask = (mask > 0.5).float().detach().cpu().numpy()
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return None
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0
    largest_component_center = center_of_mass(labeled_mask == np.argmax(sizes))
    center_coords = tuple(map(int, largest_component_center))
    return (center_coords[1], center_coords[0]) # x, y

def generate_dataset(input_dir, conditions, description, slice_model):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    MASK_SIZE = (96, 96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Generates segmentation masks"):

        series_id = df_study_descriptions[(df_study_descriptions['study_id'] == study_id)
                                        & (df_study_descriptions['series_description'] == description)]['series_id'].to_list()
        
        for s_id in series_id:
            coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                                & (df_study_coordinates['condition'].isin(conditions))
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            # add to dataset only if all vertebraes in gt
            if len(coordinates_dict) == len(LEVELS):
                dataset_item = dict()
                dataset_item['study_id'] = study_id
                dataset_item['series_id'] = s_id
                dataset_item['all_slices_path'] = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))
                dataset_item['slice_per_level'] = get_best_slice_selection(slice_model, dataset_item['all_slices_path'], device)
                dataset_item['output_mask'] = np.zeros((len(LEVELS), *MASK_SIZE))
                dataset_item['gt_positions'] = [[None, None]] * len(LEVELS)

                for coordinate in coordinates_dict:
                    idx_level = LEVELS[coordinate['level']]
                    instance = coordinate['instance_number']
                    x, y = coordinate['x'], coordinate['y']

                    original_slice = f"{input_dir}/train_images/{study_id}/{s_id}/{instance}.dcm"
                    original_shape = pydicom.dcmread(original_slice).pixel_array.shape

                    x = int((x / original_shape[1]) * MASK_SIZE[1])
                    y = int((y /original_shape[0]) * MASK_SIZE[0])
                    dataset_item['gt_positions'][idx_level] = [x, y]
                    dataset_item['output_mask'][idx_level][y-5:y+5,x-5:x+5] = 1

                dataset.append(dataset_item)
    return dataset

class SegmentationDataset(Dataset):
    def __init__(self, infos):
        self.datas = infos

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        slices_path = data['slice_per_level']
        nb_slices = len(slices_path)
        images = np.zeros((nb_slices, 1, 224, 224))
        for k, path in enumerate(slices_path):
            im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                           (224, 224),
                                           interpolation=cv2.INTER_LINEAR)
            im = (im - im.min()) / (im.max() - im.min() + 1e-9)
            images[k, 0, ...] = im
        images = torch.tensor(images).expand(nb_slices, 3, 224, 224).float()

        return images, torch.tensor(data['output_mask']).float(), np.array(data['gt_positions'])  

def validate(model, loader, criterion, device):
    model.eval()

    valid_loss = 0
    metrics = [{
        'dx': list(),
        'dy': list(),
    } for _ in range(5)]
    nb_errors = 0

    for images, labels, gt_positions in tqdm.tqdm(loader, desc="Valid"):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)

            for level in range(5):
                mask_ = predictions[:, level]
                mask_ = mask_.squeeze()
                pos_pred = find_center_of_largest_activation(mask_)
                if pos_pred is not None:
                    predx, predy = pos_pred
                else:
                    predx, predy = 1234, 1234
                    nb_errors += 1

                gt_posx, gt_posy = gt_positions[:, level].squeeze()
                metrics[level]['dx'].append(abs(predx - gt_posx.item()))
                metrics[level]['dy'].append(abs(predy - gt_posy.item()))

            valid_loss += loss.item() / len(loader)

    for level in range(5):
        metrics[level]['dx'] = sum(metrics[level]['dx']) / (len(metrics[level]['dx']) + 1e-9)
        metrics[level]['dy'] = sum(metrics[level]['dy']) / (len(metrics[level]['dy']) + 1e-9)
        metrics[level]['d_total'] = (metrics[level]['dx'] + metrics[level]['dy']) / 2

    outputs = {
        "dx": sum(metrics[i]['dx'] for i in range(5)) / 5,
        "dy": sum(metrics[i]['dy'] for i in range(5)) / 5,
    }
    outputs['d_total'] = (outputs['dx'] + outputs['dy']) / 2
    outputs['errors_percent'] = nb_errors / (len(loader) * 5)
    outputs['nb_errors'] = nb_errors
    
    print("-" * 50)
    print("VALIDATION")
    for level in range(5):
        print(f"\t{level}")
        print(f"\t{metrics[level]}")
    print("-" * 50)
    return valid_loss, outputs

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()

    epoch_loss = 0
    for images, labels, _ in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()

        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss

def train_model(input_dir, conditions, description, slice_model_path, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slice_model = torch.jit.load(slice_model_path, map_location='cpu')
    slice_model = slice_model.eval()
    slice_model = slice_model.to(device)

    dataset = generate_dataset(input_dir, conditions, description, slice_model)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = SegmentationDataset(dataset[nb_valid:])
    valid_dataset = SegmentationDataset(dataset[:nb_valid])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = LumbarSegmentationModel()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss()

    best_metrics = None
    best = 123456
    for epoch in range(8):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device)
        loss_valid, metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "valid_loss=", loss_valid, "metrics=", metrics)
        if metrics['d_total'] < best:
            print("New best model !", model_name)
            best = metrics["d_total"]
            best_metrics = metrics
            model_scripted = torch.jit.script(model)
            model_scripted.save(model_name)
        print("-" * 55)
    
    print("-" * 55)
    print("-" * 55)
    print("-" * 55)
    print("DONE !")
    print("Model saved:", model_name)
    print("Best metrics:", best_metrics)
    print("-" * 55)
    print("-" * 55)
    print("-" * 55)
    return best_metrics