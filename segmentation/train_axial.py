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
import pickle
import concurrent.futures
from models_axial import LumbarSegmentationModel

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def generate_dataset(input_dir, condition, description):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    MASK_SIZE = (384, 384)
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
                                & (df_study_coordinates['condition'] == condition)
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            if len(coordinates_dict) != 5:
                continue

            # get slices to use
            all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))

            # get items
            for coordinate in coordinates_dict:
                original_idx = None
                for idx, sp in enumerate(all_slices_path):
                    if get_instance(sp) == coordinate['instance_number']:
                        original_idx = idx
                        break

                if original_idx > 2 and original_idx < len(all_slices_path) - 3:

                    dataset_item = dict()
                    dataset_item['all_slices'] = all_slices_path
                    dataset_item['original_idx'] = original_idx
                    dataset_item['output_mask'] = np.zeros((1, 384, 384))
                    dataset_item['labels_positions'] = [None, None]
                    dataset_item['gt_positions'] = [None, None]

                    instance = coordinate['instance_number']
                    x, y = coordinate['x'], coordinate['y']
                    original_slice = f"{input_dir}/train_images/{study_id}/{s_id}/{instance}.dcm"
                    original_shape = pydicom.dcmread(original_slice).pixel_array.shape
                    x = int((x / original_shape[1]) * MASK_SIZE[1])
                    y = int((y /original_shape[0]) * MASK_SIZE[0])
                    dataset_item['gt_positions'] = [x, y]
                    dataset_item['labels_positions'] = [x / MASK_SIZE[1], y / MASK_SIZE[0]]
                    dataset_item['output_mask'][0, y-17:y+17,x-17:x+17] = 1
                    dataset.append(dataset_item)
    return dataset


def load_dicom_series(dicom_files):
    dicom_files = sorted(dicom_files, key = lambda x : get_instance(x))
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices = [s.pixel_array for s in slices]
    slices = [cv2.resize(s, (384, 384), interpolation=cv2.INTER_LINEAR) for s in slices]
    slices = np.array(slices)
    slices = slices.transpose(1, 2, 0)
    return slices

def z_score_normalize_all_slices(scan):
    scan_flattened = scan.flatten()
    mean = np.mean(scan_flattened)
    std = np.std(scan_flattened)
    z_normalized_scan = (scan - mean) / std
    return z_normalized_scan

class SegmentationDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        
        original_idx = data['original_idx']
        all_slices_path = data['all_slices']

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
    
        volume = load_dicom_series(slices_path)
        normalised_volume = z_score_normalize_all_slices(volume)
        normalised_volume = normalised_volume.transpose(2, 0, 1)
        images = torch.tensor(normalised_volume).float()
        return images, torch.tensor(data['output_mask']).float(), torch.tensor(data['gt_positions']), torch.tensor(data['labels_positions']).float()


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

def validate(model, loader, criterion_mask, criterion_position, device, threshold=0.5):
    model.eval()
    total_mask_loss = 0
    total_pos_loss = 0
    total_metrics = {
        'loss': 0,
        "pos_loss": 0
    }

    metrics_position = {
        'mdx': list(),
        'mdy': list(),
        'pdx': list(),
        'pdy': list(),
        'adx': list(),
        'ady': list(),
    }
    nb_errors = 0

    with torch.no_grad():
        for images, gt_mask, mask_position, gt_position in tqdm.tqdm(loader, desc="Valid", total=len(loader)):

            images = images.to(device)
            gt_mask = gt_mask.to(device)
            gt_position = gt_position.to(device)
            pred_mask, pred_position = model(images)

            loss_mask = criterion_mask(pred_mask, gt_mask)
            loss_position = criterion_position(pred_position, gt_position)

            total_mask_loss += loss_mask.item()
            total_pos_loss += loss_position.item()
            
            _pred_positionX, _pred_positionY = pred_position.squeeze()

            _pred_positionX = _pred_positionX * 384
            _pred_positionY = _pred_positionY * 384

            mask_ = pred_mask[:, 0]
            mask_ = mask_.squeeze()
            pos_pred = find_center_of_largest_activation(mask_)
            if pos_pred is not None:
                predx, predy = pos_pred
            else:
                predx, predy = 1234, 1234
                nb_errors += 1
            gt_posx, gt_posy = mask_position.squeeze()
            metrics_position['mdx'].append(abs(predx - gt_posx.item()))
            metrics_position['mdy'].append(abs(predy - gt_posy.item()))

            metrics_position['pdx'].append(abs(_pred_positionX - gt_posx.item()))
            metrics_position['pdy'].append(abs(_pred_positionY - gt_posy.item()))

            avx = (_pred_positionX.item() + predx) / 2
            avy = (_pred_positionY.item() + predy) / 2
            metrics_position['adx'].append(abs(avx - gt_posx.item()))
            metrics_position['ady'].append(abs(avy - gt_posy.item()))

    metrics_position['mdx'] = sum(metrics_position['mdx']) / (len(metrics_position['mdx']) + 1e-9)
    metrics_position['mdy'] = sum(metrics_position['mdy']) / (len(metrics_position['mdy']) + 1e-9)
    metrics_position['pdx'] = sum(metrics_position['pdx']) / (len(metrics_position['pdx']) + 1e-9)
    metrics_position['pdy'] = sum(metrics_position['pdy']) / (len(metrics_position['pdy']) + 1e-9)
    metrics_position['adx'] = sum(metrics_position['adx']) / (len(metrics_position['adx']) + 1e-9)
    metrics_position['ady'] = sum(metrics_position['ady']) / (len(metrics_position['ady']) + 1e-9)
    metrics_position['md_total'] = (metrics_position['mdx'] + metrics_position['mdy']) / 2
    metrics_position['pd_total'] = (metrics_position['pdx'] + metrics_position['pdy']) / 2
    metrics_position['ad_total'] = (metrics_position['adx'] + metrics_position['ady']) / 2

    avg_mask_loss = total_mask_loss / len(loader)
    avg_pos_loss = total_pos_loss / len(loader)
    avg_metrics = dict()
    avg_metrics['mask_loss'] = avg_mask_loss
    avg_metrics['pos_loss'] = avg_pos_loss
    avg_metrics['mdx'] = metrics_position['mdx']
    avg_metrics['mdy'] = metrics_position['mdy']
    avg_metrics['mean_md'] = avg_metrics['mdx'] * 0.5 + avg_metrics['mdy'] * 0.5
    avg_metrics['pdx'] = metrics_position['pdx']
    avg_metrics['pdy'] = metrics_position['pdy']
    avg_metrics['mean_pd'] = avg_metrics['pdx'] * 0.5 + avg_metrics['pdy'] * 0.5
    avg_metrics['adx'] = metrics_position['adx']
    avg_metrics['ady'] = metrics_position['ady']
    avg_metrics['mean_ad'] = avg_metrics['adx'] * 0.5 + avg_metrics['ady'] * 0.5
    avg_metrics['errors_percent'] = nb_errors / len(loader)
    avg_metrics['nb_errors'] = nb_errors

    return avg_metrics


def train_epoch(model, loader, criterion_mask, criterion_position, optimizer, device):
    model.train()

    epoch_loss = 0
    for images, gt_mask, mask_position, gt_position in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()

        images = images.to(device)
        gt_mask = gt_mask.to(device)
        gt_position = gt_position.to(device)
        pred_mask, pred_position = model(images)

        loss_mask = criterion_mask(pred_mask, gt_mask)
        loss_position = criterion_position(pred_position, gt_position)
        
        loss = (loss_mask + loss_position) / 2
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / len(loader)

    return epoch_loss


def train_model_axial(input_dir, condition, description, out_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = generate_dataset(input_dir, condition, description)
    nb_valid = int(len(dataset) * 0.1)
    train_dataset = SegmentationDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = SegmentationDataset(dataset[:nb_valid])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4)

    model = LumbarSegmentationModel()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma = 0.5)
    criterion_mask = torch.nn.BCELoss()
    criterion_position = torch.nn.MSELoss()

    best_metrics = None
    best = 123456
    for epoch in range(20):
        loss_train = train_epoch(model, train_loader, criterion_mask, criterion_position, optimizer, device)
        metrics = validate(model, valid_loader, criterion_mask, criterion_position, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['mean_md'] < best:
            print("New best model !", out_name)
            best = metrics["mean_md"]
            best_metrics = metrics
            scripted_model = torch.jit.script(model)
            scripted_model.save(out_name)

        scheduler.step()
        print(scheduler.get_last_lr())
        print("-" * 55)

    return best_metrics

###################################################################
###################################################################
# MAIN TRAIN
###################################################################
###################################################################

def train_and_collect_metrics(cond, desc, out):
    print('-' * 50)
    print(f"Training: {cond}")
    best = train_model_axial(
        input_dir="../",
        condition=cond,
        description=desc,
        out_name=out
    )
    return best

conditions = ["Left Subarticular Stenosis", "Right Subarticular Stenosis"]
descriptions = ["Axial T2", "Axial T2"]
out_name = ["segmentation_ax_left.ts", "segmentation_ax_right.ts"]
metrics = {cond: [] for cond in conditions}

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future_to_cond = {executor.submit(train_and_collect_metrics, cond, desc, out): cond for cond, desc, out in zip(conditions, descriptions, out_name)}

    for future in concurrent.futures.as_completed(future_to_cond):
        cond = future_to_cond[future]
        try:
            best = future.result()
            metrics[cond].append(best)
            print(f"Metrics for {cond}: {best}")
        except Exception as exc:
            print(f"{cond} generated an exception: {exc}")

print("Done !")
print(metrics)