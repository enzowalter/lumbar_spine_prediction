import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import concurrent.futures
import glob
import torch
import warnings
warnings.filterwarnings("ignore") # warning on lstm

from models_v2 import *

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

def cut_crops(slices_path, x, y, crop_size, image_resize, flip):
    output_crops = np.zeros((len(slices_path), 128, 128))
    for k, slice_path in enumerate(slices_path):
        pixel_array = pydicom.dcmread(slice_path).pixel_array.astype(np.float32)
        pixel_array = cv2.resize(pixel_array, image_resize, interpolation=cv2.INTER_LINEAR)
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # x y reversed in array
        if flip:
            crop = cv2.flip(crop, 1)
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

############################################################
############################################################
#          GEN DATASET
############################################################
############################################################

def generate_dataset(input_dir, crop_condition, crop_size, image_resize):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")

    datasets = list()
    nb_label_none = 0
    nb_bad_pos = 0
    
    for condition in crop_condition:
        all_coordinates = df_study_coordinates[df_study_coordinates['condition'] == condition].to_dict('records')
        dataset = list()
        for coordinate in tqdm.tqdm(all_coordinates, desc=f"Parsing coordinates for {condition}..."):

            s_id = coordinate['series_id']
            study_id = coordinate['study_id']
            level_str = coordinate['level'].lower().replace('/', '_')
            column_name = condition.lower().replace(' ', '_') + "_" + level_str
            label_str = df_study_labels[
                (df_study_labels['study_id'] == study_id)
            ][column_name].item()
            label_int = LABELS[label_str] if label_str in LABELS else None

            if label_int is None:
                nb_label_none += 1
                continue

            all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))

            original_idx = None
            for idx, sp in enumerate(all_slices_path):
                if get_instance(sp) == coordinate['instance_number']:
                    original_idx = idx
                    break

            x, y = coordinate['x'], coordinate['y']
            original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{s_id}/{coordinate['instance_number']}.dcm").pixel_array.shape
            x = int((x / original_shape[1]) * image_resize[1]) 
            y = int((y / original_shape[0]) * image_resize[0])

            dataset_item = dict()
            dataset_item['study_id'] = study_id
            dataset_item['condition'] = condition
            dataset_item['all_slices'] = all_slices_path
            dataset_item['original_index'] = original_idx
            dataset_item['series_id'] = s_id
            dataset_item['position'] = (x, y)
            dataset_item['gt_label'] = label_int
            dataset_item['crop_size'] = crop_size
            dataset_item['image_resize'] = image_resize

            dataset.append(dataset_item)

        datasets.append(dataset)

    print(nb_bad_pos, "items with bad index position", nb_label_none, "items with label none")
    return datasets


############################################################
############################################################
#          DATALOADER
############################################################
############################################################

def create_soft_labels_at_index(center_idx):
    weights_crops = np.array([0, 0, 0, 0, 0])
    weights = [0.025, 0.2, 2, 0.2, 0.025]
    soft_labels = np.zeros_like(weights_crops, dtype=float)
    for i, weight in enumerate(weights):
        offset = i - 2
        if 0 <= center_idx + offset < len(weights_crops):
            soft_labels[center_idx + offset] = weight
    return soft_labels

def z_score_normalize_all_slices(scan):
    scan_flattened = scan.flatten()
    mean = np.mean(scan_flattened)
    std = np.std(scan_flattened)
    z_normalized_scan = (scan - mean) / std
    return z_normalized_scan

def get_soft_data_transforms():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    ])

class CropClassifierDataset(Dataset):
    def __init__(self, infos, flip_right, weight_labels, is_train=False):
        self.datas = infos
        self.is_train = is_train
        self.flip_right = flip_right
        self.transforms = get_soft_data_transforms()

        nb_normal = len([x for x in self.datas if x['gt_label'] == 0])
        nb_moderate = len([x for x in self.datas if x['gt_label'] == 1])
        nb_severe = len([x for x in self.datas if x['gt_label'] == 2])

        if is_train == True:
            print("nb_normal", nb_normal)
            print("nb_moderate", nb_moderate)
            print("nb_severe", nb_severe)

        if weight_labels:
            print('-' * 50)
            print("Sampling dataset")

            nb_normal = len([x for x in self.datas if x['gt_label'] == 0])
            nb_moderate = len([x for x in self.datas if x['gt_label'] == 1])
            nb_severe = len([x for x in self.datas if x['gt_label'] == 2])

            target_severe_count = nb_severe
            target_moderate_count = nb_moderate
            target_normal_count = max(nb_moderate * 2, nb_severe * 4)
            
            normal_samples = [x for x in self.datas if x['gt_label'] == 0]
            moderate_samples = [x for x in self.datas if x['gt_label'] == 1]
            severe_samples = [x for x in self.datas if x['gt_label'] == 2]

            selected_normal = random.sample(normal_samples, target_normal_count)
            selected_moderate = random.sample(moderate_samples, target_moderate_count)
            selected_severe = random.sample(severe_samples, target_severe_count)

            self.datas = selected_normal + selected_moderate + selected_severe
            random.shuffle(self.datas)
            
            print(f"Selected labels to fit 1, 2, 4 distribution: {len(selected_normal)} normal, {len(selected_moderate)} moderate {len(selected_severe)} severe !")
            print('-' * 50)


        print(f"Nb items in dataset: {len(self.datas)}")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        x, y = data['position']
        original_idx = data['original_index']
        all_slices_path = data['all_slices']

        index_offsets = [-2, -1, 0, 1, 2]
        weights = [0.025, 0.175, 0.6, 0.175, 0.025]

        valid_offsets = []
        valid_weights = []
        for offset, weight in zip(index_offsets, weights):
            offset_index = original_idx + offset
            if 0 <= offset_index - 2 and offset_index + 2 < len(all_slices_path):
                valid_offsets.append(offset)
                valid_weights.append(weight)

        if len(valid_offsets) == 0:
            raise ValueError(f"No valid offsets available for original index {original_idx} "
                            f"with slice count {len(all_slices_path)}.")


        valid_weights = np.array(valid_weights) / np.sum(valid_weights)
        offset = np.random.choice(valid_offsets, p=valid_weights)
        offset_index = original_idx + offset

        slices_path = [
            all_slices_path[offset_index - 2],
            all_slices_path[offset_index - 1],
            all_slices_path[offset_index],
            all_slices_path[offset_index + 1],
            all_slices_path[offset_index + 2]
        ]

        if self.is_train:
            x += random.randint(-8, 9)
            y += random.randint(-8, 9)

        good_slice_idx = slices_path.index(all_slices_path[original_idx])
        if self.flip_right:
            crops = cut_crops(slices_path, x, y, data['crop_size'], data['image_resize'], flip = "Right" in data['condition'])
        else:
            crops = cut_crops(slices_path, x, y, data['crop_size'], data['image_resize'], flip = False)
        crops = z_score_normalize_all_slices(crops)
        crops = torch.tensor(crops).float()
        crops = crops.unsqueeze(1).expand(5, 3, 128, 128)

        # if self.is_train:
        #     crops = self.transforms(crops)

        weights_crops = create_soft_labels_at_index(good_slice_idx)
        weights_crops = torch.softmax(torch.tensor(weights_crops), dim=0)

        return crops, data['gt_label'], torch.tensor(good_slice_idx), weights_crops

############################################################
############################################################
#          TRAINING
############################################################
############################################################

def validate(model, loader, criterion, device):
    print("-" * 50)
    model.eval()
    classification_loss_sum = 0
    all_predictions = []
    all_labels = []
    good_crops_selection = 0
    total_loss_weight = 0

    with torch.no_grad():
        for images, labels_gt, good_slice_idx, gt_weights in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()
            good_slice_idx = good_slice_idx.to(device).long()
            gt_weights = gt_weights.to(device).float()
            
            outputs, crops_weight, selected_crops = model(images.to(device), mode = "inference")
            
            weights_loss = 0
            for i in range(model.nb_encoders):
                good_crops_selection += (selected_crops[:, i] == good_slice_idx).float().sum()
                weights_loss += nn.BCELoss()(crops_weight[:, i], gt_weights)

            total_loss_weight += weights_loss.mean().item()

            all_predictions.append(outputs.cpu())
            all_labels.append(labels_gt.cpu())
            
            loss = criterion(outputs, labels_gt)
            classification_loss_sum += loss.item()

    all_predictions = torch.cat(all_predictions, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    
    concat_loss = criterion(all_predictions, all_labels).item()
    avg_classification_loss = classification_loss_sum / len(loader)
    print("Accuracy on crop selection:", good_crops_selection / (len(loader) * model.nb_encoders * 16))
    print("Loss on weights:", total_loss_weight / (len(loader)))
    return {"concat_loss": concat_loss, "mean_loss": avg_classification_loss}


def validate_metamodel(model, loader, criterion, device):
    print("-" * 50)
    model.eval()
    classification_loss_sum = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels_gt, good_slice_idx, gt_weights in tqdm.tqdm(loader, desc="Valid"):
            labels_gt = labels_gt.to(device).long()
            good_slice_idx = good_slice_idx.to(device).long()
            gt_weights = gt_weights.to(device).float()
            
            outputs = model(images.to(device))

            all_predictions.append(outputs.cpu())
            all_labels.append(labels_gt.cpu())
            
            loss = criterion(outputs, labels_gt)
            classification_loss_sum += loss.item()

    all_predictions = torch.cat(all_predictions, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    
    concat_loss = criterion(all_predictions, all_labels).item()
    avg_classification_loss = classification_loss_sum / len(loader)
    return {"concat_loss": concat_loss, "mean_loss": avg_classification_loss}

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    print("-" * 50)

    model.train()
    epoch_loss = 0
    for images, labels, good_slice_idx, gt_weights in tqdm.tqdm(loader, desc="Training", total=len(loader)):

        images = images.to(device).float()
        labels = labels.to(device).long()
        good_slice_idx = good_slice_idx.to(device).long()
        gt_weights = gt_weights.to(device).float()
        
        outputs, crops_weights, _ = model(images, mode = "train")
        weights_loss = nn.BCELoss()(crops_weights, gt_weights)
    
        classification_loss = criterion(outputs, labels)
        total_loss = classification_loss + weights_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item() / len(loader)


    return epoch_loss


def get_loaders(datasets, flip_right):
    train_datasets = []
    valid_datasets = []

    n_items = [0, 0, 0]
    for dataset in datasets:
        for item in dataset:
            n_items[item['gt_label']] += 1

    weights = [0, 0, 0]
    for i in range(3):
        weights[i] = sum(n_items) / n_items[i]

    for dataset in datasets:
        nb_valid = int(len(dataset) * 0.1)
        train_dataset = CropClassifierDataset(dataset[nb_valid:], flip_right, weight_labels=False, is_train=True)
        valid_dataset = CropClassifierDataset(dataset[:nb_valid], flip_right, weight_labels=False, is_train=False)
        valid_datasets.append(valid_dataset)
        train_datasets.append(train_dataset)
    
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=3)
    valid_loader = DataLoader(valid_dataset, batch_size=16, num_workers=3)

    return train_loader, valid_loader, weights

def train_submodel(model_name, flip_right, datasets):
    print()
    print("-" * 50)
    print("-" * 50)
    print("-" * 50)
    print("-" * 50)
    print("TRAINING", model_name)
    print("-" * 50)
    print("-" * 50)
    print("-" * 50)
    print("-" * 50)

    train_loader, valid_loader, class_weights = get_loaders(datasets, flip_right)

    backbones = [
        'hgnet_tiny.paddle_in1k', 
        'densenet201.tv_in1k', 
        'regnety_008.pycls_in1k', 
        'focalnet_tiny_lrf.ms_in1k', 
        'convnext_base.fb_in22k_ft_in1k',
        'seresnext101_32x4d.gluon_in1k',
    ]

    model = REM(
        n_classes=3,
        n_classifiers=3,
        unification_size=768,
        backbones=backbones,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)
    criterion_train = torch.nn.CrossEntropyLoss(weight = torch.tensor(class_weights).float().to(device))
    criterion_valid = torch.nn.CrossEntropyLoss(weight = torch.tensor([1, 2, 4]).float().to(device))
    best = 123456

    for epoch in range(20):
        loss_train = train_epoch(model, train_loader, criterion_train, optimizer, device, epoch)
        metrics = validate(model, valid_loader, criterion_valid, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['concat_loss'] < best:
            print("New best model !")
            best = metrics["concat_loss"]
            torch.save(model.state_dict(), model_name)

        print('-' * 50)
        scheduler.step()
        print(scheduler.get_last_lr())

    return best

if __name__ == "__main__":

    conditions = [
        ['Spinal Canal Stenosis'],
        ["Left Neural Foraminal Narrowing"], 
        ["Right Neural Foraminal Narrowing"],
        ["Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    ]
    crop_sizes = [(80, 120), (80, 120), (80, 120), (128, 128)]
    out_name = ["trained_realweightloss/classification_st2", "trained_realweightloss/classification_st1_left", "trained_realweightloss/classification_st1_right", "trained_realweightloss/classification_axial"]
    use_flip = [False, False, False, True]

    metrics = dict()
    def train_and_collect_metrics(model_name, flip, datasets, step):
        print('-' * 50)
        print(f"Training: {cond}")
        best = train_submodel(
            model_name=model_name,
            flip_right=flip,
            datasets = datasets,
        )
        return step, best

    for cond, cs, out, flip in zip(conditions, crop_sizes, out_name, use_flip):

        metrics[str(cond)] = []
        datasets = generate_dataset("../", cond, cs, (640, 640))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            print("?")
            future_to_step = {executor.submit(train_and_collect_metrics, f"{out}_step_{step}.pth", flip, datasets, step): step for step in range(1)}
            
            for future in concurrent.futures.as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    step_num, best = future.result()
                    metrics[str(cond)].append((step_num, best))
                    print(f"Metrics for {cond}, step {step_num}: {best}")
                except Exception as exc:
                    print(f"{cond}, step {step} generated an exception: {exc}")

    print("Done!")
    print("All metrics:", metrics)
