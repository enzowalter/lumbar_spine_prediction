import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torch
import warnings
warnings.filterwarnings("ignore") # warning on lstm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

from models import *

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
        slices_by_level[level] = [pathes[i.item()] for i in max_indice]
        #slices_by_level[level] = sorted([pathes[i.item()] for i in max_indice], key = lambda x : get_instance(x))
    return slices_by_level

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
                except Exception as e:
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
    output_crops = np.zeros((len(slices_path), *crop_size))
    for k, slice_path in enumerate(slices_path):
        pixel_array = pydicom.dcmread(slice_path).pixel_array.astype(np.float32)
        pixel_array = cv2.resize(pixel_array, image_resize, interpolation=cv2.INTER_LINEAR)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-9)
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # x y reversed in array
        output_crops[k, ...] = crop
    return output_crops

def generate_dataset(input_dir, conditions, description, slice_model, crop_size, image_resize):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}
    LABELS = {"Normal/Mild" : 0, "Moderate": 1, "Severe": 2}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Generates crops"):

        series_id = df_study_descriptions[(df_study_descriptions['study_id'] == study_id)
                                        & (df_study_descriptions['series_description'] == description)]['series_id'].to_list()
        
        for s_id in series_id:
            coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                                & (df_study_coordinates['condition'].isin(conditions))
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            # add to dataset only if all vertebraes in gt
            if len(coordinates_dict) == len(LEVELS):
                dataset_item = dict()
                all_slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))
                slice_per_level = get_best_slice_selection(slice_model, all_slices_path, topk=3, device=device)
                gt_labels = get_study_labels(study_id, df_study_labels, conditions[0], LEVELS, LABELS)
                if gt_labels is not None:
                    dataset_item['study_id'] = study_id
                    dataset_item['series_id'] = s_id
                    #dataset_item['crops'] = cut_crops(slice_per_level[idx_level], x, y, crop_size, image_resize)
                    dataset_item['slices_path'] = list()
                    dataset_item['positions'] = list()
                    dataset_item['gt_labels'] = gt_labels
                    dataset_item['crop_size'] = crop_size
                    dataset_item['image_resize'] = image_resize

                    has_error = False
                    for coordinate in coordinates_dict:
                        try:
                            level_int = LEVELS[coordinate['level']]
                            x, y = coordinate['x'], coordinate['y']
                            original_shape = pydicom.dcmread(f"{input_dir}/train_images/{study_id}/{s_id}/{coordinate['instance_number']}.dcm").pixel_array.shape
                            x = int((x / original_shape[1]) * image_resize[0]) 
                            y = int((y / original_shape[0]) * image_resize[1])
                            dataset_item['positions'].append((x, y))
                            dataset_item['slices_path'].append(slice_per_level[level_int])
                        except Exception as e:
                            print("Error", e)
                            has_error = True
                            continue
                    if has_error is False:
                        dataset.append(dataset_item)

    return dataset

class ClassificationDataset(Dataset):
    def __init__(self, infos: list, is_train: bool):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]

        crop_size = data['crop_size']
        image_resize = data['image_resize']

        crops_tensor = torch.zeros(5, 3, *crop_size)

        for level in range(5):
            slices_to_use = data['slices_path'][level]
            position = data['positions'][level]
            x, y = position
            dx = torch.randint(-7, 8, (1,)).item()
            dy = torch.randint(-7, 8, (1,)).item()

            x += dx
            y += dy

            crops = cut_crops(slices_to_use, x, y, crop_size, image_resize)[0].astype(np.float32)
            crops_tensor[level] = torch.tensor(crops)

        return crops_tensor.float(), data['gt_labels']

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

    with torch.no_grad():
        for im1, labels_gt in tqdm.tqdm(loader, desc="Valid"):
            labels_gts = labels_gt.to(device).long()
            final_outputs = model(im1.to(device), mode="valid")            
            final_outputs = final_outputs.reshape(final_outputs.shape[0], 5, 3)
            loss = 0
            for i in range(final_outputs.shape[1]):
                final_output = final_outputs[:, i]
                labels_gt = labels_gts[:, i]

                loss += criterion(final_output, labels_gt) / final_outputs.shape[1]
                log_loss_batch = calculate_log_loss(final_output, labels_gt, weights)
                total_log_loss += log_loss_batch.sum().item()
                total_samples += labels_gt.size(0)

            classification_loss_sum += loss.item()

    avg_classification_loss = classification_loss_sum / len(loader)
    avg_log_loss = total_log_loss / total_samples

    print(f"Average Log Loss: {avg_log_loss:.4f}")
    return {
        "loss_classi": avg_classification_loss,
        "log_loss": avg_log_loss
    }

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, labels in tqdm.tqdm(loader, desc="Training", total=len(loader)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device).long()

        predictions = model(images, mode="train")
        predictions = predictions.reshape(predictions.shape[0], 5, 3)
        loss = 0
        for i in range(predictions.shape[1]):
            loss += criterion(predictions[:, i], labels[:, i]) / predictions.shape[1]
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
    train_dataset = ClassificationDataset(dataset[nb_valid:], is_train=True)
    valid_dataset = ClassificationDataset(dataset[:nb_valid], is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    model = EfficientNetClassifierFoldFromSeries(
        n_classes=15,
        n_fold_classifier=3,
        seq_lenght=5,
        backbones=["squeezenet", "convnext", "efficientnet", "mobilenet", "resnet", "densenet"],
        features_size=256,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1/7, 2/7, 4/7]).to(device))

    best_metrics = None
    best = 123456
    for epoch in range(20):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "train_loss=", loss_train, "metrics=", metrics)
        if metrics['log_loss'] < best:
            print("New best model !", model_name)
            best = metrics["log_loss"]
            #torch.save(model.state_dict(), model_name)

        print("-" * 55)
    
    print("-" * 55)
    print("-" * 55)
    print("-" * 55)
    print("DONE !")
    print("Model saved:", model_name)
    print("Best metrics:", best)
    print("-" * 55)
    print("-" * 55)
    print("-" * 55)

    torch.cuda.empty_cache()
    return best_metrics


if __name__ == "__main__":
    m1 = train_model(
                "../../REFAIT",
                ["Spinal Canal Stenosis"],
                "Sagittal T2/STIR",
                "../trained_models/v0/model_slice_selection_st2.ts",
                (96, 128),
                (600, 600),
                f"model_classification_st2_v2.ts",
            ) # 0.3300
    