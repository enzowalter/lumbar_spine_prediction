import numpy as np
import pandas as pd
import tqdm
import cv2
import pydicom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torch
import pickle
import pathlib
import shutil
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter, affine_transform, map_coordinates
import random
import concurrent.futures

from models import *

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def create_soft_labels(hard_labels, radius=2, sigma=1.0, max_value=1.0):
    soft_labels = np.zeros_like(hard_labels, dtype=float)
    for idx, label in enumerate(hard_labels):
        if label == 1:
            for i in range(-radius, radius + 1):
                if 0 <= idx + i < len(hard_labels):
                    soft_labels[idx + i] = max_value * np.exp(-0.5 * (i / sigma) ** 2)
    return soft_labels


def generate_scores(num_images, index):
    scores = np.zeros(num_images)
    scores[index] = 1
    return create_soft_labels(scores)

def resize_slices_to_224(volume):
    num_slices = volume.shape[-1]
    resized_slices = []
    for i in range(num_slices):
        slice_ = volume[:, :, i]
        resized_slice = cv2.resize(slice_, (224, 224), interpolation=cv2.INTER_LINEAR)
        resized_slices.append(resized_slice)
    resized_volume = np.stack(resized_slices, axis=-1)
    return resized_volume

def load_dicom_series(dicom_files):
    dicom_files = sorted(dicom_files, key = lambda x : get_instance(x))
    slices = [pydicom.dcmread(f) for f in dicom_files]
    slices = [s.pixel_array for s in slices]
    slices = [cv2.resize(s, (224, 224), interpolation=cv2.INTER_LINEAR) for s in slices]
    slices = np.array(slices)
    slices = slices.transpose(1, 2, 0)
    return slices

def z_score_normalize_all_slices(scan):
    mean = np.mean(scan)
    std = np.std(scan)
    z_normalized_scan = (scan - mean) / std
    return z_normalized_scan

def generate_dataset(input_dir, condition, description, scores_fct):

    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}

    df_study_labels = pd.read_csv(f"{input_dir}/train.csv")
    df_study_coordinates = pd.read_csv(f"{input_dir}/train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    dataset = list()
    # keep 100 for tests
    for study_id in tqdm.tqdm(studies_id[:180], desc="Generates slices hotmap"):

        series_id = df_study_descriptions[(df_study_descriptions['study_id'] == study_id)
                                        & (df_study_descriptions['series_description'] == description)]['series_id'].to_list()
        
        for s_id in series_id:
            coordinates_dict = df_study_coordinates[(df_study_coordinates['study_id'] == study_id)
                                & (df_study_coordinates['condition'] == condition)
                                & (df_study_coordinates['series_id'] == s_id)].to_dict('records')

            # add to dataset only if all vertebraes in series
            if len(coordinates_dict) == len(LEVELS):
                dataset_item = dict()
                dataset_item['study_id'] = study_id
                dataset_item['series_id'] = s_id
                dataset_item['slices_path'] = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))

                dataset_item['nb_slices'] = len(dataset_item['slices_path'])
                dataset_item['labels'] = np.zeros((len(LEVELS), len(dataset_item['slices_path'])))
                dataset_item['gt_indices'] = np.zeros(len(LEVELS))

                # Fill GT
                for coordinate in coordinates_dict:
                    idx_level = LEVELS[coordinate['level']]
                    instance = coordinate['instance_number']
                    instance_index = None
                    for i, slice_path in enumerate(dataset_item['slices_path']):
                        if get_instance(slice_path) == instance:
                            instance_index = i
                            break
                    if instance_index is None:
                        raise Exception("Error in GT !")
                    dataset_item['gt_indices'][idx_level] = instance_index
                    dataset_item['labels'][idx_level] = scores_fct(dataset_item['nb_slices'], instance_index)

                if len(dataset_item['slices_path']) > 10 and len(dataset_item['slices_path']) < 60: # otherwise gpu explodes
                    dataset.append(dataset_item)

    return dataset

class SlicePredicterDataset(Dataset):
    def __init__(self, infos, is_train=False):
        self.datas = infos
        self.is_train = is_train

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        labels = data['labels']
        slices = data['slices_path']
        volume = load_dicom_series(slices)
        normalised_volume = z_score_normalize_all_slices(volume)

        normalised_volume = normalised_volume.transpose(2, 0, 1)
        images = torch.tensor(normalised_volume).unsqueeze(1).float()
        labels = torch.tensor(labels).float()
        return images, labels, np.array(data['gt_indices'])

def get_max_consecutive(preds, gt_indice, n):
    max_sum = -float('inf')
    max_idx = -1
    
    for i in range(len(preds) - n + 1):
        current_sum = preds[i:i + n].sum().item()
        if current_sum > max_sum:
            max_sum = current_sum
            max_idx = i
    
    max_consecutive_indices = list(range(max_idx, max_idx + n))
    is_gt_in_max = gt_indice in max_consecutive_indices
    
    return is_gt_in_max

def validate(model, loader, criterion, device):
    model.eval()

    valid_loss = 0
    tops = {
        1: 0,
        3: 0,
        5: 0,
        10: 0,
    }
    for images, labels, gt_indices in tqdm.tqdm(loader, desc="Valid"):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            predictions, _ = model(images)
            loss = criterion(predictions, labels)
            
            for level in range(5):
                preds = predictions[:, level]
                gt_indice = gt_indices[:, level].item()

                for top in tops: 
                    tops[top] += 1 if get_max_consecutive(preds.squeeze(0), gt_indice, top) else 0

            valid_loss += loss.item() / len(loader)
    return valid_loss, {t: tops[t] / (len(loader) * 5) for t in tops}

def validate_metamodel(model, loader, criterion, device):
    model.eval()

    valid_loss = 0
    tops = {
        1: 0,
        3: 0,
        5: 0,
        10: 0,
    }
    for images, labels, gt_indices in tqdm.tqdm(loader, desc="Valid"):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            for level in range(5):
                preds = predictions[:, level]
                gt_indice = gt_indices[:, level].item()

                for top in tops: 
                    tops[top] += 1 if get_max_consecutive(preds.squeeze(0), gt_indice, top) else 0

            valid_loss += loss.item() / len(loader)
    return valid_loss, {t: tops[t] / (len(loader) * 5) for t in tops}


def train_epoch(model, loader, criterion, optimizer, accumulation_step, device):
    model.train()

    epoch_loss = 0
    optimizer.zero_grad()
    for step, (images, labels, gt_indices) in tqdm.tqdm(enumerate(loader), desc="Training", total=len(loader)):
        images = images.to(device)
        labels = labels.to(device)

        # predictions = model(images)
        predictions, predictions_logits = model(images)

        #loss = criterion(predictions, labels)
        loss = criterion(predictions, labels) / accumulation_step
        loss.backward()
        epoch_loss += loss.item() / len(loader)

        if (step + 1) % accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    if (step + 1) % accumulation_step != 0:
        optimizer.step()
        optimizer.zero_grad()

    return epoch_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = torch.nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def train_model(input_dir, condition, description, scores_fct, step):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = generate_dataset(input_dir, condition, description, scores_fct)

    # select validation and training set
    
    nb_valid = len(dataset) // 4
    valid_indices = list(range(step * nb_valid, (step + 1) * nb_valid))
    all_indices = list(range(len(dataset)))
    train_indices = list(set(all_indices) - set(valid_indices))
    
    train_dataset = SlicePredicterDataset([dataset[i] for i in train_indices], is_train=True)
    valid_dataset = SlicePredicterDataset([dataset[i] for i in valid_indices])

    # batch size = 1 because no padding on sequences
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4)

    model = SliceSelecterModelTransformer()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = FocalLoss(alpha=1, gamma=2)

    best_metrics = None

    current_name = f"trained_models/{condition.lower().replace(' ', '_')}_step_{step}.pth"

    best = -1
    for epoch in range(10):
        loss_train = train_epoch(model, train_loader, criterion, optimizer, 8, device)
        loss_valid, instance_accuracy = validate(model, valid_loader, criterion, device)
        print("Epoch", epoch, "step", step, "train_loss=", loss_train, "valid_loss=", loss_valid, "instance_accuracy=", instance_accuracy)
        if instance_accuracy[3] > best:
            print("New best model !")
            best = instance_accuracy[3]
            best_metrics = instance_accuracy
            torch.save(model.state_dict(), current_name)
        print("-" * 55)

    return best_metrics

if __name__ == "__main__":

    conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Spinal Canal Stenosis", "Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR", "Axial T2", "Axial T2"]
    out_name = ["slice_selector_st1_left_bs16.ts", "slice_selector_st1_right_bs16.ts", "slice_selector_st2_bs16.ts", "slice_selector_ax_left_bs16.ts", "slice_selector_ax_right_bs16.ts"]
    scores_fct = [generate_scores, generate_scores, generate_scores, generate_scores, generate_scores]

    # All metrics: {
    #     'Left Neural Foraminal Narrowing': [
    #         (0, {1: 0.6345528455284553, 3: 0.9841463414634146, 5: 0.9922764227642277, 10: 0.9995934959349594}), 
    #         (1, {1: 0.6239837398373984, 3: 0.9800813008130081, 5: 0.9910569105691057, 10: 0.9963414634146341}), 
    #         (2, {1: 0.6426829268292683, 3: 0.983739837398374, 5: 0.9922764227642277, 10: 0.9947154471544716}), 
    #         (3, {1: 0.6524390243902439, 3: 0.9796747967479674, 5: 0.990650406504065, 10: 0.9959349593495935})], 
    #     'Right Neural Foraminal Narrowing': [
    #         (0, {1: 0.6105691056910569, 3: 0.9760162601626017, 5: 0.9898373983739838, 10: 0.9963414634146341}), 
    #         (1, {1: 0.6565040650406504, 3: 0.9861788617886179, 5: 0.990650406504065, 10: 0.9967479674796748}), 
    #         (2, {1: 0.6495934959349593, 3: 0.9869918699186991, 5: 0.9926829268292683, 10: 0.9955284552845528}), 
    #         (3, {1: 0.6260162601626016, 3: 0.9841463414634146, 5: 0.9943089430894309, 10: 0.9979674796747967})], 
    #     'Spinal Canal Stenosis': [
    #         (0, {1: 0.7749469214437368, 3: 0.9936305732484076, 5: 0.9991507430997877, 10: 1.0}), 
    #         (1, {1: 0.7609341825902336, 3: 0.9961783439490446, 5: 0.9983014861995754, 10: 0.9995753715498938}), 
    #         (3, {1: 0.7609341825902336, 3: 0.9953290870488323, 5: 0.9983014861995754, 10: 1.0}), 
    #         (2, {1: 0.7762208067940553, 3: 0.9949044585987261, 5: 0.9983014861995754, 10: 0.9995753715498938})], 
    #     'Left Subarticular Stenosis': [], 'Right Subarticular Stenosis': []
    #     }
    # 'Left Subarticular Stenosis': [
    #     (0, {1: 0.5843434343434344, 3: 0.9444444444444444, 5: 0.9651515151515152, 10: 0.9939393939393939}), (
    #         1, {1: 0.5767676767676768, 3: 0.9318181818181818, 5: 0.9505050505050505, 10: 0.9914141414141414}), 
    #         (2, {1: 0.5898989898989899, 3: 0.946969696969697, 5: 0.9626262626262626, 10: 0.9924242424242424}), 
    #         (3, {1: 0.591919191919192, 3: 0.9454545454545454, 5: 0.9651515151515152, 10: 0.9934343434343434})], 
    # 'Right Subarticular Stenosis': [
    #     (0, {1: 0.6326633165829145, 3: 0.9522613065326633, 5: 0.964321608040201, 10: 0.9964824120603015}), 
    #     (1, {1: 0.5959798994974874, 3: 0.942713567839196, 5: 0.9577889447236181, 10: 0.9969849246231156}), 
    #     (2, {1: 0.6060301507537689, 3: 0.9381909547738694, 5: 0.9577889447236181, 10: 0.9899497487437185}), 
    #     (3, {1: 0.6231155778894473, 3: 0.9492462311557789, 5: 0.9628140703517588, 10: 0.9919597989949749})]


    # v0
    # All metrics: {
    #     'Left Neural Foraminal Narrowing': [
    #         (1, {1: 0.6115631691648822, 3: 0.974304068522484, 5: 0.9888650963597431, 10: 0.9948608137044967}),
    #         (0, {1: 0.5944325481798716, 3: 0.9773019271948609, 5: 0.9880085653104925, 10: 0.9974304068522484}), 
    #         (2, {1: 0.6077087794432549, 3: 0.9768736616702356, 5: 0.9922912205567452, 10: 0.9948608137044967}), 
    #         (3, {1: 0.6154175588865096, 3: 0.9798715203426124, 5: 0.9905781584582442, 10: 0.9961456102783726})],
    #     'Right Neural Foraminal Narrowing': [
    #         (0, {1: 0.6111349036402569, 3: 0.9747323340471092, 5: 0.9922912205567452, 10: 0.9965738758029978}), 
    #         (1, {1: 0.6252676659528907, 3: 0.9768736616702356, 5: 0.987152034261242, 10: 0.995289079229122}), 
    #         (2, {1: 0.6059957173447538, 3: 0.9828693790149893, 5: 0.9927194860813704, 10: 0.9970021413276231}), 
    #         (3, {1: 0.6059957173447538, 3: 0.9725910064239829, 5: 0.9901498929336189, 10: 0.9965738758029978})], 
    #     'Spinal Canal Stenosis': [
    #         (1, {1: 0.756152125279642, 3: 0.9959731543624161, 5: 0.9968680089485459, 10: 0.9995525727069351}), 
    #         (0, {1: 0.774496644295302, 3: 0.9968680089485459, 5: 0.9995525727069351, 10: 1.0}), 
    #         (2, {1: 0.7659955257270693, 3: 0.992841163310962, 5: 0.9986577181208054, 10: 0.9995525727069351}), 
    #         (3, {1: 0.7395973154362416, 3: 0.9937360178970918, 5: 0.9982102908277405, 10: 1.0})], 
    #     'Left Subarticular Stenosis': [
    #         (0, {1: 0.5367021276595745, 3: 0.948404255319149, 5: 0.9664893617021276, 10: 0.9925531914893617}), 
    #         (1, {1: 0.5648936170212766, 3: 0.9329787234042554, 5: 0.9462765957446808, 10: 0.9920212765957447}), 
    #         (2, {1: 0.5595744680851064, 3: 0.950531914893617, 5: 0.9686170212765958, 10: 0.9925531914893617}), 
    #         (3, {1: 0.5484042553191489, 3: 0.9398936170212766, 5: 0.9579787234042553, 10: 0.9941489361702127})], 
    #     'Right Subarticular Stenosis': [
    #         (1, {1: 0.5444444444444444, 3: 0.9285714285714286, 5: 0.9455026455026455, 10: 0.9920634920634921}), 
    #         (0, {1: 0.5835978835978836, 3: 0.9492063492063492, 5: 0.9698412698412698, 10: 0.9962962962962963}), 
    #         (2, {1: 0.5761904761904761, 3: 0.9476190476190476, 5: 0.9629629629629629, 10: 0.9883597883597883}), 
    #         (3, {1: 0.5656084656084656, 3: 0.9322751322751323, 5: 0.9513227513227513, 10: 0.9925925925925926})]
    #     }

    # metrics = dict()
    # def train_and_collect_metrics(cond, desc, fct, step):
    #     print('-' * 50)
    #     print(f"Training: {cond} cross step {step}")
    #     best = train_model(
    #         input_dir="../",
    #         condition=cond,
    #         description=desc,
    #         scores_fct=fct,
    #         step=step,
    #     )
    #     return step, best

    # for cond, desc, fct, out_name in zip(conditions, descriptions, scores_fct, out_name):
    #     if not "Subarticular" in cond:
    #         continue

    #     metrics[cond] = []
        
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    #         future_to_step = {executor.submit(train_and_collect_metrics, cond, desc, fct, step): step for step in range(4)}
            
    #         for future in concurrent.futures.as_completed(future_to_step):
    #             step = future_to_step[future]
    #             try:
    #                 step_num, best = future.result()
    #                 metrics[cond].append((step_num, best))
    #                 print(f"Metrics for {cond}, step {step_num}: {best}")
    #             except Exception as exc:
    #                 print(f"{cond}, step {step} generated an exception: {exc}")

    # print("Done!")
    # print("All metrics:", metrics)

    # class SliceSelecterMetamodel(nn.Module):
    #     def __init__(self, models):
    #         super().__init__()
    #         self.models = nn.ModuleList(models)

    #     def forward(self, images):
    #         outputs = list()
    #         for model in self.models:
    #             _, preds_logits = model(images)
    #             outputs.append(preds_logits)
    #         outputs = torch.stack(outputs, dim = 1)
    #         outputs = torch.mean(outputs, dim = 1)
    #         return outputs.sigmoid()

    class SliceSelecterMetamodel(nn.Module):
        def __init__(self, models):
            super().__init__()
            self.models = nn.ModuleList(models)

        def forward(self, images):
            outputs = [model(images)[1] for model in self.models]
            outputs = torch.stack(outputs, dim=1)
            return outputs.mean(dim=1).sigmoid()

    # test metamodels
    conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Spinal Canal Stenosis", "Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR", "Axial T2", "Axial T2"]
    out_name = ["slice_selector_st1_left_metamodel.ts", "slice_selector_st1_right_metamodel.ts", "slice_selector_st2_metamodel.ts", "slice_selector_ax_left_metamodel.ts", "slice_selector_ax_right_metamodel.ts"]
    scores_fct = [generate_scores, generate_scores, generate_scores, generate_scores, generate_scores]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = FocalLoss(alpha=1, gamma=2)
    for cond, desc, fct, o in zip(conditions, descriptions, scores_fct, out_name):
        
        models = list()
        for step in range(4):
            current_name = f"trained_models/{cond.lower().replace(' ', '_')}_step_{step}.pth"
            step_model = SliceSelecterModelTransformer()
            step_model.load_state_dict(torch.load(current_name, weights_only=True, map_location='cpu'))
            models.append(step_model)

        new_metamodel = SliceSelecterMetamodel(models)
        new_metamodel = new_metamodel.eval().to(device)
        dataset = generate_dataset("../", cond, desc, fct)

        valid_dataset = SlicePredicterDataset(dataset)
        valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=4)
        loss_valid, instance_accuracy = validate_metamodel(new_metamodel, valid_loader, criterion, device)
        
        scripted_model = torch.jit.script(new_metamodel)
        scripted_model.save(o)

        print()
        print(cond)
        print(loss_valid, instance_accuracy)