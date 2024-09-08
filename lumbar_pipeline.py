# In one file for kaggle :3
import numpy as np
import pandas as pd
import tqdm
import glob
import cv2
import pydicom
from scipy.ndimage import label, center_of_mass

##########################################################
#
#   MODEL
#
##########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ExpertEncoder(nn.Module):
    def __init__(self, encoder_name, out_features, sequence_lenght):
        super().__init__()
        self.sequence_lenght = sequence_lenght
        if encoder_name == "efficientnet":
            self.encoder = torchvision.models.efficientnet_b0(weights="DEFAULT")
            self.classifier = nn.Linear(1280, out_features)
        elif encoder_name == "squeezenet":
            self.encoder = torchvision.models.squeezenet1_0(weights="DEFAULT")
            self.classifier = nn.Linear(512, out_features)
        else:
            raise NotImplementedError("Unknown encoder")

    def _forward_image(self, image):
        features = self.encoder.features(image)
        features = features.mean(dim=(2, 3))
        out = self.classifier(features)
        return out

    def forward(self, images):
        out = torch.stack([self._forward_image(images[:, i]) for i in range(self.sequence_lenght)], dim=1)
        return out

class ExpertClassifier(nn.Module):
    def __init__(self, in_features, sequence_lenght, num_classes):
        super().__init__()
        self.sequence_lenght = sequence_lenght
        self.in_features = in_features

        self.gating_encoder = nn.Sequential(
            nn.Linear(self.in_features, 1),
            nn.Sigmoid()
        )

        in_size = in_features * sequence_lenght
        self.classifier = nn.Sequential(
            nn.Linear(in_size, in_size // 4),
            nn.ReLU(),
            nn.Linear(in_size // 4, num_classes)
        )

    def forward(self, encoded):
        # shape = (batch_size, sequence_lenght, self.encoder_features)
        gates_encoder = torch.sigmoid(self.gating_encoder(encoded))
        encoder_output = torch.sum(gates_encoder * encoded, dim=1)
        # encoder_output.shape = (batch_size, self.sequence_lenght, self.encoder_features)

        encoder_output = encoder_output.reshape(encoded.shape[0], self.sequence_lenght * self.in_features)
        return self.classifier(encoder_output)

class ExperCeption(nn.Module):
    def __init__(self, 
                num_expert_encoder=3,
                num_expert_classifier=3,
                encoder_features=64,
                num_classes=3,
                sequence_lenght=5,
                encoder="squeezenet"
                ):
        super().__init__()
        self.num_expert_encoder = num_expert_encoder    
        self.num_expert_classifier = num_expert_classifier    
        self.encoder_features = encoder_features    
        self.num_classes = num_classes    
        self.sequence_lenght = sequence_lenght
        self.encoder = encoder

        self.experts_encoder = nn.ModuleList([ExpertEncoder(encoder_name=encoder, out_features=self.encoder_features, sequence_lenght=self.sequence_lenght) for _ in range(self.num_expert_encoder)])
        self.experts_classifier = nn.ModuleList([ExpertClassifier(in_features=self.encoder_features, sequence_lenght=self.sequence_lenght, num_classes=self.num_classes) for _ in range(self.num_expert_classifier)])

        self.gating_classifier = nn.Sequential(
            nn.Linear(self.num_classes, 1),
            nn.Sigmoid()
        )

    def diversity_loss(self, num_experts, expert_outputs):
        if num_experts > 1:
            loss = 0.0
            for i in range(num_experts):
                for j in range(i + 1, num_experts):
                    cos_sim = F.cosine_similarity(expert_outputs[:, i], expert_outputs[:, j], dim=-1)
                    loss += cos_sim.mean()
            return loss / (num_experts * (num_experts - 1) / 2)
        return 0

    def forward(self, images):
        expert_encoder_output = torch.stack([self.experts_encoder[i](images) for i in range(self.num_expert_encoder)], dim=1)
        loss_diversity_encoder = self.diversity_loss(self.num_expert_encoder, expert_encoder_output)
        # expert_encoder_output.shape = (batch_size, self.num_expert_encoder, self.sequence_lenght, self.encoder_features)

        expert_classifier_output = torch.stack([self.experts_classifier[i](expert_encoder_output) for i in range(self.num_expert_classifier)], dim = 1)
        loss_diversity_classifier = self.diversity_loss(self.num_expert_classifier, expert_classifier_output)
        # expert_classifier_output.shape = (batch_size, self.num_expert_classifier, self.sequence_lenght * self.num_classes)

        gates_classifier = torch.sigmoid(self.gating_classifier(expert_classifier_output))
        classifier_output = torch.sum(gates_classifier * expert_classifier_output, dim=1)
        # classifier_output.shape = (batch_size, self.num_classes)

        return classifier_output, loss_diversity_encoder + loss_diversity_classifier

##########################################################
#
#   USEFULL
#
##########################################################

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def load_model_classification(model_path, encoder="efficientnet"):
    model_classification = ExperCeption(
        num_classes=3,
        num_expert_classifier=3,
        num_expert_encoder=3,
        sequence_lenght=5,
        encoder_features=64,
        encoder=encoder,
    )
    model_classification.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    return model_classification.eval()

def load_torch_script_model(path):
    return torch.jit.load(path, map_location="cpu").eval()

##########################################################
#
#   SLICES SELECTION INFERENCE
#
##########################################################

def get_best_slice_selection(config, pathes, topk, device='cpu'):
    """
    Return best slices for each level.
    Slice are not sorted by instance_number, they are sorted by topk
    """

    nb_slices = len(pathes)
    images = np.zeros((nb_slices, 1, *config['slice_selection_input_shape']))
    for k, path in enumerate(pathes):
        im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                        config['slice_selection_input_shape'],
                                        interpolation=cv2.INTER_LINEAR)
        im = (im - im.min()) / (im.max() - im.min() + 1e-9)
        images[k, 0, ...] = im
    images = torch.tensor(images).expand(nb_slices, 3, *config['slice_selection_input_shape']).float()
    preds = config['model_slice_selection'](images.to(device).unsqueeze(0)).squeeze()
    preds_overall = torch.sum(preds, dim=0)

    # get best by level
    slices_by_level = [
        {"pathes": list(), "values": list()} for _ in range(preds.shape[0])
    ]
    for level in range(preds.shape[0]):
        pred_level = preds[level, :]
        values, max_indice = torch.topk(pred_level, topk)
        slices_by_level[level]['pathes'] = [pathes[i.item()] for i in max_indice]
        slices_by_level[level]['values'] = [v.item() for v in values]

    # get best overall (=> best after sum of each level)
    values, max_indices = torch.topk(preds_overall, topk)
    best_slices_overall = dict()
    best_slices_overall['pathes'] = [pathes[i.item()] for i in max_indices]
    best_slices_overall['values'] = [v.item() for v in values]

    return slices_by_level, best_slices_overall

def get_slices_to_use(study_id, series_ids, config):
    """
    Check slices over all series and return best selected for each level
    """

    slices_by_series = dict()
    for s_id in series_ids:
        _pathes = glob.glob(f"{config['input_images_folder']}/{study_id}/{s_id}/*.dcm")
        _pathes = sorted(_pathes, key = lambda x : get_instance(x))
        slices_by_series[s_id] = _pathes

    # compute best slices for each series
    best_slices_per_series = dict()
    for s_id in slices_by_series:
        best_slices, best_slices_overall = get_best_slice_selection(config, slices_by_series[s_id], topk=5)
        best_slices_per_series[s_id] = {
            "best_per_level": best_slices, 
            "best_overall": best_slices_overall,
        }
    
    # find best series for each level
    #   => highest activation on selection
    best_slices_per_level = [list() for _ in range(5)]
    for level in range(5):
        best_level = 0
        for s_id in best_slices_per_series:
            sum_series = sum(best_slices_per_series[s_id]["best_per_level"][level]['values'])
            if sum_series > best_level:
                best_level = sum_series
                best_slices_per_level[level] = best_slices_per_series[s_id]["best_per_level"][level]['pathes']

    # find best overall series
    best_slices_overall = list()
    best_sum = 0
    for s_id in best_slices_per_series:
        sum_series = sum(best_slices_per_series[s_id]["best_overall"]['values'])
        if sum_series > best_sum:
            best_slices_overall = best_slices_per_series[s_id]["best_overall"]['pathes']
            best_sum = sum_series

    return best_slices_per_level, best_slices_overall

##########################################################
#
#   SEGMENTATION INFERENCE
#
##########################################################

def find_center_of_largest_activation(mask: torch.tensor) -> tuple:
    mask = (mask > 0.5).float().detach().cpu().numpy()
    labeled_mask, num_features = label(mask)
    if num_features == 0:
        return None
    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0
    largest_component_center = center_of_mass(labeled_mask == np.argmax(sizes))
    center_coords = tuple(map(int, largest_component_center))
    return (center_coords[1] / mask.shape[1], center_coords[0] / mask.shape[0]) # x, y normalised

def get_segmentation_input(slices_to_use: list, config: dict):
    if config['segmentation_slice_selection'] == "best_overall":
        images = np.zeros((3, *config['segmentation_input_shape']))
        _slices_path = slices_to_use['best_overall'][:3]
        for k, path in enumerate(_slices_path):
            im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                            config['segmentation_input_shape'],
                            interpolation=cv2.INTER_LINEAR
                        )
            im = (im - im.min()) / (im.max() - im.min() + 1e-9)
            images[k, ...] = im
        images = torch.tensor(images).float()
        return images
    elif config['segmentation_slice_selection'] == "best_by_level":
        images = np.zeros((5, 1, *config['segmentation_input_shape']))
        for level in range(5):
            slice_to_use = slices_to_use['best_by_level'][level][0] # 1 slice per level => the best one
            im = cv2.resize(pydicom.dcmread(slice_to_use).pixel_array.astype(np.float32), 
                            config['segmentation_input_shape'],
                            interpolation=cv2.INTER_LINEAR
                        )
            im = (im - im.min()) / (im.max() - im.min() + 1e-9)
            images[level, 0, ...] = im
        images = torch.tensor(images).expand(5, 3, *config['segmentation_input_shape']).float()
        return images
    else:
        return None

def get_position_by_level(slices_to_use: list, config: dict) -> dict:
    inputs = get_segmentation_input(slices_to_use, config)
    masks = config["model_segmentation"](inputs.unsqueeze(0)).squeeze()
    position_by_level = [find_center_of_largest_activation(masks[i]) for i in range(5)]
    return position_by_level

##########################################################
#
#   CLASSIFICATION INFERENCE
#
##########################################################

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
    output_crops = np.zeros((len(slices_path), 1, *crop_size))
    for k, slice_path in enumerate(slices_path):
        pixel_array = pydicom.dcmread(slice_path).pixel_array.astype(np.float32)
        pixel_array = cv2.resize(pixel_array, image_resize, interpolation=cv2.INTER_LINEAR)
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-9)
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # reverse x y for cutting
        output_crops[k, 0, ...] = crop
    return output_crops

def get_crops_by_level(slices_to_use, position_by_level, config):
    crops_output = np.zeros((5, 5, 1, *config['classification_input_size']))
    for level, (slices_, position) in enumerate(zip(slices_to_use["best_by_level"], position_by_level)):
        slices = sorted(slices_[:config['classification_sequence_lenght']], key = lambda x : get_instance(x))
        px = int(position[0] * config['classification_resize_image'][0])
        py = int(position[1] * config['classification_resize_image'][0])
        crops_output[level, ...] = cut_crops(slices, px, py, config['classification_input_size'], config['classification_resize_image'])

    crops_output = torch.tensor(crops_output).float()
    crops_output = crops_output.expand(5, 5, 3, *config['classification_input_size'])
    return crops_output

def get_classification(crops: torch.tensor, config):
    preds, _ = config['model_classification'](crops)
    preds = torch.softmax(preds, dim=1)
    return preds

##########################################################
#
#   MAIN INFERENCE
#
##########################################################

def predict_lumbar(df_description: pd.DataFrame, config: dict) -> list:
    studies_id = df_description["study_id"].unique()
    for study_id in tqdm.tqdm(studies_id, desc="Predicting"):
        series_ids = df_description[(df_description['study_id'] == study_id)
                                & (df_description['series_description'] == config['description'])]['series_id'].to_list()
        
        best_slices_by_level, best_slices_overall = get_slices_to_use(study_id, series_ids, config)
        slices_to_use = dict(
            best_by_level=best_slices_by_level,
            best_overall=best_slices_overall
        )
        
        positions_by_level = get_position_by_level(slices_to_use, config)
        crops_by_level = get_crops_by_level(slices_to_use, positions_by_level, config)
        classification_results = get_classification(crops_by_level, config)
        
        predictions = list()
        row_id = f"{study_id}_{config['condition'].lower().replace(' ', '_')}"
        for level_int, level_str in enumerate(['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']):
            predictions.append(dict(
                row_id = f"{row_id}_{level_str}",
                normal_mild = classification_results[level_int][0].item(),
                moderate = classification_results[level_int][1].item(),
                severe = classification_results[level_int][2].item(),
            ))

    return predictions

def configure_inference(slice_model_path, seg_model_path, class_model, input_folder, description, condition, class_input_size, class_resize_image, seg_mode):
    return dict(
        model_slice_selection=load_torch_script_model(slice_model_path),
        model_segmentation=load_torch_script_model(seg_model_path),
        model_classification=class_model,
        slice_selection_input_shape=(224, 224),
        segmentation_input_shape=(224, 224),
        segmentation_mask_shape=(96, 96),
        classification_input_size=class_input_size,
        classification_resize_image=class_resize_image,
        classification_sequence_lenght=5,
        description=description,
        condition=condition,
        input_images_folder=input_folder,
        segmentation_slice_selection=seg_mode,
    )

if __name__ == "__main__":
    input_images_folder = "../REFAIT/test_images/"
    df_description = pd.read_csv("../REFAIT/test_series_descriptions.csv")
    final_predictions = []

    tasks = [
        {
            "class_model_path": "trained_models/v0/model_classification_st2.pth",
            "slice_model_path": "trained_models/v0/model_slice_selection_st2.ts",
            "seg_model_path": "trained_models/model_segmentation_st2.ts",
            "encoder": "squeezenet",
            "description": "Sagittal T2/STIR",
            "condition": "Spinal Canal Stenosis",
            "class_input_size": (164, 250),
            "class_resize_image": (600, 600),
            "segmentation_slice_selection": "best_overall",
        },
        {
            "class_model_path": "trained_models/v0/model_classification_st1_left.pth",
            "slice_model_path": "trained_models/v0/model_slice_selection_st1_left.ts",
            "seg_model_path": "trained_models/model_segmentation_st1_left.ts",
            "encoder": "squeezenet",
            "description": "Sagittal T1",
            "condition": "Left Neural Foraminal Narrowing",
            "class_input_size": (164, 250),
            "class_resize_image": (600, 600),
            "segmentation_slice_selection": "best_overall",
        },
        {
            "class_model_path": "trained_models/v0/model_classification_st1_right.pth",
            "slice_model_path": "trained_models/v0/model_slice_selection_st1_right.ts",
            "seg_model_path": "trained_models/model_segmentation_st1_right.ts",
            "encoder": "efficientnet",
            "description": "Sagittal T1",
            "condition": "Right Neural Foraminal Narrowing",
            "class_input_size": (164, 250),
            "class_resize_image": (600, 600),
            "segmentation_slice_selection": "best_overall",
        },
        {
            "class_model_path": "trained_models/v0/model_classification_axt2_left.pth",
            "slice_model_path": "trained_models/v0/model_slice_selection_axt2_left.ts",
            "seg_model_path": "trained_models/v0/model_segmentation_axt2_left.ts",
            "encoder": "efficientnet",
            "description": "Axial T2",
            "condition": "Left Subarticular Stenosis",
            "class_input_size": (164, 164),
            "class_resize_image": (600, 600),
            "segmentation_slice_selection": "best_by_level",
        },
        {
            "class_model_path": "trained_models/v0/model_classification_axt2_right.pth",
            "slice_model_path": "trained_models/v0/model_slice_selection_axt2_right.ts",
            "seg_model_path": "trained_models/v0/model_segmentation_axt2_right.ts",
            "encoder": "squeezenet",
            "description": "Axial T2",
            "condition": "Right Subarticular Stenosis",
            "class_input_size": (164, 164),
            "class_resize_image": (600, 600),
            "segmentation_slice_selection": "best_by_level",
        }
    ]

    for task in tasks:
        print('task', task['condition'], task['description'])
        model_classification = load_model_classification(task["class_model_path"], encoder=task["encoder"])
        config = configure_inference(
            task["slice_model_path"],
            task["seg_model_path"],
            model_classification,
            input_images_folder,
            task["description"],
            task["condition"],
            task["class_input_size"],
            task["class_resize_image"],
            task["segmentation_slice_selection"],
        )
        _predictions = predict_lumbar(df_description, config)
        final_predictions.extend(_predictions)

    pd.DataFrame(final_predictions).to_csv("submission.csv", index=False)
