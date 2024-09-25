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
#   MODELS
#
##########################################################

import torch
import torch.nn as nn
import timm
torch.set_grad_enabled(False) # remove grad for the script

####################################################
#        CLASSIFICATION
####################################################


class REM_ModelLoader:
    def __init__(self, model_name, hidden_size=256):
        self.model = timm.create_model(model_name, pretrained=False)
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.model = self.modify_classifier()

    def modify_classifier(self):
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        elif hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.fc.in_features if hasattr(self.model.head, 'fc') else self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, self.hidden_size),
            )
        else:
            raise NotImplementedError("Unknown classifier structure")
        return self.model

class REM_Encoder(nn.Module):
    def __init__(self, features_size, backbone_name):
        super().__init__()
        self.name = backbone_name
        self.model = REM_ModelLoader(model_name=backbone_name, hidden_size=features_size).model
    def forward(self, x):
        return self.model(x)

class REM_Classifier(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(),
            nn.Linear(in_features // 4, n_classes)
        )
    def forward(self, x):
        x = self.classifier(x)
        return x

class REM(nn.Module):
    def __init__(self, backbones, n_fold_classifier, features_size, n_classes):
        super().__init__()
        self.nb_encoders = len(backbones)
        self.nb_classifiers = n_fold_classifier
        self.features_size = features_size
        self.n_classes = n_classes
        
        self.encoders = nn.ModuleList([
            REM_Encoder(features_size, backbone) for backbone in backbones
        ])
        self.classifiers = nn.ModuleList([
            REM_Classifier(features_size, n_classes) for _ in range(n_fold_classifier)
        ])
        self.weights_encoders = nn.Parameter(torch.ones(len(self.encoders)))

    def forward_encoders(self, crop):
        encodeds = torch.stack([encoder(crop) for encoder in self.encoders], dim=1)
        return encodeds

    def forward(self, crop):
        final_output = list()
        _encodeds = self.forward_encoders(crop)
        for classifier in self.classifiers:
            classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.nb_encoders)], dim=1)
            classifier_output = torch.einsum("bsf,s->bf", classified_, torch.softmax(self.weights_encoders, dim=0))
            final_output.append(classifier_output)
        final_output = torch.stack(final_output, dim=1)
        final_output = torch.mean(final_output, dim=1)
        return final_output


####################################################
#        CROP SELECTION
####################################################

class DynamicModelLoader:
    def __init__(self, model_name, hidden_size=None):
        self.model = timm.create_model(model_name, pretrained=False)
        self.model_name = model_name
        self.hidden_size = hidden_size
        if hidden_size is not None:
            self.model = self.modify_classifier()

    def modify_classifier(self):
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        elif hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        elif hasattr(self.model, 'head'):
            in_features = self.model.head.fc.in_features if hasattr(self.model.head, 'fc') else self.model.head.in_features
            self.model.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
                nn.Linear(in_features, self.hidden_size),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError("Unknown classifier structure")
        return self.model

class CropEncoder(nn.Module):
    def __init__(self, model_name, features_size):
        super().__init__()
        self.backbone = DynamicModelLoader(model_name, hidden_size=features_size).model

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = torch.mean(features, dim = (2, 3))
        return features

class CropSelecter(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.features_size = 1024
        self.encoder = CropEncoder(model_name, self.features_size)
        self.lstm = nn.LSTM(self.features_size, self.features_size // 4, bidirectional=True, batch_first=True, dropout=0.021, num_layers=2)
        self.classifier = nn.Linear(self.features_size // 2, 1)

    def forward(self, crops):
        b, s, c, h, w = crops.size()
        features = torch.empty(b, s, self.features_size).to(crops.device)
        for i in range(s):
            features[:, i] = self.encoder(crops[:, i])
        lstm_out, _ = self.lstm(features)
        scores = self.classifier(lstm_out)
        return scores.sigmoid().squeeze(-1)

##########################################################
#
#   USEFULL
#
##########################################################

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def get_device():
    #return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_torch_script_model(path):
    return torch.jit.load(path, map_location="cpu").eval().to(get_device())

##########################################################
#
#   SLICES SELECTION INFERENCE
#
##########################################################

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

def get_best_slice_selection(config, pathes, topk):
    """
    Return best slices for each level.
    Slice are not sorted by instance_number, they are sorted by topk


    TODO => REFACTO
    """

    nb_slices = len(pathes)
    images = np.zeros((nb_slices, 1, *config['slice_input_size']))
    for k, path in enumerate(pathes):
        im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                        config['slice_input_size'],
                                        interpolation=cv2.INTER_LINEAR)
        im = (im - im.min()) / (im.max() - im.min() + 1e-9)
        images[k, 0, ...] = im
    images = torch.tensor(images).expand(nb_slices, 3, *config['slice_input_size']).float()
    preds_model = config['models']['slice_selecter'](images.to(config["device"]).unsqueeze(0))
    preds_model = preds_model.squeeze()
    preds_overall = torch.sum(preds_model, dim=0)

    # get best 3 by level
    slices_by_level = [
        {"pathes": list(), "values": list()} for _ in range(preds_model.shape[0])
    ]
    for level in range(preds_model.shape[0]):
        pred_level = preds_model[level, :]
        values, max_indice = get_max_consecutive_N(pred_level, 3)
        slices_by_level[level]['pathes'] = [pathes[i] for i in max_indice]
        slices_by_level[level]['values'] = [v for v in values]

    # get best 3 overall (=> best after sum of each level)
    values, max_indices = get_max_consecutive_N(preds_overall, 3)
    best_slices_overall = dict()
    best_slices_overall['pathes'] = [pathes[i] for i in max_indices]
    best_slices_overall['values'] = [v for v in values]

    # get best 8 by level (for crop selection)
    slices8_by_level = [
        {"pathes": list(), "values": list()} for _ in range(preds_model.shape[0])
    ]
    for level in range(preds_model.shape[0]):
        pred_level = preds_model[level, :]
        values, max_indice = get_max_consecutive_N(pred_level, 8)
        slices8_by_level[level]['pathes'] = [pathes[i] for i in max_indice]
        slices8_by_level[level]['values'] = [v for v in values]

    return slices_by_level, best_slices_overall, slices8_by_level

def get_slices_to_use(study_id, series_ids, config):
    """
    Check slices over all series and return best selected for each level

    TODO : REFACTO
    """

    slices_by_series = dict()
    for s_id in series_ids:
        _pathes = glob.glob(f"{config['input_images_folder']}/{study_id}/{s_id}/*.dcm")
        _pathes = sorted(_pathes, key = lambda x : get_instance(x))
        slices_by_series[s_id] = _pathes

    # compute best slices for each series
    best_slices_per_series = dict()
    for s_id in slices_by_series:
        # WE NEED AT LEAST 8 SLICES !
        if len(slices_by_series[s_id]) < 8:
            continue
        best_slices, best_slices_overall, best_8slices = get_best_slice_selection(config, slices_by_series[s_id], topk=3)
        best_slices_per_series[s_id] = {
            "best_by_level": best_slices, 
            "best_8_by_level": best_8slices,
            "best_overall": best_slices_overall,
        }
    
    # find best series for each level
    #   => highest activation on selection
    best_slices_per_level = [list() for _ in range(5)]
    for level in range(5):
        best_level = 0
        for s_id in best_slices_per_series:
            sum_series = sum(best_slices_per_series[s_id]["best_by_level"][level]['values'])
            if sum_series > best_level:
                best_level = sum_series
                best_slices_per_level[level] = best_slices_per_series[s_id]["best_by_level"][level]['pathes']

    # find best series for each level
    #   => highest activation on selection
    best_8_slices_per_level = [list() for _ in range(5)]
    for level in range(5):
        best_level = 0
        for s_id in best_slices_per_series:
            sum_series = sum(best_slices_per_series[s_id]["best_8_by_level"][level]['values'])
            if sum_series > best_level:
                best_level = sum_series
                best_8_slices_per_level[level] = best_slices_per_series[s_id]["best_8_by_level"][level]['pathes']

    # find best overall series
    best_slices_overall = list()
    best_sum = 0
    for s_id in best_slices_per_series:
        sum_series = sum(best_slices_per_series[s_id]["best_overall"]['values'])
        if sum_series > best_sum:
            best_slices_overall = best_slices_per_series[s_id]["best_overall"]['pathes']
            best_sum = sum_series

    return best_slices_per_level, best_slices_overall, best_8_slices_per_level

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
        images = np.zeros((3, *config['seg_input_size']))
        _slices_path = slices_to_use['best_overall'][:3]
        for k, path in enumerate(_slices_path):
            im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                            config['seg_input_size'],
                            interpolation=cv2.INTER_LINEAR
                        )
            im = (im - im.min()) / (im.max() - im.min() + 1e-9)
            images[k, ...] = im
        images = torch.tensor(images).float().to(config["device"])
        return images
    
    elif config['segmentation_slice_selection'] == "best_by_level":
        images = np.zeros((5, 3, *config['seg_input_size']))
        for level in range(5):
            _slices_to_use = slices_to_use['best_by_level'][level]
            _slices_to_use = sorted(_slices_to_use, key = lambda x: get_instance(x))
            for ch, slice_to_use in enumerate(_slices_to_use):
                im = cv2.resize(pydicom.dcmread(slice_to_use).pixel_array.astype(np.float32), 
                                config['seg_input_size'],
                                interpolation=cv2.INTER_LINEAR
                            )
                im = (im - im.min()) / (im.max() - im.min() + 1e-9)
                images[level, ch, ...] = im
        images = torch.tensor(images).float().to(config["device"])
        return images
    else:
        return None

def get_position_by_level(slices_to_use: list, config: dict) -> dict:
    inputs = get_segmentation_input(slices_to_use, config)
    if config['segmentation_slice_selection'] == "best_overall":
        masks = config["models"]['segmenter'](inputs.unsqueeze(0)) # model predict 5 levels
        masks = masks.squeeze()
        position_by_level = [find_center_of_largest_activation(masks[i]) for i in range(5)]
    else:
        masks = config["models"]['segmenter'](inputs) # model predict 1 level, we put levels in batch dim
        masks = masks.squeeze(1)
        position_by_level = [find_center_of_largest_activation(masks[i]) for i in range(5)]
    return position_by_level

##########################################################
#
#   CROP SELECTION INFERENCE
#
##########################################################

def clahe_equalization_norm2(image, clip_limit=2.0, grid_size=(8, 8)):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    image = clahe.apply(np.uint8(image))
    image = image.astype(np.float32) / 255.
    return image

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
        crop = extract_centered_square_with_padding(pixel_array, y, x, *crop_size) # x y reversed in array
        crop = clahe_equalization_norm2(crop)
        crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        output_crops[k, ...] = crop
    return output_crops

def get_8crops_by_level(slices_to_use, position_by_level, config):
    crops_output = np.zeros((5, 8, 128, 128))
    for level, (slices_, position) in enumerate(zip(slices_to_use["best_8_by_level"], position_by_level)):
        if position is not None:
            slices = slices_
            px = int(position[0] * config['crop_selecter_resize_image'][1])
            py = int(position[1] * config['crop_selecter_resize_image'][0])
            crops_output[level, ...] = cut_crops(slices, px, py, config['crop_selecter_crop_size'], config['crop_selecter_resize_image'])
        else:
            print("No prediction on segmentation !", level, "for", config["condition"])

    crops_output = torch.tensor(crops_output).float().to(config["device"])
    crops_output = crops_output.unsqueeze(2).expand(5, 8, 3, 128, 128)
    return crops_output

def compute_best3_by_level(crops, config):
    preds = config['models']['crops_selecter'](crops)

    crops_to_ret = torch.zeros((5, 3, 128, 128))

    for level in range(5):
        pred = preds[level]
        values, indices = get_max_consecutive_N(pred, 3)
        for k, indice in enumerate(indices):
            crops_to_ret[level, k] = crops[level, indice, 0]

    return crops_to_ret

##########################################################
#
#   CLASSIFICATION INFERENCE
#
##########################################################

def get_classification(crops: torch.tensor, config):
    preds = config['models']['classifier'](crops.to(get_device()))
    preds = torch.softmax(preds, dim=1)
    return preds

##########################################################
#
#   INIT MODELS
#
##########################################################

def load_models(task):
    # load slice selection
    slice_selecter = load_torch_script_model(task['slice_model_path'])

    # load segmentation
    segmenter = load_torch_script_model(task['seg_model_path'])

    # load crop selection
    crops_selecter = CropSelecter(task['crop_selecter_backbone'])
    crops_selecter.load_state_dict(torch.load(task['crops_model_path'], map_location="cpu", weights_only=True))
    crops_selecter = crops_selecter.eval().to(get_device())

    # load classification
    backbones = task['class_backbones']
    classifier = REM(
        n_classes=3,
        n_fold_classifier=3,
        backbones=backbones,
        features_size=256,
    )
    classifier.load_state_dict(torch.load(task['class_model_path'], map_location="cpu", weights_only=True))
    classifier = classifier.eval().to(get_device())

    return {
        "slice_selecter": slice_selecter,
        "segmenter": segmenter,
        "crops_selecter": crops_selecter,
        "classifier": classifier,
    }

def get_tasks_metadatas():
    tasks = [
        # SPINAL CANAL STENOSIS
        {   
            # MODELS
            "slice_model_path": "trained_models/v6/slice_selector_st2.ts",
            "seg_model_path": "trained_models/v6/model_segmentation_st2.ts",
            "crops_model_path": "trained_models/v6/model_crop_selection_st2.pth",
            "class_model_path": "trained_models/v6/classification_st2.pth",
            
            # GENERAL
            "description": "Sagittal T2/STIR",
            "condition": "Spinal Canal Stenosis",

            # HYPERPARAMETERS
            #   => slice selection
            "slice_input_size": (224, 224),

            #   => segmentation
            "segmentation_slice_selection": 'best_overall',
            "seg_input_size": (384, 384),
            "seg_mask_size": (384, 384), # unused

            #   => crop selection
            "crop_selecter_backbone": 'convnext_base.fb_in22k_ft_in1k',
            "crop_selecter_crop_size": (64, 96),
            "crop_selecter_resize_image": (640, 640),
            "crop_selecter_input_size": (128, 128),

            #   => classification
            "class_backbones": ['cspresnet50.ra_in1k', 'convnext_base.fb_in22k_ft_in1k', 'ese_vovnet39b.ra_in1k', 'densenet161.tv_in1k', 'dm_nfnet_f0.dm_in1k'],
            "class_crop_size": (64, 96),
            "class_resize_image": (640, 640),
            "class_input_size": (128, 128),
        },
        
        # LEFT NEURAL FORAMINAL NARROWING
        {   
            # MODELS
            "slice_model_path": "trained_models/v6/slice_selector_st1_left.ts",
            "seg_model_path": "trained_models/v6/model_segmentation_st1_left.ts",
            "crops_model_path": "trained_models/v6/model_crop_selection_st1_left.pth",
            "class_model_path": "trained_models/v6/classification_st1_left.pth",
            
            # GENERAL
            "description": "Sagittal T1",
            "condition": "Left Neural Foraminal Narrowing",

            # HYPERPARAMETERS
            #   => slice selection
            "slice_input_size": (224, 224),

            #   => segmentation
            "segmentation_slice_selection": 'best_overall',
            "seg_input_size": (384, 384),
            "seg_mask_size": (384, 384), # unused

            #   => crop selection
            "crop_selecter_backbone": 'convnext_base.fb_in22k_ft_in1k',
            "crop_selecter_crop_size": (64, 96),
            "crop_selecter_resize_image": (640, 640),
            "crop_selecter_input_size": (128, 128),

            #   => classification
            "class_backbones": ['cspresnet50.ra_in1k', 'convnext_base.fb_in22k_ft_in1k', 'ese_vovnet39b.ra_in1k', 'densenet161.tv_in1k', 'dm_nfnet_f0.dm_in1k'],
            "class_crop_size": (64, 96),
            "class_resize_image": (640, 640),
            "class_input_size": (128, 128),
        },

        # RIGHT NEURAL FORAMINAL NARROWING
        {   
            # MODELS
            "slice_model_path": "trained_models/v6/slice_selector_st1_right.ts",
            "seg_model_path": "trained_models/v6/model_segmentation_st1_right.ts",
            "crops_model_path": "trained_models/v6/model_crop_selection_st1_right.pth",
            "class_model_path": "trained_models/v6/classification_st1_right.pth",
            
            # GENERAL
            "description": "Sagittal T1",
            "condition": "Right Neural Foraminal Narrowing",

            # HYPERPARAMETERS
            #   => slice selection
            "slice_input_size": (224, 224),

            #   => segmentation
            "segmentation_slice_selection": 'best_overall',
            "seg_input_size": (384, 384),
            "seg_mask_size": (384, 384), # unused

            #   => crop selection
            "crop_selecter_backbone": 'convnext_base.fb_in22k_ft_in1k',
            "crop_selecter_crop_size": (64, 96),
            "crop_selecter_resize_image": (640, 640),
            "crop_selecter_input_size": (128, 128),

            #   => classification
            "class_backbones": ['cspresnet50.ra_in1k', 'convnext_base.fb_in22k_ft_in1k', 'ese_vovnet39b.ra_in1k', 'densenet161.tv_in1k', 'dm_nfnet_f0.dm_in1k'],
            "class_crop_size": (64, 96),
            "class_resize_image": (640, 640),
            "class_input_size": (128, 128),
        },

        # RIGHT SUBARTICULAR STENOSIS
        {   
            # MODELS
            "slice_model_path": "trained_models/v6/slice_selector_ax_right.ts",
            "seg_model_path": "trained_models/v6/model_segmentation_ax_right.ts",
            "crops_model_path": "trained_models/v6/model_crop_selection_ax_right.pth",
            "class_model_path": "trained_models/v6/classification_ax_right.pth",
            
            # GENERAL
            "description": "Axial T2",
            "condition": "Right Subarticular Stenosis",

            # HYPERPARAMETERS
            #   => slice selection
            "slice_input_size": (224, 224),

            #   => segmentation
            "segmentation_slice_selection": 'best_by_level',
            "seg_input_size": (384, 384),
            "seg_mask_size": (384, 384), # unused

            #   => crop selection
            "crop_selecter_backbone": 'convnext_base.fb_in22k_ft_in1k',
            "crop_selecter_crop_size": (184, 184),
            "crop_selecter_resize_image": (640, 640),
            "crop_selecter_input_size": (128, 128),

            #   => classification
            "class_backbones": ['cspresnet50.ra_in1k', 'convnext_base.fb_in22k_ft_in1k', 'ese_vovnet39b.ra_in1k', 'densenet161.tv_in1k', 'dm_nfnet_f0.dm_in1k'],
            "class_crop_size": (184, 184),
            "class_resize_image": (640, 640),
            "class_input_size": (128, 128),
        },

        # LEFT SUBARTICULAR STENOSIS
        {   
            # MODELS
            "slice_model_path": "trained_models/v6/slice_selector_ax_left.ts",
            "seg_model_path": "trained_models/v6/model_segmentation_ax_left.ts",
            "crops_model_path": "trained_models/v6/model_crop_selection_ax_left.pth",
            "class_model_path": "trained_models/v6/classification_ax_left.pth",
            
            # GENERAL
            "description": "Axial T2",
            "condition": "Left Subarticular Stenosis",

            # HYPERPARAMETERS
            #   => slice selection
            "slice_input_size": (224, 224),

            #   => segmentation
            "segmentation_slice_selection": 'best_by_level',
            "seg_input_size": (384, 384),
            "seg_mask_size": (384, 384), # unused

            #   => crop selection
            "crop_selecter_backbone": 'convnext_base.fb_in22k_ft_in1k',
            "crop_selecter_crop_size": (184, 184),
            "crop_selecter_resize_image": (640, 640),
            "crop_selecter_input_size": (128, 128),

            #   => classification
            "class_backbones": ['cspresnet50.ra_in1k', 'convnext_base.fb_in22k_ft_in1k', 'ese_vovnet39b.ra_in1k', 'densenet161.tv_in1k', 'dm_nfnet_f0.dm_in1k'],
            "class_crop_size": (184, 184),
            "class_resize_image": (640, 640),
            "class_input_size": (128, 128),
        },
    ]
    return tasks

##########################################################
#
#   MAIN INFERENCE
#
##########################################################

def predict_lumbar(df_description: pd.DataFrame, config: dict, study_id: int) -> list:
    try:
        series_ids = df_description[(df_description['study_id'] == study_id)
                                & (df_description['series_description'] == config['description'])]['series_id'].to_list()

        best_slices_by_level, best_slices_overall, slices8_by_level = get_slices_to_use(study_id, series_ids, config)
        slices_to_use = dict(
            best_by_level=best_slices_by_level,
            best_overall=best_slices_overall,
            best_8_by_level=slices8_by_level,
        )

        positions_by_level = get_position_by_level(slices_to_use, config)

        crops_by_level = get_8crops_by_level(slices_to_use, positions_by_level, config)

        crops_for_classification = compute_best3_by_level(crops_by_level, config)

        classification_results = get_classification(crops_for_classification, config)

    except Exception as e:
        print(f"Error {study_id} {config['condition']}:", e)
        classification_results = None

    predictions = list()
    row_id = f"{study_id}_{config['condition'].lower().replace(' ', '_')}"
    for level_int, level_str in enumerate(['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']):
        predictions.append(dict(
            row_id = f"{row_id}_{level_str}",
            normal_mild = classification_results[level_int][0].item() if classification_results is not None else 1/3,
            moderate = classification_results[level_int][1].item() if classification_results is not None else 1/3,
            severe = classification_results[level_int][2].item() if classification_results is not None else 1/3,
        ))

    return predictions


def compute_pipeline(input_images_folder, description_file, nb_studies_id=None):
    df_description = pd.read_csv(description_file)
    studies_id = df_description["study_id"].unique()
    
    tasks_metadatas = get_tasks_metadatas()
    for task in tasks_metadatas:
        print(f"Loading models for task: {task['condition']}")
        task["models"] = load_models(task)
        task["input_images_folder"] = input_images_folder
        task['device'] = get_device()

    if nb_studies_id is None:
        nb_studies_id = len(studies_id)

    print(f"Launch pipeline on {nb_studies_id} studies !")
    final_predictions = []
    for study_id in tqdm.tqdm(studies_id[:nb_studies_id], desc="Predicting for each study"):
        for config in tasks_metadatas:
            _predictions = predict_lumbar(df_description, config, study_id)
            final_predictions.extend(_predictions)

    return pd.DataFrame(final_predictions)

if __name__ == "__main__":
    df = compute_pipeline("train_images/", "train_series_descriptions.csv", 190)
    df.to_csv("pipeline_preds.csv", index=False)