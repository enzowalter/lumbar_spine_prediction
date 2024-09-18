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
import timm
torch.set_grad_enabled(False) # remove grad for the script

class DynamicModelLoader:
    def __init__(self, model_name, num_classes=3, pretrained=False, hidden_size=256):
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model_name = model_name
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.feature_size = self.get_feature_size()
        self.model = self.modify_classifier()

    def get_feature_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 128, 128)
            features = self.model.forward_features(dummy_input)
            return features.shape[1]
    
    def modify_classifier(self):
        if 'vit_small_patch16_224' in self.model_name:
                self.model.head = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.Linear(384, self.hidden_size),
                )
        else:
            if hasattr(self.model, 'fc'):
                in_features = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Linear(in_features, self.hidden_size),
                )
            elif hasattr(self.model, 'classifier'):
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Sequential(
                    nn.Linear(in_features, self.hidden_size),
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

class FoldEncoderAttention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.query = nn.Linear(features, features)
        self.key = nn.Linear(features, features)
        self.value = nn.Linear(features, features)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoded):
        query = self.query(encoded)
        key = self.key(encoded)
        value = self.value(encoded)

        attention_scores = torch.einsum('bsf,bsf->bs', query, key) / (encoded.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        weighted_sum = torch.einsum('bsf,bs->bf', value, attention_weights)

        return weighted_sum

class FoldEncoder(nn.Module):
    def __init__(self, features_size, backbone_name):
        super().__init__()
        self.name = backbone_name
        self.model = DynamicModelLoader(model_name=backbone_name, hidden_size=features_size).model

    def forward(self, x):
        return self.model(x)

class FoldClassifier(nn.Module):
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

class FoldModelClassifier(nn.Module):
    def __init__(self, backbones, n_fold_classifier, features_size, n_classes):
        super().__init__()
        self.n_fold_enc = len(backbones)
        self.n_fold_cla = n_fold_classifier
        self.features_size = features_size
        self.n_classes = n_classes
        
        self.fold_backbones = nn.ModuleList([
            FoldEncoder(features_size, backbone) for backbone in backbones
        ])

        self.fold_classifiers = nn.ModuleList([
            FoldClassifier(features_size, n_classes) for _ in range(n_fold_classifier)
        ])

        self.classifiers_weight = torch.ones((self.n_fold_cla, self.n_fold_enc), dtype=torch.float32)
        self.final_classifier_weight = nn.Parameter(torch.tensor([1. for _ in range(self.n_fold_cla)], dtype=torch.float32))

    def forward_encoders(self, crop):
        encodeds = torch.stack([backbone(crop) for backbone in self.fold_backbones], dim=1)
        return encodeds

    def forward(self, crop):
        final_output = list()
        _encodeds = self.forward_encoders(crop)
        for classifier in self.fold_classifiers:
            classified_ = torch.stack([classifier(_encodeds[:, i]) for i in range(self.n_fold_enc)], dim=1)
            classifier_output = torch.mean(classified_, dim=1)
            final_output.append(classifier_output)

        final_output = torch.stack(final_output, dim=1)
        final_output = torch.mean(final_output, dim=1)
        return final_output

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

def load_model_classification(model_path):
    model = FoldModelClassifier(
        n_classes=3,
        n_fold_classifier=3,
        backbones=['densenet201.tv_in1k', 'seresnext101_32x4d.gluon_in1k', 'convnext_base.fb_in22k_ft_in1k', 'dm_nfnet_f0.dm_in1k', 'mobilenetv3_small_100.lamb_in1k'],
        features_size=384,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    return model.eval().to(get_device())

def load_torch_script_model(path):
    return torch.jit.load(path, map_location="cpu").eval().to(get_device())

##########################################################
#
#   SLICES SELECTION INFERENCE
#
##########################################################

def get_max_consecutive3(predictions):
    index = 1
    max_index = predictions.size(0) - 1
    best_index = None
    best_sum = -1
    while index < max_index:
        current_sum = predictions[index - 1] + predictions[index] + predictions[index + 1]
        if current_sum > best_sum:
            best_sum = current_sum
            best_index = index
        index += 1
    
    indices = [best_index-1, best_index, best_index+1]
    values = [predictions[best_index-1], predictions[best_index], predictions[best_index+1]]
    return values, indices

def get_best_slice_selection(config, pathes, topk):
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
    preds = config['model_slice_selection'](images.to(config["device"]).unsqueeze(0)).squeeze()
    preds_overall = torch.sum(preds, dim=0)

    # get best by level
    slices_by_level = [
        {"pathes": list(), "values": list()} for _ in range(preds.shape[0])
    ]
    for level in range(preds.shape[0]):
        pred_level = preds[level, :]
        values, max_indice = get_max_consecutive3(pred_level)
        slices_by_level[level]['pathes'] = [pathes[i] for i in max_indice]
        slices_by_level[level]['values'] = [v for v in values]

    # get best overall (=> best after sum of each level)
    values, max_indices = get_max_consecutive3(preds_overall)
    best_slices_overall = dict()
    best_slices_overall['pathes'] = [pathes[i] for i in max_indices]
    best_slices_overall['values'] = [v for v in values]

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
        best_slices, best_slices_overall = get_best_slice_selection(config, slices_by_series[s_id], topk=3)
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
        images = torch.tensor(images).float().to(config["device"])
        return images
    
    elif config['segmentation_slice_selection'] == "best_by_level":
        images = np.zeros((5, 3, *config['segmentation_input_shape']))
        for level in range(5):
            _slices_to_use = slices_to_use['best_by_level'][level][:3] # 3 slice per level => the best ones
            _slices_to_use = sorted(_slices_to_use, key = lambda x: get_instance(x))
            for ch, slice_to_use in enumerate(_slices_to_use):
                im = cv2.resize(pydicom.dcmread(slice_to_use).pixel_array.astype(np.float32), 
                                config['segmentation_input_shape'],
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
        masks = config["model_segmentation"](inputs.unsqueeze(0)) # model predict 5 levels
        masks = masks.squeeze()
        position_by_level = [find_center_of_largest_activation(masks[i]) for i in range(5)]
    else:
        masks = config["model_segmentation"](inputs) # model predict 1 level, we put levels in batch dim
        masks = masks.squeeze(1)
        position_by_level = [find_center_of_largest_activation(masks[i]) for i in range(5)]
    return position_by_level

##########################################################
#
#   CLASSIFICATION INFERENCE
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

def get_crops_by_level(slices_to_use, position_by_level, config):
    crops_output = np.zeros((5, 3, 128, 128))
    for level, (slices_, position) in enumerate(zip(slices_to_use["best_by_level"], position_by_level)):
        if position is not None:
            slices = sorted(slices_[:config['classification_sequence_lenght']], key = lambda x : get_instance(x))
            px = int(position[0] * config['classification_resize_image'][0])
            py = int(position[1] * config['classification_resize_image'][0])
            crops_output[level, ...] = cut_crops(slices, px, py, config['classification_input_size'], config['classification_resize_image'])
        else:
            print("No prediction on segmentation !", level)

    crops_output = torch.tensor(crops_output).float().to(config["device"])
    return crops_output

def get_classification(crops: torch.tensor, config):
    preds = config['model_classification'](crops)
    preds = torch.softmax(preds, dim=1)
    return preds

##########################################################
#
#   MAIN INFERENCE
#
##########################################################

def predict_lumbar(df_description: pd.DataFrame, config: dict, study_id: int) -> list:
    try:
        series_ids = df_description[(df_description['study_id'] == study_id)
                                & (df_description['series_description'] == config['description'])]['series_id'].to_list()
        
        best_slices_by_level, best_slices_overall = get_slices_to_use(study_id, series_ids, config)
        slices_to_use = dict(
            best_by_level=best_slices_by_level,
            best_overall=best_slices_overall
        )

        positions_by_level = get_position_by_level(slices_to_use, config)

        # print(positions_by_level)

        crops_by_level = get_crops_by_level(slices_to_use, positions_by_level, config)
        classification_results = get_classification(crops_by_level, config)

        # print(classification_results)

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

def configure_inference(slice_model_path, seg_model_path, class_model, input_folder, description, condition, class_input_size, class_resize_image, seg_mode):
    return dict(
        device=get_device(),

        # slices infos
        model_slice_selection=load_torch_script_model(slice_model_path),
        slice_selection_input_shape=(224, 224),

        # segmentation infos
        segmentation_input_shape=(384, 384),
        segmentation_slice_selection=seg_mode,
        model_segmentation=load_torch_script_model(seg_model_path),

        # classification infos
        classification_input_size=class_input_size,
        classification_resize_image=class_resize_image,
        classification_sequence_lenght=5,
        model_classification=class_model,
        
        # general infos
        description=description,
        condition=condition,
        input_images_folder=input_folder,
    )

def compute_pipeline(input_images_folder, description_file, nb_studies_id=None):
    df_description = pd.read_csv(description_file)
    final_predictions = []

    tasks = [
        {
            "class_model_path": "classification/classification_spinal_canal_stenosis.pth",
            "slice_model_path": "trained_models/v2/model_slice_selection_st2.ts",
            "seg_model_path": "segmentation/model_segmentation_st2_384x384.ts",
            "description": "Sagittal T2/STIR",
            "condition": "Spinal Canal Stenosis",
            "class_input_size": (80, 120),
            "class_resize_image": (640, 640),
            "segmentation_slice_selection": "best_overall",
        },
        {
            "class_model_path": "classification/classification_right_neural_foraminal_narrowing.pth",
            "slice_model_path": "trained_models/v2/model_slice_selection_st1_right.ts",
            "seg_model_path": "segmentation/model_segmentation_st1_right_384x384.ts",
            "description": "Sagittal T1",
            "condition": "Right Neural Foraminal Narrowing",
            "class_input_size": (96, 144),
            "class_resize_image": (640, 640),
            "segmentation_slice_selection": "best_overall",
        },
        {
            "class_model_path": "classification/classification_left_neural_foraminal_narrowing.pth",
            "slice_model_path": "trained_models/v2/model_slice_selection_st1_left.ts",
            "seg_model_path": "segmentation/model_segmentation_st1_left_384x384.ts",
            "description": "Sagittal T1",
            "condition": "Left Neural Foraminal Narrowing",
            "class_input_size": (64, 96),
            "class_resize_image": (640, 640),
            "segmentation_slice_selection": "best_overall",
        },
        {
            "class_model_path": "classification/classification_right_subarticular_stenosis.pth",
            "slice_model_path": "trained_models/v2/model_slice_selection_axt2_right.ts",
            "seg_model_path": "segmentation/model_segmentation_axt2_right_384x384.ts",
            "description": "Axial T2",
            "condition": "Right Subarticular Stenosis",
            "class_input_size": (164, 164),
            "class_resize_image": (720, 720),
            "segmentation_slice_selection": "best_by_level",
        },
        {
            "class_model_path": "classification/classification_left_subarticular_stenosis.pth",
            "slice_model_path": "trained_models/v2/model_slice_selection_axt2_left.ts",
            "seg_model_path": "segmentation/model_segmentation_axt2_left_384x384.ts",
            "description": "Axial T2",
            "condition": "Left Subarticular Stenosis",
            "class_input_size": (164, 164),
            "class_resize_image": (720, 720),
            "segmentation_slice_selection": "best_by_level",
        },
    ]

    studies_id = df_description["study_id"].unique()
    task_configs = []
    for task in tasks[:1]:
        print(f'Loading models and configuring for task: {task["condition"]}')
        model_classification = load_model_classification(task["class_model_path"])
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
        task_configs.append(config)

    if nb_studies_id is None:
        nb_studies_id = len(studies_id)

    print(f"Launch pipeline on {nb_studies_id} studies !")
    for study_id in tqdm.tqdm(studies_id[:nb_studies_id], desc="Predicting for each study"):
        for config in task_configs:
            _predictions = predict_lumbar(df_description, config, study_id)
            final_predictions.extend(_predictions)
    return pd.DataFrame(final_predictions)

if __name__ == "__main__":

    df = compute_pipeline("../REFAIT/train_images/", "../REFAIT/train_series_descriptions.csv", 180)
    df.to_csv("spinal_preds.csv")