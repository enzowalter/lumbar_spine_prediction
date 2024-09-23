import torch
import pydicom
import numpy as np
import cv2
import pandas as pd
import pickle
import tqdm
import glob

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_max_consecutive(preds, n):
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

def get_best_slice_selection(model, pathes, topk):
    """
    Return best slices for each level.
    Slice are not sorted by instance_number, they are sorted by topk
    """

    nb_slices = len(pathes)
    images = np.zeros((nb_slices, 1, 224, 224))
    for k, path in enumerate(pathes):
        im = cv2.resize(pydicom.dcmread(path).pixel_array.astype(np.float32), 
                                        (224, 224),
                                        interpolation=cv2.INTER_LINEAR)
        im = (im - im.min()) / (im.max() - im.min() + 1e-9)
        images[k, 0, ...] = im
    images = torch.tensor(images).expand(nb_slices, 3, 224, 224).float()
    images = images.to(get_device())
    model = model.to(get_device())
    preds_logits, preds_softmax = model(images.unsqueeze(0))
    preds_softmax = preds_softmax.squeeze()
    preds_overall = torch.sum(preds_softmax, dim=0)

    # get best by level
    slices_by_level = [
        {"pathes": list(), "values": list()} for _ in range(preds_softmax.shape[0])
    ]
    for level in range(preds_softmax.shape[0]):
        pred_level = preds_softmax[level, :]
        values, max_indice = get_max_consecutive(pred_level, n=topk)
        slices_by_level[level]['pathes'] = [pathes[i] for i in max_indice]
        slices_by_level[level]['values'] = [v for v in values]

    # get best overall (=> best after sum of each level)
    values, max_indices = get_max_consecutive(preds_overall, n=topk)
    best_slices_overall = dict()
    best_slices_overall['pathes'] = [pathes[i] for i in max_indices]
    best_slices_overall['values'] = [v for v in values]

    return slices_by_level, best_slices_overall

def run(input_dir, slice_model, condition, description):
    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}

    df_study_descriptions = pd.read_csv(f"{input_dir}/train_series_descriptions.csv")
    studies_id = df_study_descriptions["study_id"].unique()

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Generates slices hotmap"):

        series_id = df_study_descriptions[(df_study_descriptions['study_id'] == study_id)
                                        & (df_study_descriptions['series_description'] == description)]['series_id'].to_list()
        
        for s_id in series_id:
                slices_path = sorted(glob.glob(f"{input_dir}/train_images/{study_id}/{s_id}/*.dcm"), key = lambda x : get_instance(x))
                best_by_level, best_overall = get_best_slice_selection(slice_model, slices_path, topk=3)

    return dataset


if __name__ == "__main__":

    # with open("../trained_models/v6/model_slice_selection_st1_left.pkl", 'rb') as f:
    #     model = pickle.load(f)
    
    # model = model.eval().to(get_device())
    # run("../", model, "Left Neural Foraminal Narrowing", "Sagittal T1")


    model = torch.jit.load("../trained_models/v6/model_slice_selection_st1_left_scripted.ts")
    model = model.eval()
    model = model.to(get_device())
    run("../", model, "Left Neural Foraminal Narrowing", "Sagittal T1")