import tqdm
import torch
import glob

from models_v2 import *

if __name__ == "__main__":

    out_name = glob.glob("trained_axial/*.pth")
    backbones = [
        'hgnet_tiny.paddle_in1k', 
        'densenet201.tv_in1k', 
        'regnety_008.pycls_in1k', 
        'focalnet_tiny_lrf.ms_in1k', 
        'convnext_base.fb_in22k_ft_in1k',
        'seresnext101_32x4d.gluon_in1k',
    ]

    for out in tqdm.tqdm(out_name, desc="Converting"):

        model = REM_torchscript(
            n_classes=3,
            n_classifiers=3,
            unification_size=768,
            backbones=backbones,
        )

        model.load_state_dict(torch.load(out, weights_only=True, map_location='cpu'))
        scripted_model = torch.jit.script(model)
        scripted_model.save(out.replace(".pth", ".ts"))
