import timm
import torch
import pandas as pd
import tqdm

models = [
    # 'cspresnet50.ra_in1k', 
    # 'convnext_base.fb_in22k_ft_in1k', 
    # 'ese_vovnet39b.ra_in1k', 
    # 'densenet161.tv_in1k', 
    # 'dm_nfnet_f0.dm_in1k',
    # 'ecaresnet26t.ra2_in1k',
    # 'seresnet50.a1_in1k',
    # 'resnet26t.ra2_in1k',
    # 'mobilenetv3_small_100.lamb_in1k',
    # 'efficientnet_b0.ra_in1k',
    # 'efficientnet_b3.ra2_in1k',
    # 'xception65.ra3_in1k',
    # 'resnet50.a1_in1k',
    # 'inception_v3',
    # 'regnety_002.pycls_in1k',
    # 'ese_vovnet19b_dw.ra_in1k',
    # 'densenet201.tv_in1k',
    # 'seresnext101_32x4d.gluon_in1k',
    # 'ecaresnet50d.miil_in1k',
    # 'mobilenetv3_large_100.ra_in1k',
    # 'tf_efficientnetv2_s.in21k_ft_in1k',
    # 'xcit_small_12_p16_224.fb_dist_in1k',
    # 'dpn68.mx_in1k',
    # 'regnetz_c16.ra3_in1k',
    # 'res2net50_26w_4s.in1k',
    # 'xception41p.ra3_in1k',
    # 'convnext_tiny.in12k_ft_in1k',
    # 'convnext_base.fb_in22k_ft_in1k',
    # 'mobilenetv2_100.ra_in1k',
    # 'resnet34.a1_in1k'
    # 'densenet169.tv_in1k',
    # 'efficientnet_b2.ra_in1k',
    # 'resnext50_32x4d.a1h_in1k',
    # 'densenet121.tv_in1k',
    # 'resnet101.a1h_in1k',
]
# List all models
all_models = timm.list_models(pretrained=True)

# Keywords to exclude (transformers and non-vision models)
exclude_keywords = ['vit', 'swin', 'beit', 'deit', 'tnt', 'mlp', 'mixer', 'transformer', 'nlp', 'eva']

# Filter models: exclude those that contain transformer or NLP keywords
vision_models = [model for model in all_models if not any(keyword in model.lower() for keyword in exclude_keywords)]

dummy_input = torch.randn((1, 3, 128, 128)).cuda()
output = list()

for k, model_name in tqdm.tqdm(enumerate(vision_models), desc="Checking models", total=len(vision_models)):
    model = timm.create_model(model_name=model_name, pretrained=False)
    param_count = sum(p.numel() for p in model.parameters())
    if param_count / 1000000 > 200: # no model with more than 200m param
        output.append({
            'name': model_name,
            'features': model.num_features,
            'fail': "too_big",
        })
    else:
        output.append({
            'name': model_name,
            'features': model.num_features,
            'fail': "idk",
        })

pd.DataFrame(output).to_csv("models_num_features.csv", index=False)

# 2048
# ecaresnet26t.ra2_in1k
# seresnet50.a1_in1k
# resnet26t.ra2_in1k
# xception65.ra3_in1k
# resnet50.a1_in1k
# inception_v3
# seresnext101_32x4d.gluon_in1k
# res2net50_26w_4s.in1k

# 1024
# cspresnet50.ra_in1k
# convnext_base.fb_in22k_ft_in1k
# densenet121.tv_in1k
# ese_vovnet39b.ra_in1k
# ese_vovnet19b_dw.ra_in1k

"""
cspresnet50.ra_in1k 1024
torch.Size([1, 1024, 4, 4])
--------------------------------------------------
convnext_base.fb_in22k_ft_in1k 1024
torch.Size([1, 1024, 4, 4])
--------------------------------------------------
ese_vovnet39b.ra_in1k 1024
torch.Size([1, 1024, 4, 4])
--------------------------------------------------
densenet161.tv_in1k 2208
torch.Size([1, 2208, 4, 4])
--------------------------------------------------
dm_nfnet_f0.dm_in1k 3072
torch.Size([1, 3072, 4, 4])
--------------------------------------------------
ecaresnet26t.ra2_in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------
seresnet50.a1_in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------
resnet26t.ra2_in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------
mobilenetv3_small_100.lamb_in1k 576
torch.Size([1, 576, 4, 4])
--------------------------------------------------
efficientnet_b0.ra_in1k 1280
torch.Size([1, 1280, 4, 4])
--------------------------------------------------
efficientnet_b3.ra2_in1k 1536
torch.Size([1, 1536, 4, 4])
--------------------------------------------------
xception65.ra3_in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------
resnet50.a1_in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------
inception_v3 2048
torch.Size([1, 2048, 2, 2])
--------------------------------------------------
regnety_002.pycls_in1k 368
torch.Size([1, 368, 4, 4])
--------------------------------------------------
ese_vovnet19b_dw.ra_in1k 1024
torch.Size([1, 1024, 4, 4])
--------------------------------------------------
densenet201.tv_in1k 1920
torch.Size([1, 1920, 4, 4])
--------------------------------------------------
seresnext101_32x4d.gluon_in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------
ecaresnet50d.miil_in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------
mobilenetv3_large_100.ra_in1k 960
torch.Size([1, 960, 4, 4])
--------------------------------------------------
tf_efficientnetv2_s.in21k_ft_in1k 1280
torch.Size([1, 1280, 4, 4])
--------------------------------------------------
xcit_small_12_p16_224.fb_dist_in1k 384
torch.Size([1, 65, 384])
--------------------------------------------------
dpn68.mx_in1k 832
torch.Size([1, 832, 4, 4])
--------------------------------------------------
regnetz_c16.ra3_in1k 1536
torch.Size([1, 1536, 4, 4])
--------------------------------------------------
res2net50_26w_4s.in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------
xception41p.ra3_in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------
convnext_tiny.in12k_ft_in1k 768
torch.Size([1, 768, 4, 4])
--------------------------------------------------
mobilenetv2_100.ra_in1k 1280
torch.Size([1, 1280, 4, 4])
--------------------------------------------------
resnet34.a1_in1k 512
torch.Size([1, 512, 4, 4])
--------------------------------------------------
densenet169.tv_in1k 1664
torch.Size([1, 1664, 4, 4])
--------------------------------------------------
efficientnet_b2.ra_in1k 1408
torch.Size([1, 1408, 4, 4])
--------------------------------------------------
resnext50_32x4d.a1h_in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------
densenet121.tv_in1k 1024
torch.Size([1, 1024, 4, 4])
--------------------------------------------------
resnet101.a1h_in1k 2048
torch.Size([1, 2048, 4, 4])
--------------------------------------------------

"""