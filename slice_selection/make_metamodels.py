import torch
import torch.nn as nn

from models import SliceSelecterModelSqueezeNet

class SliceSelecterMetamodel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, images):
        outputs = list()
        for model in self.models:
            preds_probs, preds_logits = model(images)
            outputs.append(preds_logits)
        outputs = torch.stack(outputs, dim = 1)
        outputs = torch.mean(outputs, dim = 1)
        return outputs.sigmoid()

def make_metamodel(condition, nb_steps, out_name):
    models = list()
    for i in range(nb_steps):
        _model_name = f"trained_models/{condition.lower().replace(' ', '_')}_step_{i}.pth"
        _model = SliceSelecterModelSqueezeNet()
        _model.load_state_dict(torch.load(_model_name, map_location='cpu', weights_only=True))
        models.append(_model)

    metamodel = SliceSelecterMetamodel(models)
    scripted_model = torch.jit.script(metamodel)
    scripted_model.save(out_name)
    print("Metamodel generated:", out_name)

conditions = [
    "Left Neural Foraminal Narrowing", 
    "Right Neural Foraminal Narrowing", 
    "Spinal Canal Stenosis", 
    "Left Subarticular Stenosis", 
    "Right Subarticular Stenosis", 
]
out_names = [
    "slice_selector_st1_left_metamodel.ts", 
    "slice_selector_st1_right_metamodel.ts", 
    "slice_selector_st2_metamodel.ts", 
    "slice_selector_ax_left_metamodel.ts", 
    "slice_selector_ax_right_metamodel.ts"
]

for cond, out in zip(conditions, out_names):
    make_metamodel(condition=cond, nb_steps=4, out_name=out)