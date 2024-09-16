from train_model import train_submodel

if __name__ == "__main__":

    model_names = [
        'classification_left_neural_foraminal_narrowing.pth',
        'classification_right_neural_foraminal_narrowing.pth',
        'classification_spinal_canal_stenosis.pth',
        'classification_left_subarticular_stenosis.pth',
        'classification_right_subarticular_stenosis.pth',
    ]
    conditions = [
            "Left Neural Foraminal Narrowing", 
            "Right Neural Foraminal Narrowing", 
            "Spinal Canal Stenosis", 
            "Left Subarticular Stenosis", 
            "Right Subarticular Stenosis"
        ]
    descriptions = ["Sagittal T1", "Sagittal T1", "Sagittal T2/STIR", "Axial T2", "Axial T2"]
    crop_sizes = [(96, 128), (96, 128), (96, 128), (164, 164), (164, 164)]
    image_resizes = [(640, 640), (640, 640), (640, 640), (800, 800), (800, 800)]

    for name, cond, desc, crop_size, image_resize in zip(model_names, conditions, descriptions, crop_sizes, image_resizes):
        train_submodel(
                    input_dir="../../REFAIT",
                    model_name=name,
                    crop_condition=cond,
                    label_condition=cond,
                    crop_description=desc,
                    crop_size=crop_size,
                    image_resize=image_resize,
            )
        break
