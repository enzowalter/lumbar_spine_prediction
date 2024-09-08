from train import train_model

if __name__ == "__main__":
    m1 = train_model(
                "../../REFAIT",
                ["Spinal Canal Stenosis"],
                "Sagittal T2/STIR",
                "../trained_models/v0/model_slice_selection_st2.ts",
                (128, 256),
                (700, 700),
                "model_classification_st2.pth",
            )
    m2 = train_model(
                "../../REFAIT",
                ["Right Neural Foraminal Narrowing"],
                "Sagittal T1",
                "../trained_models/v0/model_slice_selection_st1_right.ts",
                (128, 256),
                (700, 700),
                "model_classification_st1_right.pth",
            )
    m3 = train_model(
                "../../REFAIT",
                ["Left Neural Foraminal Narrowing"],
                "Sagittal T1",
                "../trained_models/v0/model_slice_selection_st1_left.ts",
                (128, 256),
                (700, 700),
                "model_classification_st1_left.pth",
            )
    m4 = train_model(
                "../../REFAIT",
                ["Right Subarticular Stenosis"],
                "Axial T2",
                "../trained_models/v0/model_slice_selection_axt2_right.ts",
                (224, 224),
                (600, 600),
                "model_classification_axt2_right.pth",
            )
    m5 = train_model(
                "../../REFAIT",
                ["Left Subarticular Stenosis"],
                "Axial T2",
                "../trained_models/v0/model_slice_selection_axt2_left.ts",
                (224, 224),
                (600, 600),
                "model_classification_axt2_left.pth",
            )
    
    print("Sagittal t2", m1)
    print("Sagittal t1 right", m2)
    print("Sagittal t2 left", m3)
    print("Sagittal ax2 right", m4)
    print("Sagittal ax 2 left", m5)

