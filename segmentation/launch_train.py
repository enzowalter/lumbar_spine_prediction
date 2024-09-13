from train_sagittal import train_model_sagittal
from train_axial import train_model_axial

if __name__ == "__main__":
    """
    m1 = train_model_sagittal(
                "../../REFAIT",
                ["Spinal Canal Stenosis"],
                "Sagittal T2/STIR",
                "../trained_models/v0/model_slice_selection_st2.ts",
                "model_segmentation_st2.ts",
            )
    m2 = train_model_sagittal(
                "../../REFAIT",
                ["Right Neural Foraminal Narrowing"],
                "Sagittal T1",
                "../trained_models/v0/model_slice_selection_st1_right.ts",
                "model_segmentation_st1_right.ts",
            )
    m3 = train_model_sagittal(
                "../../REFAIT",
                ["Left Neural Foraminal Narrowing"],
                "Sagittal T1",
                "../trained_models/v0/model_slice_selection_st1_left.ts",
                "model_segmentation_st1_left.ts",
            )
    """
    m4 = train_model_axial(
                "../../REFAIT",
                ["Right Subarticular Stenosis"],
                "Axial T2",
                "../trained_models/v1/model_slice_selection_axt2_right.ts",
                "model_segmentation_axt2_right.ts",
            )
    m5 = train_model_axial(
                "../../REFAIT",
                ["Left Subarticular Stenosis"],
                "Axial T2",
                "../trained_models/v1/model_slice_selection_axt2_left.ts",
                "model_segmentation_axt2_left.ts",
            )
    #print("Sagittal t2", m1)
    #print("Sagittal t1 right", m2)
    #print("Sagittal t2 left", m3)
    print("Sagittal ax2 right", m4)
    print("Sagittal ax 2 left", m5)

