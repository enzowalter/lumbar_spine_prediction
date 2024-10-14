import numpy as np
import pandas as pd
import tqdm
import pydicom
import glob

def get_instance(path):
    return int(path.split("/")[-1].split('.')[0])

def get_coordinates_infos(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    H,W = dicom.pixel_array.shape
    sx,sy,sz = [float(v) for v in dicom.ImagePositionPatient]
    o0, o1, o2, o3, o4, o5, = [float(v) for v in dicom.ImageOrientationPatient]
    delx,dely = dicom.PixelSpacing

    xx =  o0*delx*coordinate['x'] + o3*dely*coordinate['y'] + sx
    yy =  o1*delx*coordinate['x'] + o4*dely*coordinate['y'] + sy
    zz =  o2*delx*coordinate['x'] + o5*dely*coordinate['y'] + sz

    return H, W, xx, yy, zz

def backproject_XYZ(xx, yy, zz, dicom_data, nb_slices):

    sx, sy, sz = [float(v) for v in dicom_data.ImagePositionPatient]
    o0, o1, o2, o3, o4, o5, = [float(v) for v in dicom_data.ImageOrientationPatient]
    delx, dely = dicom_data.PixelSpacing
    delz = dicom_data.SpacingBetweenSlices

    ax = np.array([o0,o1,o2])
    ay = np.array([o3,o4,o5])
    az = np.cross(ax,ay)

    p = np.array([xx-sx,yy-sy,zz-sz])
    x = np.dot(ax, p)/delx
    y = np.dot(ay, p)/dely
    z = np.dot(az, p)/delz
    x = int(round(x))
    y = int(round(y))
    z = int(round(z))

    D,H,W = nb_slices, dicom_data.pixel_array.shape[0], dicom_data.pixel_array.shape[1]
    inside = \
        (x>=0) & (x<W) &\
        (y>=0) & (y<H) &\
        (z>=0) & (z<D)
    if not inside:
        return False,0,0,0,0
    return True, x, y, z

if __name__ == "__main__":

    LEVELS = {"L1/L2":0, "L2/L3":1, "L3/L4":2, "L4/L5":3, "L5/S1":4}

    df_study_labels = pd.read_csv(f"../train.csv")
    df_study_coordinates = pd.read_csv(f"../train_label_coordinates.csv")
    df_study_descriptions = pd.read_csv(f"../train_series_descriptions.csv")
    studies_id = df_study_labels["study_id"].to_list()

    nb_ok = 0
    nb_ko = 0
    nb_diff = 0

    dataset = list()
    for study_id in tqdm.tqdm(studies_id, desc="Get sagittal instance"):

        coordinates = df_study_coordinates[(df_study_coordinates['study_id'] == study_id) & (df_study_coordinates['condition'] == "Spinal Canal Stenosis")]
        coordinates_axial = df_study_coordinates[(df_study_coordinates['study_id'] == study_id) & (df_study_coordinates['condition'] == "Left Subarticular Stenosis")]

        series_axial = df_study_descriptions[(df_study_descriptions['study_id'] == study_id) & (df_study_descriptions['series_description'] == "Axial T2")]
        series_axial = series_axial['series_id'].unique()

        for coordinate in coordinates.to_dict('records'):

            corresponding_axial = None
            for c in coordinates_axial.to_dict('records'):
                if c['level'] == coordinate['level']:
                    corresponding_axial = c
                    break

            if corresponding_axial is not None:
                H, W, xx, yy, zz = get_coordinates_infos(f"../train_images/{study_id}/{coordinate['series_id']}/{coordinate['instance_number']}.dcm")
                for s_id in series_axial:
                    if s_id == corresponding_axial['series_id']:
                        slices_axial = glob.glob(f"../train_images/{study_id}/{s_id}/*.dcm")
                        slices_axial = sorted(slices_axial, key = get_instance)

                        for s_a in slices_axial:
                            dicom = pydicom.dcmread(s_a)
                            lapin = backproject_XYZ(xx, yy, zz, dicom, len(slices_axial))
                            if lapin[0]:
                                if get_instance(s_a) == corresponding_axial['instance_number']:
                                    nb_ok += 1
                                else:
                                    nb_ko += 1
                                    nb_diff += abs(get_instance(s_a) - corresponding_axial['instance_number'])
                                break

    print("Accuracy = ", nb_ok / (nb_ko + nb_ok))
    print("Mean error = ", nb_diff / (nb_ko))
