import numpy as np
import pandas as pd
import os.path as osp
import os
import SimpleITK as sitk
import glob
from tqdm import tqdm
import ipdb
import sys
import os
sys.path.insert(0, os.path.abspath(".."))
from data.data_enhancement import horizontal_flip, vertical_flip, deep_flip



LUNA16_PATH     = "/home/hthieu/cig_luna16_experiments/luna16/subsets"
ANNOTATION_CSV  = "/home/hthieu/plumonary_nods_classification/NAS-Lung/data/annotationdetclsconvfnl_v3.csv"
ANNOTATION_CSV_AUGMENT  = "/home/hthieu/plumonary_nods_classification/NAS-Lung/data/annotationdetclsconvfnl_v3_enhance.csv"
CROPSIZE        = 32
DCOM_EXT        = ".mhd"
CROP_PATH       = "/media/DATA/LUNA16/crop"
AUGMENT_PATH    = "/media/DATA/LUNA16/crop_augment/"
PREPROCESS_PATH = "/media/DATA/LUNA16/preprocess/"

def read_annotation_csv(inp_csv):
    annot_df = pd.read_csv(inp_csv,
        names = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
    srslst  = annot_df['seriesuid'].tolist()[1:]
    crdxlst = annot_df['coordX'].tolist()[1:]
    crdylst = annot_df['coordY'].tolist()[1:]
    crdzlst = annot_df['coordZ'].tolist()[1:]
    dimlst  = annot_df['diameter_mm'].tolist()[1:]
    mlglst  = annot_df['malignant'].tolist()[1:]
    
    newlst  = [] 
    for i in range(len(srslst)):
        newlst.append([srslst[i], crdxlst[i], crdylst[i], crdzlst[i], dimlst[i], mlglst[i]])
    return newlst

def load_dcom_image(file_name):
    itkImage = sitk.ReadImage(file_name)
    return sitk.GetArrayFromImage(itkImage).transpose(2,1,0)

def get_images_path_in_subsets(root_subsets, ext = "_clean.npy"):
    all_files = glob.glob(osp.join(root_subsets, "*/*" + ext))
    dcom_path = {}
    for file in all_files:
        pid = osp.basename(file)
        dcom_path[pid] = file
    return dcom_path

def do_cropping(preprocess_dir, annot_csv, out_path):
    #Get path to dcom files
    images_path_dict  = get_images_path_in_subsets(preprocess_dir)
    #Read info from annotation files
    newlst          = read_annotation_csv(annot_csv)
    #Make output directory
    os.makedirs(out_path, exist_ok = True)

    error_lst = []
    for idx in tqdm(range(len(newlst))):
        fname = newlst[idx][0]
        # if fname != '1.3.6.1.4.1.14519.5.2.1.6279.6001.119209873306155771318545953948-581': continue
        pid = fname.split('-')[0]
        crdx = int(float(newlst[idx][1]))
        crdy = int(float(newlst[idx][2]))
        crdz = int(float(newlst[idx][3]))
        dim = int(float(newlst[idx][4]))
        try:
            data = np.load(images_path_dict[pid + '_clean.npy'])
        except:
            error_lst.append(images_path_dict[pid + '_clean.npy'])
            continue
        bgx = int(max(0, crdx - CROPSIZE / 2))
        bgy = int(max(0, crdy - CROPSIZE / 2))
        bgz = int(max(0, crdz - CROPSIZE / 2))

        cropdata = np.ones((CROPSIZE, CROPSIZE, CROPSIZE)) * 170
        cropdatatmp = np.array(data[0, bgx:bgx + CROPSIZE, bgy:bgy + CROPSIZE, bgz:bgz + CROPSIZE])
        cropdata[
            int(CROPSIZE / 2 - cropdatatmp.shape[0] / 2): int(CROPSIZE / 2 - cropdatatmp.shape[0] / 2 + cropdatatmp.shape[0]), \
            int(CROPSIZE / 2 - cropdatatmp.shape[1] / 2): int(CROPSIZE / 2 - cropdatatmp.shape[1] / 2 + cropdatatmp.shape[1]), \
            int(CROPSIZE / 2 - cropdatatmp.shape[2] / 2): int(CROPSIZE / 2 - cropdatatmp.shape[2] / 2 + cropdatatmp.shape[2])] \
            = np.array(2 - cropdatatmp)
        assert cropdata.shape[0] == CROPSIZE and cropdata.shape[1] == CROPSIZE and cropdata.shape[2] == CROPSIZE
        np.save(os.path.join(out_path, fname + '.npy'), cropdata)
    with open("error.txt", "w") as fo:
        fo.write("\n".join(error_lst))

def do_data_augmentation(annot_csv_crop_nod, cropped_path, out_augment_path):
    #Read info from annotation files
    annot_df  = pd.read_csv(annot_csv_crop_nod)
    nods_lst  = annot_df["seriesuid"].tolist() 
    #Make output directory
    os.makedirs(out_augment_path, exist_ok = True)
    def saver(file_name, data):
        out_path = osp.join(out_augment_path, file_name)
        np.save(out_path, data)
    def enhance(func, aug_name, nod_name, nod_crop):
        #Do data augmentation
        augmented_data = func(nod_crop)
        #Get new file name
        augmented_file_name = "{}{}".format(nod_name, aug_name)
        #Save new data augmentation
        saver("{}.npy".format(augmented_file_name), augmented_data)
        return augmented_file_name

    for idx in tqdm(range(len(nods_lst))):
        nod_name = nods_lst[idx]
        # if fname != '1.3.6.1.4.1.14519.5.2.1.6279.6001.119209873306155771318545953948-581': continue
        nod_crop = np.load(osp.join(cropped_path, "{}.npy".format(nod_name)))
        enhance(horizontal_flip, "horizontal", nod_name, nod_crop)
        enhance(vertical_flip, "vertical", nod_name, nod_crop)
        enhance(deep_flip, "deep", nod_name, nod_crop)
        #copy the original
        saver("{}.npy".format(nod_name), nod_crop)
        # annot_df.append(annot_data)



if __name__ == "__main__":
    do_cropping(
        preprocess_dir=PREPROCESS_PATH,
        annot_scv=ANNOTATION_CSV,
        out_path =CROP_PATH)

    do_data_augmentation(annot_csv_crop_nod=ANNOTATION_CSV,
                        cropped_path=CROP_PATH,
                        out_augment_path=AUGMENT_PATH
                        )
    # generated_file = [osp.basename(i).replace(".npy","") for i in glob.glob(osp.join(AUGMENT_PATH,"*.npy"))]
    # augm_df = pd.read_csv(ANNOTATION_CSV_AUGMENT)
    # file_in_lbl = augm_df["seriesuid"].tolist()
    # missing = 0
    # for nod_id in file_in_lbl:
    #     if nod_id not in generated_file:
    #         missing += 1
    #         print(nod_id)
    # print(missing)