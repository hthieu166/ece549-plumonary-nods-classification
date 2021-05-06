from radiomics import featureextractor, getTestCase
import os.path as osp
import glob
import pandas as pd
import six
import os
import pickle as pkl
from multiprocessing import Pool
# root_dir  = "/home/hthieu/data/LIDC_IDRI/segmentations/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"

# imageName = osp.join(root_dir, "LIDC-IDRI-0001_CT.nrrd")
# maskName  = osp.join(root_dir, "Nodule 1 - Annotation 0.nrrd")
# 
# 
SEG_DIR = "/home/hthieu/data/LIDC_IDRI/segmentations/"
OUT_DIR = "/home/hthieu/data/LIDC_IDRI/pyradiomics_features/"

def process_one_case(annot_dict, case_id):
  case_name = "LIDC-IDRI-%04d" %case_id
  print("Processing %s" %case_name)
  case_dir  = osp.join(SEG_DIR, case_name)
  out_dir   = osp.join(OUT_DIR, case_name)
  nrrd_files= glob.glob(osp.join(case_dir, "*/*/*.nrrd"))
  nrrd_files= {osp.basename(i).replace(".nrrd",""): i for i in nrrd_files}
  nrrd_annot_files = {i.split("-")[1][12:]: nrrd_files[i] for i in nrrd_files if i != case_name}
  nrrd_image_file= nrrd_files["LIDC-IDRI-%04d_CT" %case_id]
  
  for roi in annot_dict[case_id]:
    roi_out_dir = osp.join(out_dir, str(roi))
    os.makedirs(roi_out_dir, exist_ok=True)
    extractor = featureextractor.RadiomicsFeatureExtractor(correctMask = True)
    for annot in annot_dict[case_id][roi]:
      if annot in nrrd_annot_files:
        result  = extractor.execute(nrrd_image_file, nrrd_annot_files[annot])
        with open(osp.join(roi_out_dir, "%s.pkl" % annot), "wb") as fo:
          pkl.dump(result, fo)
  # print(annot_dict[3])
  # imageName = glob.glob(osp.join(SEG_DIR,))

def read_csv(list32_path):
  annot_dict = {}
  df = pd.read_csv(list32_path)
  for i in range(len(df)):
    case_id = df.iloc[i,0]
    roi_id  = df.iloc[i,2]
    if case_id not in annot_dict:
      annot_dict[case_id] = {}
    annot_dict[case_id][roi_id] = df.iloc[i,9:][df.iloc[i,9:].notnull()].tolist()
  return annot_dict

def read_feats(file):
  with open(file, "rb") as fi:
    res = pkl.load(fi)
  print(file)
  for key, val in six.iteritems(res):
    print("\t%s: %s" %(key, val))
  return res

annot_dict = read_csv("list3.2.csv")
def func(case_id):
  process_one_case(annot_dict, case_id)
  # print(case_id)

def run_main():
  # process_one_case(annot_dict, case_id)
  n_workers = 8
  todo = sorted(list(annot_dict.keys()))
  with Pool(n_workers) as p:
    # print(annot_dict.keys())
    p.map(func,todo)

if __name__ == "__main__":
  process_one_case(annot_dict, 311)
  # run_main()
  # files = sorted(glob.glob(osp.join(OUT_DIR, "LIDC-IDRI-0003/1/*.pkl")))
  # for f in files:
  #   read_feats(f)