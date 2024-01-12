import zipfile

zip_path = "dataset/img_align_celeba.zip"
extract_path = "dataset/data_faces"

with zipfile.ZipFile(zip_path,"r") as zip_ref:
  zip_ref.extractall(extract_path) # extract zip file to path