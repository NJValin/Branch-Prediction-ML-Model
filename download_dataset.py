import kaggle
import os

if not os.listdir("./csv/dataset_A/"):
    kaggle.api.dataset_download_files("pradhyumna2021/branch-prediction", path="./csv/dataset_A/", unzip=True)

if not os.listdir("./csv/dataset_B/"):
    kaggle.api.dataset_download_files("dmitryshkadarevich/branch-prediction", path="./csv/dataset_B/", unzip=True)


