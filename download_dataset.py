import kaggle
import os
##############################################################################################################
# Requires you to have kaggle api key. Create on on kaggle, and put the generated kaggle.json
# in the directory ~/.kaggle/
##############################################################################################################

if not os.listdir("./csv/dataset_A/"):
    kaggle.api.dataset_download_files("pradhyumna2021/branch-prediction", path="./csv/dataset_A/", unzip=True)
else:
    print("Already Downloaded dataset A!")

if not os.listdir("./csv/dataset_B/"):
    kaggle.api.dataset_download_files("dmitryshkadarevich/branch-prediction", path="./csv/dataset_B/", unzip=True)
else:
    print("Already Downloaded dataset B!")
