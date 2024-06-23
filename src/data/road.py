import os
import glob
import gdown
from PIL import Image
from zipfile import ZipFile 
from torch.utils.data import Dataset

class Road(Dataset):
    def __init__(self, data_dir: str = 'data'):
        self.prepare_data(data_dir)

        self.dataset = glob.glob(pathname=data_dir + "/**/*.jpg", recursive=True)

        labels = glob.glob(pathname=data_dir + "/*")
        self.labels_map = {}
        for i, label in enumerate(labels):
            self.labels_map[label.split("/")[-1]] = i

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx]
        image = Image.open(image_path).convert("RGB")

        label = image_path.split("/")[-2]

        return image, self.labels_map[label]

    def prepare_data(self, data_dir):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """

        # if data if already download, do nothing
        if len([folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]) != 0:
            print("Data is downloaded")
            return

        file_path = os.path.join(data_dir, "road_dataset.zip")
        if not os.path.exists(file_path):
            id_ = "16JABnlow7gFhvyoQeKCleeJRvb1zm1F8"
            print("Downloading dataset...")
            gdown.download(id=id_, output=file_path)
        else:
            print("Data file is downloaded")

        print("Extracting...")
        with ZipFile(file_path) as zObject: 
            zObject.extractall(path=data_dir) 

        os.remove(file_path)  # delete zip file

        print("Download Finished!")


if __name__ == "__main__":
    data = Road()
    print(data.labels_map)
    print(data.labels_count)