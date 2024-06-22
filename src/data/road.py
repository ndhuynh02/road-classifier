import glob
from PIL import Image
from torch.utils.data import Dataset

class Road(Dataset):
    def __init__(self, data_dir: str = 'data', data_type='train'):
        assert data_type in ['train', 'val', 'test']
        data_dir = data_dir + f"/{data_type}"
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

