import os
from torch.utils.data import Dataset
from PIL import Image

class RPSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Mapeo de nombre de carpeta -> etiqueta numérica
        self.class_to_idx = {"rock": 0, "paper": 1, "scissors": 2}

        for clase, idx in self.class_to_idx.items():
            carpeta_clase = os.path.join(root_dir, clase)
            for nombre_archivo in os.listdir(carpeta_clase):
                ruta_completa = os.path.join(carpeta_clase, nombre_archivo)
                self.image_paths.append(ruta_completa)
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        ruta = self.image_paths[idx]
        etiqueta = self.labels[idx]

        img = Image.open(ruta).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, etiqueta