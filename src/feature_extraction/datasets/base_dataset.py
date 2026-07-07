from torch.utils.data import Dataset

class BaseFeatureDataset(Dataset):
    def __init__(self):
        self.frame_list = []
        self.obj_counts = {}

    def __len__(self):
        return len(self.frame_list)
        
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")

    @staticmethod
    def process_item(data, **kwargs):
        raise NotImplementedError("Subclasses must implement process_item")

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError("Subclasses must implement collate_fn")