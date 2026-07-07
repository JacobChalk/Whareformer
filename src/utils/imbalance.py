from torch.utils.data import Dataset

def compute_new_tracks(dataset: Dataset):
    labels = dataset.get_class_counts()
    assigned_detections = sum(labels)
    new_tracks = len(labels) - assigned_detections
    print(f"Assigned Detections: {assigned_detections}. New Tracks: {new_tracks}")
