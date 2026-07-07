import json

def load_object_data(obj_data_path, obj_identifier='name'):
    obj_masks = {}
    obj_data = json.load(open(obj_data_path, 'r'))['video_annotations']

    for entry in obj_data:
        frame_name = entry['frame_name']

        obj_list = []
        bbox_list = []

        annotations = entry.get('annotations', [])
        for annotation in annotations:
            obj_list.append(annotation[obj_identifier])
            bbox_list.append(annotation['bounding_box'])

        obj_masks[frame_name] = (obj_list, bbox_list)
    
    return obj_masks