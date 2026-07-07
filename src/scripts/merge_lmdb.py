import argparse
import lmdb
import glob
import os

from tqdm import tqdm

def merge_lmdbs(source_dir: str, dest_path: str):
    if not os.path.exists(source_dir):
        print(f"Directory: {source_dir} does not exist")
        return

    source_paths = glob.glob(os.path.join(source_dir, '**', '*.lmdb'), recursive=True)
    if not source_paths:
        print(f"No .lmdb databases found in {source_dir}")
        return
    
    print(f"Found {len(source_paths)} LMDB databases to merge.")

    total_size = 0
    for path in source_paths:
        data_file = os.path.join(path, 'data.mdb')
        if os.path.exists(data_file):
            total_size += os.path.getsize(data_file)
    
    dest_map_size = int(total_size * 2.0)
    print(f"Estimated total size: {total_size / (1024**3):.2f} GB. Setting map_size to {dest_map_size / (1024**3):.2f} GB.")

    dest_env = lmdb.open(dest_path, map_size=dest_map_size)

    with dest_env.begin(write=True) as dest_txn:
        total_keys = 0
        for path in tqdm(source_paths, desc="Merging Databases"):
            try:
                src_env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
                with src_env.begin() as src_txn:
                    cursor = src_txn.cursor()
                    for key, value in cursor:
                        dest_txn.put(key, value)
                        total_keys += 1
                src_env.close()
            except lmdb.Error as e:
                print(f"Could not process {path}: {e}")

    print(f"\nMerge complete. Total keys written: {total_keys}")
    dest_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Directory to .lmdb folders")
    parser.add_argument("--split", type=str, choices=['all', 'train', 'test'],
                        default='all', help="What data split(s) to use")

    args = parser.parse_args()
    if args.split == 'all':
        splits = ['train', 'test']
    else:
        splits = [args.split]
    for split in splits:
        split_directory = f'{args.source_dir}/{split}'
        lmdb_out_path = f'{split_directory}.lmdb'    
        merge_lmdbs(split_directory, lmdb_out_path)