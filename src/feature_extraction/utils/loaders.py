from torch.utils.data import Sampler
from concurrent.futures import ThreadPoolExecutor

class ObjectCountBatchSampler(Sampler):
    def __init__(self, dataset, target_object_count):
        self.dataset = dataset
        self.target_object_count = target_object_count
        
        self.object_counts = [self.dataset.obj_counts[f_name] for f_name in self.dataset.frame_list]        
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        batch = []
        current_object_sum = 0
        
        for idx in self.indices:
            object_count = self.object_counts[idx]
            
            if object_count > self.target_object_count:
                if batch:
                    yield batch
                yield [idx]
                batch = []
                current_object_sum = 0
                continue

            if current_object_sum + object_count > self.target_object_count:
                if not batch:
                    yield [idx]
                    continue
                yield batch
                batch = [idx]
                current_object_sum = object_count
            else:
                batch.append(idx)
                current_object_sum += object_count
        
        if batch:
            yield batch

    def __len__(self):
        count = 0
        current_sum = 0
        for oc in self.object_counts:
            if oc > self.target_object_count:
                if current_sum > 0:
                    count += 1
                count += 1
                current_sum = 0
            elif current_sum + oc > self.target_object_count:
                count += 1
                current_sum = oc
            else:
                current_sum += oc
        if current_sum > 0:
            count += 1
        return count


class ThreadPoolDataLoader:
    def __init__(self, dataset, batch_sampler, collate_fn, worker_fn, num_workers=None):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.num_workers = num_workers
        self.worker_fn = worker_fn

    def __iter__(self):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            sampler_iter = iter(self.batch_sampler)
            
            try:
                first_indices = next(sampler_iter)
                futures_N = [executor.submit(self.worker_fn, self.dataset[i]) for i in first_indices]
            except StopIteration:
                return 

            for indices_N_plus_1 in sampler_iter:
                futures_N_plus_1 = [executor.submit(self.worker_fn, self.dataset[i]) for i in indices_N_plus_1]
                processed_data_N = [f.result() for f in futures_N]
                collated_batch_N = self.collate_fn(processed_data_N)
                
                if collated_batch_N is not None:
                    yield collated_batch_N
                
                futures_N = futures_N_plus_1
            
            processed_data_last = [f.result() for f in futures_N]
            last_batch = self.collate_fn(processed_data_last)
            if last_batch is not None:
                yield last_batch

    def __len__(self):
        return len(self.batch_sampler)