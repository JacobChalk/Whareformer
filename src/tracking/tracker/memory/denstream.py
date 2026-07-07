import numpy as np
import math

class FeatureDenStream:
    def __init__(self, creation_timestamp: float, max_radius: float = 0.25, 
                 mu: float = 10.0, decay_rate: float = 1e-3,
                 initial_capacity: int = 512):
        
        self.max_radius = max_radius
        self.mu = mu
        self.decay_rate = decay_rate

        if decay_rate > 0:
            self.prune_interval = math.ceil((1.0 / decay_rate) * math.log(mu / (mu - 1.0)))
        else:
            self.prune_interval = float("inf")

        self.capacity = initial_capacity
        self.active_count = 0
        self.feature_dim = None

        self.weights = np.zeros(self.capacity, dtype=np.float32)
        self.last_update_times = np.zeros(self.capacity, dtype=np.float32)
        self.creation_times = np.zeros(self.capacity, dtype=np.float32)
        self.is_p_cluster = np.zeros(self.capacity, dtype=bool)

        self.linear_sums = None
        self.squared_sums = None
        self.cached_centroids = None

        self.last_prune_time = creation_timestamp

    def _resize_array(self, arr: np.ndarray, new_cap: int) -> np.ndarray:
        shape = list(arr.shape)
        shape[0] = new_cap
        new_arr = np.zeros(shape, dtype=arr.dtype)
        new_arr[:self.active_count] = arr[:self.active_count]
        return new_arr

    def _expand_arrays(self):
        new_cap = self.capacity * 2

        self.weights = self._resize_array(self.weights, new_cap)
        self.last_update_times = self._resize_array(self.last_update_times, new_cap)
        self.creation_times = self._resize_array(self.creation_times, new_cap)
        self.is_p_cluster = self._resize_array(self.is_p_cluster, new_cap)

        if self.linear_sums is not None:
            self.linear_sums = self._resize_array(self.linear_sums, new_cap)
            self.cached_centroids = self._resize_array(self.cached_centroids, new_cap)
            
        if self.squared_sums is not None:
            self.squared_sums = self._resize_array(self.squared_sums, new_cap)

        self.capacity = new_cap

    def _spawn_new_cluster(self, point: np.ndarray, timestamp: float):
        if self.feature_dim is None:
            self.feature_dim = point.shape[0]
            self.linear_sums = np.zeros((self.capacity, self.feature_dim), dtype=np.float32)
            self.cached_centroids = np.zeros((self.capacity, self.feature_dim), dtype=np.float32)
            self.squared_sums = np.zeros((self.capacity, self.feature_dim), dtype=np.float32)

        if self.active_count >= self.capacity:
            self._expand_arrays()

        idx = self.active_count
        self.linear_sums[idx] = point
        
        self.cached_centroids[idx] = point
        self.squared_sums[idx] = point ** 2

        self.weights[idx] = 1.0
        self.last_update_times[idx] = timestamp
        self.creation_times[idx] = timestamp
        self.is_p_cluster[idx] = False

        self.active_count += 1

    def _try_merge(self, mask: np.ndarray, distances: np.ndarray, point: np.ndarray, timestamp: float) -> bool:
        if not np.any(mask):
            return False

        valid_distances = distances[mask]
        subset_argmin = int(np.argmin(valid_distances))
        best_idx = int(mask.nonzero()[0][subset_argmin])

        time_delta = max(0.0, timestamp - self.last_update_times[best_idx])
        decay_factor = np.exp2(-self.decay_rate * time_delta)

        temp_ls = (self.linear_sums[best_idx] * decay_factor) + point
        temp_w = (self.weights[best_idx] * decay_factor) + 1.0

        temp_ss = (self.squared_sums[best_idx] * decay_factor) + point ** 2
        new_centroid = temp_ls / temp_w
        new_variance = (temp_ss / temp_w) - new_centroid ** 2

        new_variance[new_variance < 0] = 0 
        new_radius = math.sqrt(np.sum(new_variance))

        if new_radius <= self.max_radius:
            self.linear_sums[best_idx] = temp_ls
            self.weights[best_idx] = temp_w
            self.last_update_times[best_idx] = timestamp
            self.cached_centroids[best_idx] = new_centroid
            self.squared_sums[best_idx] = temp_ss

            if not self.is_p_cluster[best_idx] and temp_w >= self.mu:
                self.is_p_cluster[best_idx] = True

            return True

        return False

    def _check_prune(self, timestamp: float):
        if (timestamp - self.last_prune_time) >= self.prune_interval:
            self._prune_clusters(timestamp)
            self.last_prune_time = timestamp

    def _prune_clusters(self, timestamp: float):
        if self.active_count == 0:
            return

        time_deltas = np.maximum(0.0, timestamp - self.last_update_times[:self.active_count])
        dfs = np.exp2(-self.decay_rate * time_deltas)

        self.weights[:self.active_count] *= dfs
        self.linear_sums[:self.active_count] *= dfs[:, None]
        self.squared_sums[:self.active_count] *= dfs[:, None]

        immune_mask = (time_deltas == 0.0)
        p_mask = self.is_p_cluster[:self.active_count]
        p_keep = p_mask & ((self.weights[:self.active_count] >= self.mu) | immune_mask)

        t_mask = ~p_mask
        denom = np.exp2(-self.decay_rate * self.prune_interval) - 1.0
        
        if denom == 0:
            t_keep = t_mask
        else:
            numer = np.exp2(-self.decay_rate * (timestamp - self.creation_times[:self.active_count] + self.prune_interval)) - 1.0
            thresholds = numer / denom
            t_keep = t_mask & ((self.weights[:self.active_count] >= thresholds) | immune_mask)

        keep_mask = p_keep | t_keep
        new_count = np.sum(keep_mask)

        if new_count < self.active_count:
            self.weights[:new_count] = self.weights[:self.active_count][keep_mask]
            self.creation_times[:new_count] = self.creation_times[:self.active_count][keep_mask]
            self.is_p_cluster[:new_count] = self.is_p_cluster[:self.active_count][keep_mask]
            self.linear_sums[:new_count] = self.linear_sums[:self.active_count][keep_mask]
            self.cached_centroids[:new_count] = self.cached_centroids[:self.active_count][keep_mask]
            self.last_update_times[:new_count] = self.last_update_times[:self.active_count][keep_mask]
            self.squared_sums[:new_count] = self.squared_sums[:self.active_count][keep_mask]

        self.last_update_times[:new_count] = timestamp
        self.active_count = new_count

    def update(self, point: np.ndarray, timestamp: float):
        if self.active_count == 0:
            self._spawn_new_cluster(point, timestamp)
            return

        active_centroids = self.cached_centroids[:self.active_count]
        diff = active_centroids - point
        distances = np.einsum('ij,ij->i', diff, diff)

        p_mask = self.is_p_cluster[:self.active_count]
        
        if not self._try_merge(p_mask, distances, point, timestamp):
            if not self._try_merge(~p_mask, distances, point, timestamp):
                self._spawn_new_cluster(point, timestamp)

        self._check_prune(timestamp)

    def get_clusters(self) -> np.ndarray:
        if self.active_count == 0:
            return np.empty((0, self.feature_dim if self.feature_dim else 256), dtype=np.float32)
        return self.cached_centroids[:self.active_count]

    def __repr__(self):
        num_p = np.sum(self.is_p_cluster[:self.active_count])
        num_t = self.active_count - num_p
        return f"<FeatureDenStream p_clusters={num_p}, t_clusters={num_t}>"