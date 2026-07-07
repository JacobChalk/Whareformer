import numpy as np 

class FeatureBuffer:
    def __init__(self, chunk_size=1000, max_memory=None):
        self.chunk_size  = chunk_size
        self.max_memory  = max_memory
        self._feat_dim   = None

        self._flushed_chunks = []
        self._chunk_buf  = None
        self._chunk_ptr  = 0

        self._cache       = None
        self._cache_valid = False

    def update(self, feature):
        if self._chunk_buf is None:
            self._feat_dim = len(feature)
            self._chunk_buf = np.empty((self.chunk_size, self._feat_dim), dtype=np.float32)

        self._chunk_buf[self._chunk_ptr] = feature
        self._chunk_ptr += 1
        
        self._cache_valid = False

        if self._chunk_ptr == self.chunk_size:
            self._flushed_chunks.append(self._chunk_buf)
            self._chunk_buf = np.empty((self.chunk_size, self._feat_dim), dtype=np.float32)
            self._chunk_ptr = 0
            
            if self.max_memory is not None:
                max_chunks_needed = (self.max_memory + self.chunk_size - 1) // self.chunk_size
                if len(self._flushed_chunks) > max_chunks_needed:
                    self._flushed_chunks = self._flushed_chunks[-max_chunks_needed:]

    def get_array(self):
        if self._cache_valid:
            return self._cache

        if self._chunk_buf is None:
            return np.empty((0, 0), dtype=np.float32)

        n_active = self._chunk_ptr
        active_slice = self._chunk_buf[:n_active]

        if self.max_memory is None:
            result = np.vstack(self._flushed_chunks + [active_slice]) if self._flushed_chunks else active_slice
            
        elif n_active >= self.max_memory:
            result = self._chunk_buf[n_active - self.max_memory : n_active]
            
        else:
            needed = self.max_memory - n_active
            if not self._flushed_chunks:
                result = active_slice
            elif needed <= self.chunk_size:
                result = np.vstack((self._flushed_chunks[-1][-needed:], active_slice))
            else:
                parts = [active_slice] if n_active > 0 else []
                for chunk in reversed(self._flushed_chunks):
                    if needed <= 0: break
                    take = min(self.chunk_size, needed)
                    parts.append(chunk[-take:])
                    needed -= take
                parts.reverse()
                result = np.vstack(parts)

        self._cache = result
        self._cache_valid = True
        return result