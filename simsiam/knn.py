# adapted from
# https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
import numpy as np
import faiss
import torch


class FaissKNeighbors:
    def __init__(self, k=5, temperature=None, run_on_gpu=True):
        self.index = None
        self.y = None
        self.k = k
        self.temperature = temperature
        self.run_on_gpu = run_on_gpu

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])

        if self.run_on_gpu:
            if torch.cuda.device_count() > 1:
                co = faiss.GpuMultipleClonerOptions()
                co.shards = True
                self.index = faiss.index_cpu_to_all_gpus(self.index, co)
            else:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]

        if self.temperature is not None:
            weights = np.exp((distances / self.temperature))
        else:
            weights = np.ones_like(distances, dtype=np.float32)

        predictions = np.array([np.argmax(np.bincount(x, weights=w)) for x, w in zip(votes, weights)])
        return predictions
