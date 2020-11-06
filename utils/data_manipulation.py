import numpy as np
from numpy import ndarray

class Data:
    @staticmethod
    def flatten_data(data: ndarray) -> ndarray:
        if isinstance(data, ndarray):
            return data.flatten()
        return np.array(data).flatten()
    
    @staticmethod
    def normalize_data(data: ndarray, divide_factor: float) -> ndarray:
        if not isinstance(data, ndarray):
            data = np.array(data)
        return data / divide_factor
