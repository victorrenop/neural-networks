import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

class DataViz:
    @staticmethod
    def plot_loss(loss: ndarray, val_loss: ndarray, epochs: int):
        plt.figure()
        plt.plot(epochs, loss, label='Training loss')
        plt.plot(epochs, val_loss, label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_learning_rate(learning_rate: ndarray, epochs: int):
        plt.figure()
        plt.plot(epochs, learning_rate, label='Learning Rate')
        plt.title('Learning Over Time')
        plt.legend()
        plt.show()
