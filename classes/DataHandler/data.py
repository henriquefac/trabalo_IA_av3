import numpy as np

class DataHandler():
    def __init__(self, path) -> None:
        self.data = np.loadtxt(path, delimiter=',')

        self.lines = self.data.shape[0]
        self.colls = self.data.shape[1]
    def MonteCarlo(self) -> tuple[np.ndarray]:
        """
        Divide os ddados em uma proporção 80/20%
        """
        index = np.random.permutation(self.data.shape[0])
        divide = int(len(index)*0.8)

        return self.data[index[:divide],:],self.data[index[divide:], :]
    def nomrData(self, entry:np.ndarray):
        return (entry - entry.mean(axis=1, keepdims=True)) / entry.std(axis=1, keepdims=True)
    @staticmethod
    def SepXY(dataMatrix):
        dataMatrix = dataMatrix.data if isinstance(dataMatrix, DataHandler) else dataMatrix
        return dataMatrix[:, :dataMatrix.shape[1]-1],dataMatrix[:, dataMatrix.shape[1]-1]