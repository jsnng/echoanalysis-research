import torch

__all__ = ['ToTensor']


class ToTensor:
    def __call__(self, ndarray):
        if not ndarray.ndim == 4:
            raise ValueError(f"unexpected ndarray.ndim = {ndarray.ndim} != 4")
        ndarray = ndarray.transpose((0, 3, 1, 2))
        return torch.from_numpy(ndarray)
