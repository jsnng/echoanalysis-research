import torch

__all__ = ['ToTensor']


class ToTensor:
    """
    re-implementation of torchvision.transforms.ToTensor() for ndarrays with 4 dimensions.

    converts:
        numpy.ndarray with 4 dimensions of shape [fc, h, w, c] to
        torch.floatTensor of shape (fc x c x h x w) in the range [0, 1].
    """
    def __call__(self, ndarray):
        """
        
        Args:
            ndarray (numpy.ndarray): float ndarray to be converted to tensor.

        Returns:
            torch.Tensor with shape [fc, c, h, w]

        Raises:
            ValueError: if ndarray.ndim != 4

        """
        if not ndarray.ndim == 4:
            raise ValueError(f'unexpected ndarray.ndim = {ndarray.ndim}. require ndarray ndim to be 4')
        ndarray = ndarray.transpose((0, 3, 1, 2))
        return torch.from_numpy(ndarray)
