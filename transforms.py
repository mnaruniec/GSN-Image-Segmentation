import torch


def assert_square_batch(x):
    assert 3 <= len(x.shape) <= 4
    assert x.size(-1) == x.size(-2)


class ReversibleTransform:
    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        raise NotImplementedError

    def reverse(self, x):
        raise NotImplementedError


class Identity(ReversibleTransform):
    def apply(self, x):
        return x

    def reverse(self, x):
        return x


class HorizontalFlip(ReversibleTransform):
    def apply(self, x):
        assert_square_batch(x)
        return x.flip(-1)

    def reverse(self, x):
        return self.apply(x)


class Rotation(ReversibleTransform):
    dims = (-2, -1)

    def __init__(self, times):
        self.times = times % 4

    def apply(self, x):
        assert_square_batch(x)
        return torch.rot90(x, k=self.times, dims=self.dims)

    def reverse(self, x):
        assert_square_batch(x)
        return torch.rot90(x, k=(4 - self.times), dims=self.dims)


class Rotation90(Rotation):
    def __init__(self):
        super().__init__(1)


class Rotation270(Rotation):
    def __init__(self):
        super().__init__(3)
