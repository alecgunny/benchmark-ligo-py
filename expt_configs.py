import attr
from itertools import product


@attr.s(auto_attribs=True)
class Expt:
    kernel_stride: float
    instances: int
    gpus: int

    def __str__(self):
        attrs = []
        for a in self.__attrs_attrs__:
            attrs.append("{}={}".format(a.name, self.__dict__[a.name]))
        return "-".join(attrs)


kernel_strides = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
instances = [1, 2, 4]
gpus = [1, 2, 4]

expts = []
for i, g, k in product(instances, gpus, kernel_strides):
    expts.append(Expt(k, i, g))
