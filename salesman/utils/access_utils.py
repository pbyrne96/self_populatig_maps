from typing import Any, Dict
from types import FunctionType
import numpy_utils

class PopulateSlots(type):
    def __init__(self, *args) -> None:
        self.all_function_attrs: Dict[str, callable] = {}

    def check_and_set(self, file_access, m) -> None:
        if(isinstance(file_access.__dict__[m], FunctionType)):
            self.all_function_attrs[m] = file_access.__dict__[m]
            setattr(self, m , file_access.__dict__[m])

    def __call__(self) -> Any:
        file_access = numpy_utils
        strip_attrs = lambda i :  not(i.startswith('__'))
        methods = list(i for i in file_access.__dir__() if strip_attrs(i))
        for _,m in enumerate(methods):
            self.check_and_set(file_access, m)

        fn_names = list(self.all_function_attrs.keys())
        setattr(self, 'fn_names', fn_names)

        return self

class CreateAccessPoint(metaclass=PopulateSlots):
    ...


def access_point(fn_name: str):
    access_cls = CreateAccessPoint()
    assert fn_name in access_cls.fn_names
    return access_cls.__dict__[fn_name]
