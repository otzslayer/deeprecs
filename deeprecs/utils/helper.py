import inspect
from types import ModuleType
from typing import Tuple


def get_classes_from_module(module: ModuleType) -> Tuple[type, ...]:
    """
    _summary_

    Parameters
    ----------
    module : ModuleType
        _description_

    Returns
    -------
    Tuple[type, ...]
        _description_

    References
    ----------
    https://stackoverflow.com/questions/44325153/python-type-hint-for-any-class
    """
    classes = inspect.getmembers(module, inspect.isclass)
    classes = tuple(class_[1] for class_ in classes)
    return classes
