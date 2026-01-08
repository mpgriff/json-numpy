from __future__ import annotations

__version__ = "2.1.0"
__all__ = ["default", "object_hook", "dumps", "loads", "dump", "load", "patch"]

import json
from base64 import b64decode, b64encode
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from numpy import frombuffer, generic, ndarray, array, allclose
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from torch import Tensor as torch_ndarray, from_numpy, allclose as torch_allclose

if TYPE_CHECKING:  # pragma: no cover
    from _typeshed import SupportsRead


def default(
    o: Any, *, binary_threshold=100, fallback_default: Callable[[Any], dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Encodes numpy objects to a JSON-serializable dictionary.

    Args:
        o (object): The object to encode.
        fallback_default (Callable[[Any], dict[str, Any]] | None): A fallback encoder function to handle objects that are not numpy objects.

    Returns:
        dict[str, Any]: The JSON-serializable dictionary representation of the numpy object, or the result of the fallback encoder if present and if the object is not a numpy object.

    Raises:
        TypeError: If the object is not JSON serializable.
    """
    array_package = '__numpy__'
    if isinstance(o, torch_ndarray):
        o = o.cpu().numpy()
        array_package = '__torch__'
    if isinstance(o, (ndarray, generic)):
        if o.size > binary_threshold:
            data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
            values = b64encode(data).decode()
        else:
            values = ' '.join(o.__repr__().replace('\n', '').split())
            values = values.replace(', dtype=float32', '').replace(', dtype=float64', '').replace(', dtype=int64', '').replace(', dtype=int32', '')
        return {
            array_package: values,
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }

    if fallback_default is not None:
        return fallback_default(o)

    msg = f"Object of type {o.__class__.__name__} is not JSON serializable"
    raise TypeError(msg)


def object_hook(dct: dict) -> dict | ndarray | generic:
    """Custom object hook function for decoding JSON objects into numpy arrays.

    Args:
        dct (dict): The dictionary to decode.

    Returns:
        dict | np.ndarray | np.generic: The decoded numpy object or the original dictionary.
    """
    if "__numpy__" in dct:
        if dct['__numpy__'].startswith('array'):
            np_obj = eval(dct['__numpy__']).astype(dct['dtype'])
        else:
            np_obj = frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(shape) if (shape := dct["shape"]) else np_obj[0]
    elif "__torch__" in dct:
        if dct['__torch__'].startswith('array'):
            np_obj = eval(dct['__torch__']).astype(dct['dtype'])
        else:
            np_obj = frombuffer(b64decode(dct["__torch__"]), descr_to_dtype(dct["dtype"]))
        np_obj = np_obj.reshape(shape) if (shape := dct["shape"]) else np_obj#[0]
        return from_numpy(np_obj.copy())
    return dct


_default = default
_hook = object_hook
_dumps = json.dumps
_loads = json.loads
_dump = json.dump
_load = json.load


def _patch_encoder(
    *args: Any,
    default: Callable[[Any], Any] | None = None,
    user_cls: type[json.JSONEncoder] | None = None,
    **kwargs: Any,
) -> json.JSONEncoder:
    """Ensures cooperation with the provided `default` and/or `cls` by manipulating the JSONEncoder."""
    if user_cls is None:
        user_cls = json.JSONEncoder
    elif default is None:
        encoder = user_cls(*args, **kwargs)
        encoder.default = partial(_default, fallback_default=encoder.default)  # type: ignore[method-assign]
        return encoder
    return user_cls(
        *args, default=partial(_default, fallback_default=default), **kwargs
    )


def dumps(*args: Any, cls: type[json.JSONEncoder] | None = None, **kwargs: Any) -> str:
    kwargs["user_cls"] = cls
    return _dumps(*args, cls=_patch_encoder, **kwargs)  # type: ignore[arg-type]


def loads(
    *args: Any, object_hook: Callable[[dict], Any] | None = None, **kwargs: Any
) -> Any:
    return _loads(
        *args,
        object_hook=_hook
        if object_hook is None
        else lambda dct: _hook(object_hook(dct)),
        **kwargs,
    )


def dump(*args: Any, cls: type[json.JSONEncoder] | None = None, **kwargs: Any) -> None:
    kwargs["user_cls"] = cls
    return _dump(*args, cls=_patch_encoder, **kwargs)  # type: ignore[arg-type]


def load(fp: SupportsRead[str | bytes], **kwargs: Any) -> Any:
    return loads(fp.read(), **kwargs)


def patch() -> None:
    """Monkey patch json module to support encoding/decoding NumPy arrays/scalars."""
    json.dumps = dumps
    json.loads = loads
    json.dump = dump
    json.load = load

patch()

from dataclasses import dataclass, replace, asdict as dataclass2dict
from typing import List, _GenericAlias

def jsonable_dataclass(cls):
    """Decorator to add utilities to a class"""
    def update(self, **kwargs):
        return replace(self, **kwargs)

    def save(self, fname):
        if self.extension is not None and '.' not in fname:
            fname = fname + '.' + self.extension
        if self.extension is not None:
            assert fname.endswith('.' + self.extension), 'File extension does not match class extension'

        
        other = self.copy()

        for key,val in other.__dict__.items():
            if (hasattr(val, 'description') and val.description is not None) and (hasattr(val, 'extension') and val.extension is not None):
                if val.description.endswith(val.extension):
                    exec(f'other.{key} = val.description')
        
        res = json.dumps(dataclass2dict(other), indent=4)
        with open(fname, 'wt') as f:
            f.write(res)
    
    @classmethod
    def load(cls, fname, load_subclasses=False):
        try:
            with open(fname, 'rb') as f:
                res = json.loads(f.read())
            res['description'] = fname
            res = cls._reinstantiate_subclasses(cls, res, load_subclasses=load_subclasses)
            return res
        except UnicodeDecodeError:
            import warnings
            warnings.warn('Could not load as json, trying to load as binary')
            old_obj = cls._load(fname)
            container = {}

            for key in cls.__annotations__.keys():
                if key in old_obj.__dict__.keys():
                    container[key] = old_obj.__dict__[key]
            return cls(**container)
    
    def __hash__(self):
        descript = self.description
        self.description = 'hash'
        res = hash(self.__repr__())
        self.description = descript
        return res
    
    def __eq__(self, other):
        is_equal = hash(self) == hash(other)
        if is_equal:
            for x in self.__annotations__.keys():
                cvar = vars(self)[x]
                ovar = vars(other)[x]

                if x == 'description':
                    continue
                elif isinstance(cvar, ndarray):
                    var_equal = allclose(cvar, ovar)
                elif isinstance(cvar, torch_ndarray):
                    var_equal = bool(torch_allclose(cvar, ovar))
                else:
                    var_equal = (cvar == ovar)

                is_equal = is_equal and var_equal

        return is_equal
    
        
    
    
    def reinstantiate_subclasses(cls, d, load_subclasses=False):
        """recursive function to get attributes back into their right classes"""
        if cls.__base__ != (object, str) and hasattr(cls.__base__, '__annotations__'):
            class_dict = cls.__annotations__ | cls.__base__.__annotations__
        else:
            class_dict = cls.__annotations__

        if hasattr(d, 'keys'):
            # assert all(class_keys == list(d.keys())), 'Keys do not match'
            for key in d.keys():
                if class_dict[key] != type(d[key]) and type(d[key]) == dict:
                    d[key] = reinstantiate_subclasses(class_dict[key], d[key])
                elif isinstance(class_dict[key], _GenericAlias):
                    subclass = [x for x in class_dict[key].__args__]
                    if len(subclass) < len(d[key]):
                        subclass = subclass * len(d[key])
                    if hasattr(subclass[0], '__annotations__'):
                        d[key] = [reinstantiate_subclasses(const, x, load_subclasses=load_subclasses) for const, x in zip(subclass,d[key])]
                    else:
                        d[key] = [x for const, x in zip(subclass,d[key])]

                elif class_dict[key] == ndarray:
                    pass
                elif class_dict[key] != type(d[key]) and type(d[key]) == str and '.' in d[key] and load_subclasses:
                    d[key] = class_dict[key].load(d[key])
                elif class_dict[key] != type(d[key]) and load_subclasses:
                    if d[key] is not None:
                        d[key] = class_dict[key](**d[key])
                    
            return cls(**d)
        elif isinstance(d, str) and load_subclasses:
            return cls.load(d)
        else:
            return d    
    def copy(self):
        return replace(self)
    
    cls.description = cls.__name__
    if 'extension' not in cls.__dict__:
        cls.extension = None
    cls.update       = update
    cls.__hash__     = __hash__
    cls.__eq__       = __eq__
    cls.save         = save
    cls.load         = load
    cls._reinstantiate_subclasses = reinstantiate_subclasses
    cls.copy         = copy

    return dataclass(cls)

apsu_class = jsonable_dataclass# maintain backward compatibility