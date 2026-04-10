from collections.abc import MutableMapping
from typing import Iterator, Any


class DictWrapper(MutableMapping):
    """
    내부에 dict를 감싸고, 바깥에서는 dict처럼 동작하게 하는 래퍼 클래스.
    필요하면 여기에 검증, 로깅, key 변환 등의 기능을 추가할 수 있다.
    """

    def __init__(self, initial_data=None, **kwargs):
        self._data = dict(initial_data or {})
        self._data.update(kwargs)

    # --- MutableMapping 필수 구현 ---
    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # --- dict처럼 보이게 하는 편의 메서드 ---
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def copy(self):
        return self.__class__(self._data.copy())

    def get(self, key: Any, default=None) -> Any:
        return self._data.get(key, default)

    def setdefault(self, key: Any, default=None) -> Any:
        return self._data.setdefault(key, default)

    def pop(self, key: Any, default=...):
        if default is ...:
            return self._data.pop(key)
        return self._data.pop(key, default)

    def popitem(self):
        return self._data.popitem()

    def clear(self) -> None:
        self._data.clear()

    def update(self, other=None, **kwargs) -> None:
        if other is not None:
            if hasattr(other, "keys"):
                for k in other:
                    self._data[k] = other[k]
            else:
                for k, v in other:
                    self._data[k] = v
        for k, v in kwargs.items():
            self._data[k] = v

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    # 원본 dict가 필요할 때
    def unwrap(self) -> dict:
        return self._data
