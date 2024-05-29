from typing import Protocol

IndexLike = list | tuple | int | slice | None

class ArrayTree(Protocol[T]):

    def ndim(self) -> int:
        ...

    def shape(self) -> tuple[int]:
        ...

    def __getitem__(self, key: IndexLike) -> 'Self[T]':
        ...
