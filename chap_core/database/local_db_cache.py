from typing import Type

from sqlmodel import Session, SQLModel, select


class LocalDbCache:
    def __init__(self, session: Session, model: Type[SQLModel]):
        self._session = session
        self._model = model

    def __contains__(self, item: tuple[str, str]) -> bool:
        period_id, region_id = item
        statement = select(self._model).where(self._model.period_id == period_id, self._model.region_id == region_id)
        result = self._session.exec(statement).first()
        return bool(result)

    def __getitem__(self, item: tuple[str, str]) -> SQLModel:
        period_id, region_id = item
        statement = select(self._model).where(self._model.period_id == period_id, self._model.region_id == region_id)
        result = self._session.exec(statement).first()
        return result

    def __setitem__(self, key: tuple[str, str], value: float) -> None:
        period_id, region_id = key
        new_entry = self._model(period_id=period_id, region_id=region_id, value=value)
        self._session.add(new_entry)
        self._session.commit()

    @classmethod
    def decorate(cls, model: Type[SQLModel]):
        def decorator(func):
            def wrapper(period_id, region_id, *args, **kwargs):
                if "session" not in kwargs:
                    return func(period_id, region_id, *args, **kwargs)
                session = kwargs.pop("session")
                self = cls(session, model)
                if (period_id, region_id) in self:
                    return self[period_id, region_id]
                value = func(period_id, region_id, *args, **kwargs)
                self[period_id, region_id] = value
                return value

            return wrapper

        return decorator
