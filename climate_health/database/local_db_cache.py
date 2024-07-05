from select import select
from typing import Type

from sqlmodel import Session, SQLModel


class LocalDbCache:
    def __init__(self, session: Session, model: Type[SQLModel]):
        self._session = session
        self._model = model

    def __contains__(self, item: tuple[str, str]) -> bool:
        period_id, region_id = item
        statement = select(self._model).where(self._model.period_id == period_id, self._model.region_id == region_id)
        result = self._session.exec(statement).first()
        return result

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

    def decorate(self, func):
        def wrapper(period_id, region_id):
            if (period_id, region_id) in self:
                return self[period_id, region_id]
            else:
                value = func(period_id, region_id)
                self[period_id, region_id] = value
                return value

        return wrapper


