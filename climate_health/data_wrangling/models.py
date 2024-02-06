from typing import TypeVar, Generic

from omnipy import JsonListOfListsOfScalarsModel, Dataset, JsonListOfDictsOfScalarsModel, Model
from omnipy.modules.json.models import JsonListM, JsonDictM, JsonCustomListModel
from pydantic import BaseModel

#
# class TableWithColNamesInFirstRowModel(JsonListOfListsOfScalarsModel):
#     @classmethod
#     def _parse_data(cls, data: JsonListOfListsOfScalarsModel):
#         if len(data) > 0:
#             if len(data[0]) > 0:
#                 for item in data[0]:
#                     assert isinstance(item.contents, str)
#             else:
#                 return []
#         return data

# These are prototype implementations that will be moved to omnipy and improved


class TableWithColNamesModel(Model[JsonListOfListsOfScalarsModel | JsonListOfDictsOfScalarsModel]):
    @classmethod
    def _parse_data(cls, data: 'JsonListOfListsOfScalarsModel | JsonListOfDictsOfScalarsModel | TableWithColNamesModel'):
        if len(data) > 0:
            if isinstance(data[0], JsonListM):
                first_row_as_colnames = JsonCustomListModel[str](data[0])
                return [{col_name: row[i] if i < len(row) else None for i, col_name in enumerate(first_row_as_colnames)}
                        for row in data[1:]]
            else:
                assert isinstance(data[0], JsonDictM)
                return data

        return data

    @property
    def col_names(self) -> tuple[str]:
        col_names = {}
        for row in self:
            col_names.update(dict.fromkeys(row.keys()))
        return tuple(col_names.keys())

#
# class TableWithColNamesInFirstRowDataset(Dataset[TableWithColNamesInFirstRowModel]):
#     ...


class TableWithColNamesDataset(Dataset[TableWithColNamesModel]):

    @property
    def col_names(self) -> tuple[str]:
        col_names = {}
        for data_file in self.values():
            col_names.update(dict.fromkeys(data_file.col_names))
        return tuple(col_names.keys())


PydanticModelT = TypeVar('PydanticModelT', bound=BaseModel)


class TableOfPydanticRecordsModel(Model[list[PydanticModelT]], Generic[PydanticModelT]):
    ...


class TableOfPydanticRecordsDataset(Dataset[TableOfPydanticRecordsModel[PydanticModelT]],
                                    Generic[PydanticModelT]):
    ...
