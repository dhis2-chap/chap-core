from omnipy import JsonListOfListsOfScalarsModel, Dataset, JsonListOfDictsOfScalarsModel, Model
from omnipy.modules.json.models import JsonListM, JsonDictM


# These are prototype implementations that will be moved to omnipy and improved

class TableWithColNamesInFirstRowModel(JsonListOfListsOfScalarsModel):
    @classmethod
    def _parse_data(cls, data: JsonListOfListsOfScalarsModel):
        if len(data) > 0:
            if len(data[0]) > 0:
                for item in data[0]:
                    assert isinstance(item.contents, str)
            else:
                return []
        return data


class TableWithColNamesModel(Model[JsonListOfDictsOfScalarsModel | TableWithColNamesInFirstRowModel]):
    @classmethod
    def _parse_data(cls, data: JsonListOfDictsOfScalarsModel | TableWithColNamesInFirstRowModel):
        if len(data) > 0:
            if isinstance(data[0], JsonListM):
                return [{col_name.contents: row[i] if i < len(row) else None for i, col_name in enumerate(data[0])}
                        for row in data[1:]]
            else:
                assert isinstance(data[0], JsonDictM)
                return data

        return data


class TableWithColNamesInFirstRowDataset(Dataset[TableWithColNamesInFirstRowModel]):
    ...


class TableWithColNamesDataset(Dataset[TableWithColNamesModel]):
    ...
