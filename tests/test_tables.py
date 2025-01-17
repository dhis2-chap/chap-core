from chap_core.database.base_tables import DBModel


def test_dbmodel():
    class TestModel(DBModel):
        snake_case_name: str

    #config = TestModel.Config
    #assert config.allow_population_by_name is True, config
    data = TestModel(snake_case_name="test")
    json_data = data.model_dump(by_alias=True)
    assert json_data == {"snakeCaseName": "test"}
