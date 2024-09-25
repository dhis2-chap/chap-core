from sqlmodel import Field, SQLModel, create_engine, Session, Index

from chap_core.database.local_db_cache import LocalDbCache


# Define the TestRainfall model with an additional index on the value column
class TestRainfall(SQLModel, table=True):
    period_id: str = Field(primary_key=True)
    region_id: str = Field(primary_key=True)
    value: float

    __table_args__ = (
        Index("idx_period_id", "period_id"),
        Index("idx_region_id", "region_id"),
    )


# Define the database URL
DATABASE_URL = "postgresql://test_user:chap_core@localhost:5432/test_db"

# Create the database engine
engine = create_engine(DATABASE_URL, echo=True)

# Create the testrainfall table in the database with the indexes
SQLModel.metadata.create_all(engine)

# Mock external data retrieval function


@LocalDbCache.decorate(TestRainfall)
def retrieve_rainfall_data_from_external_source(period_id, region_id):
    # Simulate external data retrieval
    return 150.0


# Example usage
period_id = "202101"
region_id = "RegionA"
rainfall_data = retrieve_rainfall_data_from_external_source(
    period_id, region_id, session=Session(engine)
)
print(rainfall_data)
