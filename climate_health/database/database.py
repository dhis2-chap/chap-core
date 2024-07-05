from sqlmodel import Field, SQLModel, create_engine, Session, select, Index

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
DATABASE_URL = "postgresql://test_user:climate_health@localhost:5432/test_db"

# Create the database engine
engine = create_engine(DATABASE_URL, echo=True)

# Create the testrainfall table in the database with the indexes
SQLModel.metadata.create_all(engine)

# Mock external data retrieval function
def retrieve_rainfall_data_from_external_source(period_id, region_id):
    # Simulate external data retrieval
    return 150.0

# Function to check for an entry in the database
def check_entry_in_db(session, period_id, region_id):
    statement = select(TestRainfall).where(TestRainfall.period_id == period_id, TestRainfall.region_id == region_id)
    result = session.exec(statement).first()
    return result

# Function to insert data into the database
def insert_rainfall_data(session, period_id, region_id, value):
    new_rainfall = TestRainfall(period_id=period_id, region_id=region_id, value=value)
    session.add(new_rainfall)
    session.commit()
    return new_rainfall

# Function to retrieve or fetch and store data
def get_rainfall_data(period_id, region_id):
    with Session(engine) as session:
        # Check if the entry exists in the database
        existing_entry = check_entry_in_db(session, period_id, region_id)
        if existing_entry:
            return existing_entry
        else:
            # Retrieve data from external source
            external_value = retrieve_rainfall_data_from_external_source(period_id, region_id)
            # Store the new data in the database
            new_entry = insert_rainfall_data(session, period_id, region_id, external_value)
            return new_entry

# Example usage
period_id = "202101"
region_id = "RegionA"
rainfall_data = get_rainfall_data(period_id, region_id)
print(rainfall_data)
