# class FeatureTypes(SQLModel, table=True):
#     name: str = Field(str, primary_key=True)
#     display_name: str
#     description: str
#
#
# class Model(SQLModel, table=True):
#     id: Optional[int] = Field(primary_key=True, default=None)
#     name: str
#     estimator_id: str
#     features: List[FeatureTypes] = Relationship(back_populates="model")
#
#
# class FeatureSources(SQLModel, table=True):
#     id: Optional[int] = Field(primary_key=True, default=None)
#     name: str
#     feature_type: str
#     url: str
#     #metadata: Optional[str] = Field(default=None)
#
#
# class LocalDataSource(SQLModel, table=True):
#     dhis2_id: str = Field(primary_key=True)
#     feature_types: List[FeatureTypes] = Relationship(back_populates="source")
