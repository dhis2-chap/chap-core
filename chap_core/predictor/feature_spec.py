from pydantic import BaseModel


class Feature(BaseModel):
    id: str
    name: str
    description: str
    optional: bool = False

    def __hash__(self):
        return hash(self.id)


rainfall = Feature(id="rainfall", name="Rainfall", description="The amount of rainfall in mm", optional=True)

mean_temperature = Feature(
    id="mean_temperature",
    name="Mean Temperature",
    description="The average temperature in degrees Celsius",
    optional=True,
)

population = Feature(id="population", name="Population", description="The population of the area")

all_features = [var for var in locals().values() if isinstance(var, Feature)]
feature_dict = {var.id: var for var in all_features}
