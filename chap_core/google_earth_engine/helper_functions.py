# import ee
#
# class Era5LandGoogleEarthEngineHelperFunctions:
#     # Get every dayli image that exisist within a periode, and reduce it to a periodeReducer value
#     def get_image_for_period(self, p: Periode, band: Band, collection: ee.ImageCollection) -> ee.Image:
#         p = ee.Dictionary(p)
#         start = ee.Date(p.get("start_date"))
#         end = ee.Date(
#             p.get("end_date")
#         )  # .advance(-1, "day") #remove one day, since the end date is inclusive on current format?
#
#         # Get only images from start to end, for one bands
#         filtered: ee.ImageCollection = collection.filterDate(start, end).select(band.name)
#
#         # Aggregate the imageCollection to one image, based on the periodeReducer
#         return (
#             getattr(filtered, band.periode_reducer)()
#             .set("system:period", p.get("period"))
#             .set("system:time_start", start.millis())
#             .set("system:time_end", end.millis())
#             .set("system:indicator", band.indicator)
#         )
#
#     def create_ee_dict(self, p: TimePeriod):
#         return ee.Dictionary(
#             {
#                 "period": p.id,
#                 "start_date": p.start_timestamp.date,
#                 "end_date": p.end_timestamp.date,
#             }
#         )
#
#     def creat_ee_feature(self, feature, image, eeReducerType):
#         return ee.Feature(
#             None,  # exlude geometry
#             {
#                 "ou": feature.id(),
#                 "period": image.get("system:period"),
#                 "value": feature.get(eeReducerType),
#                 "indicator": image.get("system:indicator"),
#             },
#         )
#
#     def convert_value_by_band_converter(self, data, bands: List[Band]):
#         return [
#             {
#                 **f["properties"],
#                 # Using the right converter on the value, based on the whats defined as band-converter
#                 **{
#                     "value": next(b.converter for b in bands if f["properties"]["indicator"] == b.indicator)(
#                         f["properties"]["value"]
#                     )
#                 },
#             }
#             for f in data
#         ]
#
#     def feature_collection_to_list(self, feature_collection: "ee.FeatureCollection"):
#         size = feature_collection.size().getInfo()
#         result: List = []
#         take = 5_000
#
#         # Keeps every f.properties, and replace the band values with the converted values
#         for i in range(0, size, take):
#             result = result + (feature_collection.toList(take, i).getInfo())
#             logger.log(logging.INFO, f" Fetched {i+take} of {size}")
#
#         return result
#
#     @staticmethod
#     def parse_gee_properties(property_dicts: list[dict]) -> DataSet:
#         df = pd.DataFrame(property_dicts)
#         location_groups = df.groupby("ou")
#         full_dict = {}
#         for location, group in location_groups:
#             data_dict, pr = Era5LandGoogleEarthEngineHelperFunctions._get_data_dict(group)
#             full_dict[location] = SimpleClimateData(pr, **data_dict)
#         return DataSet(full_dict)
#
#     @staticmethod
#     def _get_data_dict(group):
#         data_dict = {band: group[group["indicator"] == band] for band in group["indicator"].unique()}
#         pr = None
#         for band, band_group in group.groupby("indicator"):
#             data_dict[band] = band_group["value"]
#             pr = PeriodRange.from_ids(band_group["period"])
#         return data_dict, pr