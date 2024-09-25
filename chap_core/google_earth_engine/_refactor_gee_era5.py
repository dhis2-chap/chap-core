class GoogleEarthEngine(BaseModel):
    def initializeClient(self):
        # read environment variables
        account = os.environ.get("GOOGLE_SERVICE_ACCOUNT_EMAIL")
        private_key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY")

        if not account:
            logger.warn(
                "GOOGLE_SERVICE_ACCOUNT_EMAIL is not set, you need to set it in the environment variables to use Google Earth Engine"
            )
        if not private_key:
            logger.warn(
                "GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY is not set, you need to set it in the environment variables to use Google Earth Engine"
            )

        if not account or not private_key:
            return

        try:
            credentials = ee.ServiceAccountCredentials(account, key_data=private_key)
            ee.Initialize(credentials)
            logger.info("Google Earth Engine initialized, with account: " + account)
        except ValueError as e:
            logger.error("\nERROR:\n", e, "\n")

    # Keeps every f.properties, and replace the band values with the converted values
    def dataParser(self, data, bands: List[Band]):
        parsed_data = [
            {
                **f["properties"],
                **{
                    "value": next(
                        b.converter for b in bands if f["properties"]["band"] == b.name
                    )(f["properties"]["value"])
                },
            }
            for f in data
        ]
        return parsed_data

    def fetch_historical_era5_from_gee(
        self, zip_file_path: str, periodes: Iterable[Periode]
    ) -> SpatioTemporalDict[ClimateData]:
        features = self.get_feature_from_zip(zip_file_path)

        # Fetch data for both temperature and precipitation
        return self.fetch_data_climate_indicator(features, periodes, bands)

    def get_feature_from_zip(self, zip_file_path: str):
        ziparchive = zipfile.ZipFile(zip_file_path)
        features = json.load(ziparchive.open("orgUnits.geojson"))["features"]
        return features

    def fetch_data_climate_indicator(
        self, features, periodes: Iterable[TimePeriod], bands: List[Band]
    ) -> SpatioTemporalDict[ClimateData]:
        eeReducerType = "mean"

        collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").select(
            [band.name for band in bands]
        )
        featureCollection: ee.FeatureCollection = ee.FeatureCollection(features)

        # Creates a ee.List for every periode, containing id (periodeId), start_date and end_date for each period
        periodeList = ee.List(
            [
                ee.Dictionary(
                    {
                        "period": p.id,
                        "start_date": p.start_timestamp.date,
                        "end_date": p.end_timestamp.date,
                    }
                )
                for p in periodes
            ]
        )

        eeScale = collection.first().select(0).projection().nominalScale()
        eeReducer = getattr(ee.Reducer, eeReducerType)()

        # Get every dayli image that exisist within a periode, and reduce it to a periodeReducer value
        def getPeriode(p, band: Band) -> ee.Image:
            p = ee.Dictionary(p)
            start = ee.Date(p.get("start_date"))
            end = ee.Date(p.get("end_date")).advance(
                -1, "day"
            )  # remove one day, since the end date is inclusive on current format?

            # Get only images from start to end, for one bands
            filtered: ee.ImageCollection = collection.filterDate(start, end).select(
                band.name
            )

            # Aggregate the imageCollection to one image, based on the periodeReducer
            return (
                getattr(filtered, band.periodeReducer)()
                .set("system:period", p.get("period"))
                .set("system:time_start", start.millis())
                .set("system:time_end", end.millis())
            )

        dailyCollection = ee.ImageCollection([])

        # Map the bands, then the periodeList for each band, and return the aggregated Image to the ImageCollection
        for b in bands:
            dailyCollection = dailyCollection.merge(
                ee.ImageCollection.fromImages(
                    periodeList.map(lambda period: getPeriode(period, b))
                ).filter(ee.Filter.listContains("system:band_names", b.name))
            )  # Remove empty images

        # Reduce the result, to contain only, orgUnitId, periodeId and the value
        reduced = dailyCollection.map(
            lambda image: image.reduceRegions(
                collection=featureCollection, reducer=eeReducer, scale=eeScale
            ).map(
                lambda feature: ee.Feature(
                    None,
                    {
                        "ou": feature.id(),
                        "period": image.get("system:period"),
                        "value": feature.get(eeReducerType),
                        "band": image.bandNames().get(0),
                    },
                )
            )
        ).flatten()

        valueCollection: ee.FeatureCollection = ee.FeatureCollection(reduced)

        size = valueCollection.size().getInfo()
        result: List = []
        take = 5_000

        for i in range(0, size, take):
            # Fetch 5000 images at once
            result = result + (valueCollection.toList(take, i).getInfo())
            logger.log(logging.INFO, f" Fetched {i + take} of {size}")

        return self.dataParser(result, bands)
