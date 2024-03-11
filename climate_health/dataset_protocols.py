




class SpatioTemporalDataSet(Protocol, Generic[T]):
    dataclass = ...

    def get_data_for_locations(self, location: Iterable[spatial_index_type]) -> 'SpatioTemporalDataSet[T]':
        ...

    def get_data_for_location(self, location: spatial_index_type) -> T:
        ...

    def restrict_time_period(self, start_period: Period=None, end_period: Period=None) -> 'SpatioTemporalDataSet[T]':
        ...

    def start_time(self) -> Period:
        ...

    def end_time(self) -> Period:
        ...

    def locations(self) -> Iterable[spatial_index_type]:
        ...

    def data(self) -> Iterable[T]:
        ...

    def items(self) -> Iterable[Tuple[spatial_index_type, T]]:
        ...

    def to_tidy_dataframe(self) -> pd.DataFrame:
        ...

    @classmethod
    def from_tidy_dataframe(cls, df: pd.DataFrame) -> 'SpatioTemporalDataSet[T]':
        ...