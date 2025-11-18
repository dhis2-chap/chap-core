from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from chap_core.assessment.flat_representations import (
    DIM_REGISTRY,
    DataDimension,
    FlatForecasts,
    FlatObserved,
)
from .base import MetricBase, MetricSpec 

class componentBasedMetric(MetricBase):
    """
    Base class for metrics that are computed based on components.
    Subclass this and implement the compute_component_metric-method to create a new metric.
    Define the spec attribute to specify what the metric outputs.
    """

    component_column: str = "component"

    def compute(
            self, 
            observations: pd.DataFrame, 
            forecasts: pd.DataFrame
            ) -> pd.DataFrame:
         
        components = self.compute_single(observations, forecasts)
        if self.component_column not in components.columns:
            raise ValueError(
                f"{self.__class__.__name__} produced wrong columns.\n"
                f"Missing component column: {self.component_column}\n"
            )
        
        out = self.aggregate_compoents(components)
        return out
    
    def compute_single(
            self, 
            observations: pd.DataFrame, 
            forecasts: pd.DataFrame
            ) -> pd.DataFrame:
        raise NotImplementedError
    #abstract base class method


    def aggregate_compoents(
            self, 
            components: pd.DataFrame
            ) -> pd.DataFrame:
        
        group_columns = [d.value for d in self.spec.output_dimensions]

        if not group_columns: 
            metric_df = (
                components[self.component_column]
                .agg("mean")
                .to_frame(name="metric")
                .reset_index(drop=True)
            )
        else:
            metric_df = (
                components
                .groupby(group_columns, as_index=False)[self.component_column]
                .mean()
                .rename(columns={self.component_column: "metric"})
            )
        return metric_df