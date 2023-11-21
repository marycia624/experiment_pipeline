import yaml
import config
import abc
import pandas as pd
import numpy as np
from yaml.loader import SafeLoader
from os import listdir


def _load_yaml_preset(preset="default"):
    preset_path = config.PATH_METRIC_CONFIGS + preset
    metrics_to_load = listdir(preset_path)
    metrics = []
    for metric in metrics_to_load:
        with open(preset_path + "/" + metric) as f:
            metrics.append(yaml.load(f, Loader=SafeLoader))
    return metrics


class Metric:
    def __init__(self, metric_config: dict):
        self._config = metric_config

    @property
    def name(self) -> str:
        return self._config.get("name", "default_value")

    @property
    def type(self) -> str:
        return self._config.get("type", "default_metric_type")

    @property
    def level(self) -> str:
        return self._config.get("level", "default_unit_level")

    @property
    def estimator(self) -> str:
        return self._config.get("estimator", "default_estimator")

    @property
    def numerator(self) -> dict:
        return self._config.get("numerator", {"aggregation_field": "default_value"})

    @property
    def denominator(self) -> dict:
        return self._config.get("denominator", {"aggregation_field": "default_value"})

    @property
    def numerator_aggregation_field(self) -> str:
        return self.numerator.get("aggregation_field", "default_value")

    @property
    def denominator_aggregation_field(self) -> str:
        return self.denominator.get("aggregation_field", "default_value")

    @property
    def numerator_aggregation_function(self) -> callable:
        return self._map_aggregation_function(self.numerator.get("aggregation_function"))

    @property
    def denominator_aggregation_function(self) -> callable:
        return self._map_aggregation_function(self.denominator.get("aggregation_function"))

    @property
    def numerator_conditions(self) -> list:
        return self._config.get("numerator_conditions", [])

    @property
    def denominator_conditions(self) -> list:
        return self._config.get("denominator_conditions", [])

    @staticmethod
    def _map_aggregation_function(aggregation_function: str) -> callable:
        mappings = {
            "count_distinct": pd.Series.nunique,
            "sum": np.sum
        }
        if aggregation_function not in mappings:
            raise ValueError(f"{aggregation_function} not found in mappings")
        return mappings[aggregation_function]

    @staticmethod
    def apply_filter(data, conditions: list) -> pd.DataFrame:
        operations = {
            'equal': lambda df, field, value: df[df[field] == value],
            'not_equal': lambda df, field, value: df[df[field] != value],
        }
        for condition in conditions:
            # print(condition)
            # print(type(condition))
            operation = operations.get(condition.get('comparison_sign'))
            # print(operation)
            # print(type(operation))
            data = operation(data, condition.get('condition_field'), condition.get('comparison_value'))

        return data

class CalculateMetric:
    def __init__(self, metric: Metric):
        self.metric = metric

    def __call__(self, df):
        return df.groupby([config.VARIANT_COL, self.metric.level]).apply(
            lambda df: pd.Series({
                "num": self.metric.numerator_aggregation_function(self.metric.apply_filter(df, self.metric.numerator_conditions)[self.metric.numerator_aggregation_field]),
                "den": self.metric.denominator_aggregation_function(self.metric.apply_filter(df, self.metric.denominator_conditions)[self.metric.denominator_aggregation_field]),
                "n": pd.Series.nunique(df[self.metric.level])
            })
        ).reset_index()


