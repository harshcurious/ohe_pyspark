from __future__ import annotations

import re
from typing import cast

import pandas as pd


def one_hot_encode_dataframe(
    dataframe: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """One-hot encode selected DataFrame columns with stable column names.

    Naming rules:
    - Prefix each encoded column with a lightly singularized source column name.
    - Normalize category labels to lowercase snake_case.
    - Append ``_na`` only when the source column contains missing values.

    Examples:
    >>> dataframe = pd.DataFrame({"animals": ["cat", "dog", "frog", "cat"]})
    >>> one_hot_encode_dataframe(dataframe)
       animal_cat  animal_dog  animal_frog
    0        True       False        False
    1       False        True        False
    2       False       False         True
    3        True       False        False

    >>> dataframe = pd.DataFrame(
    ...     {"animals": ["cat", None], "colors": ["tabby", "black"]}
    ... )
    >>> one_hot_encode_dataframe(dataframe, columns=["animals"])
       animal_cat  animal_na colors
    0        True      False  tabby
    1       False       True  black
    """

    target_columns = list(dataframe.columns if columns is None else columns)
    missing_columns = [
        column for column in target_columns if column not in dataframe.columns
    ]
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Unknown columns for one-hot encoding: {missing}")

    encoded_parts: list[pd.DataFrame] = []
    for column_name in dataframe.columns:
        if column_name not in target_columns:
            preserved_column = cast(
                pd.DataFrame, dataframe.loc[:, [column_name]].copy()
            )
            encoded_parts.append(preserved_column)
            continue

        prefix = _normalize_category_name(column_name)
        series: pd.Series = dataframe.loc[:, column_name]
        category_names = sorted(
            {_normalize_category_name(value) for value in series if not pd.isna(value)}
        )

        encoded_values = {
            f"{prefix}_{category_name}": series.eq(value_name).astype(bool)
            for category_name, value_name in (
                (_normalize_category_name(value), value)
                for value in series.dropna().unique()
            )
        }

        ordered_encoded_values = {
            f"{prefix}_{category_name}": encoded_values[f"{prefix}_{category_name}"]
            for category_name in category_names
        }

        has_missing_values = cast(bool, series.isna().any())
        if has_missing_values:
            ordered_encoded_values[f"{prefix}_na"] = series.isna().astype(bool)

        encoded_parts.append(
            pd.DataFrame(ordered_encoded_values, index=dataframe.index)
        )

    return pd.concat(encoded_parts, axis=1)


# def _singularize_column_name(column_name: str) -> str:
#     normalized = _normalize_category_name(column_name)
#     if normalized.endswith("us"):
#         return normalized
#     if normalized.endswith("ies") and len(normalized) > 3:
#         return normalized[:-3] + "y"
#     if re.search(r"(xes|zes|ches|shes|sses|ses)$", normalized):
#         return normalized[:-2]
#     if normalized.endswith("s") and not normalized.endswith("ss"):
#         return normalized[:-1]
#     return normalized


def _normalize_category_name(value: object) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return normalized or "value"
