import pandas as pd
from pandas.testing import assert_frame_equal

from ohe_pyspark.encoding import one_hot_encode_dataframe


def test_one_hot_encode_dataframe_singularizes_prefix_and_adds_na_indicator():
    source = pd.DataFrame({"animals": ["cat", "dog", "frog", "cat", None]})

    encoded = one_hot_encode_dataframe(source)

    expected = pd.DataFrame(
        {
            "animal_cat": [True, False, False, True, False],
            "animal_dog": [False, True, False, False, False],
            "animal_frog": [False, False, True, False, False],
            "animal_na": [False, False, False, False, True],
        }
    )

    assert_frame_equal(encoded, expected)


def test_one_hot_encode_dataframe_preserves_other_columns_and_uses_stable_order():
    source = pd.DataFrame(
        {
            "id": [2, 1, 3],
            "colors": ["blue", "red", "blue"],
            "status": ["new", "old", "new"],
        }
    )

    encoded = one_hot_encode_dataframe(source, columns=["colors", "status"])

    expected = pd.DataFrame(
        {
            "id": [2, 1, 3],
            "color_blue": [True, False, True],
            "color_red": [False, True, False],
            "status_new": [True, False, True],
            "status_old": [False, True, False],
        }
    )

    assert_frame_equal(encoded, expected)
