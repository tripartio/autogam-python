import pytest
import numpy as np
import pandas as pd
from statsmodels.gam.tests.test_penalized import df_autos
from autogam.autogam import AutoGAM, gam_formula

@pytest.fixture
def random_data():
    """
    Create random test data for AutoGAM.
    """
    return pd.DataFrame({
        'x1': np.random.uniform(0, 10, 100),
        'x2': np.random.uniform(0, 10, 100),
        'y': np.random.uniform(0, 10, 100)
    })

@pytest.fixture
def autos_data():
    """
    Provide the df_autos dataset from statsmodels.
    """
    return df_autos

def test_gam_with_simple_data(random_data):
    """
    Test AutoGAM with simple random data.
    """
    ag = AutoGAM(random_data, 'y')
    ag.summary()
    ag.print()

    # Assert performance metrics are calculated
    assert 'mae' in ag.perf
    assert 'rmse' in ag.perf
    assert ag.perf['mae'] > 0
    assert ag.perf['rmse'] > 0

def test_gam_with_autos_data(autos_data):
    """
    Test AutoGAM with the df_autos dataset.
    """
    ag = AutoGAM(autos_data, 'city_mpg')
    ag.summary()
    ag.print()

    # Assert performance metrics are calculated
    assert 'mae' in ag.perf
    assert 'rmse' in ag.perf
    assert ag.perf['mae'] > 0
    assert ag.perf['rmse'] > 0

def test_gam_formula_random_data(random_data):
    """
    Test the gam_formula function with random data.
    """
    formula, splines = gam_formula(random_data, 'y')

    # Assert formula is a string
    assert isinstance(formula, str)
    assert formula.startswith('y ~')

    # Assert splines structure
    assert isinstance(splines, list)
    assert len(splines) == 3
    assert all(isinstance(splines[i], list) for i in range(3))

def test_gam_formula_autos_data(autos_data):
    """
    Test the gam_formula function with the df_autos dataset.
    """
    formula, splines = gam_formula(autos_data, 'city_mpg')

    # Assert formula is a string
    assert isinstance(formula, str)
    assert formula.startswith('city_mpg ~')

    # Assert splines structure
    assert isinstance(splines, list)
    assert len(splines) == 3
    assert all(isinstance(splines[i], list) for i in range(3))
