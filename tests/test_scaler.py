import pytest 

from src.wizardml.preprocessing import scaler as s

# TEST SCALE 
def test_scale():
    data = [[1, 2, 3], [4, 5, 6] [7, 8, 9]]
    expected_mean = [4, 5, 6]
    expected_stdev = [4.2426406, 4.2426406, 4.2426406]
    mean, stdev = s.scale(data)
    assert pytest.approx(mean) == expected_mean
    assert pytest.approx(stdev) == expected_stdev

# TEST RESCALER
def test_rescale():
    data = [[1, 2, 3], [4, 5, 6] [7, 8, 9]]
    expected_result = [[-0.707107, -0.707107, -0.707107], [0, 0, 0], [0.707107, 0.707107, 0.707107]]
    result = s.rescale(data)
    assert pytest.approx(result) == expected_result