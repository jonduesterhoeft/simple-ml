import pytest

from src.wizardml.math.stats import stats as s


# TEST MEAN
def test_mean_null():
    x = []
    assert s.mean(x) == None
    
def test_mean_zero():
    x = [0]
    assert s.mean(x) == 0
    
def test_mean_one():
    x = [1]
    assert s.mean(x) == 1
    
def test_mean_positive():
    x = [0, 1, 2, 3.0]
    assert s.mean(x) == 1.5
    
def test_mean_negative():
    x = [0, -1, -2, -3]
    assert s.mean(x) == -1.5


# TEST MEDIAN
def test_median_null():
    x = []
    assert s.median(x) == None
    
def test_median_zero():
    x = [0]
    assert s.median(x) == 0
    
def test_median_positive_even_ordered():
    x = [1, 2, 3, 4]
    assert s.median(x) == 2.5
    
def test_median_positive_even_unordered():
    x = [1, 4, 3, 2]
    assert s.median(x) == 2.5
    
def test_median_positive_odd_ordered():
    x = [1, 2, 3, 4, 5]
    assert s.median(x) == 3
    
def test_median_positive_odd_unordered():
    x = [1, 4, 5, 2, 3]
    assert s.median(x) == 3
    
def test_median_negative_even_ordered():
    x = [-2, 1, 3, 4]
    assert s.median(x) == 2
    
def test_median_negative_even_unordered():
    x = [1, 4, 3, -2]
    assert s.median(x) == 2
    
def test_median_negative_odd_ordered():
    x = [-3, 1, 2, 4, 5]
    assert s.median(x) == 2
    
def test_median_negative_odd_unordered():
    x = [1, 4, 5, 2, -3]
    assert s.median(x) == 2
    

# TEST QUANTILE
def test_quantile_null():
    x = []
    p = 0.5
    assert s.quantile(x, p) == None
    
def test_quantile_single():
    x = [1]
    p = 0.5
    assert s.quantile(x, p) == 1
    
def test_quantile_ordered_two_low():
    x = [1, 2]
    p = 0.2
    assert s.quantile(x, p) == 1
    
def test_quantile_ordered_two_mid():
    x = [1, 2]
    p = 0.5
    assert s.quantile(x, p) == 2
    
def test_quantile_ordered_two_high():
    x = [1, 2]
    p = 0.8
    assert s.quantile(x, p) == 2
    
def test_quantile_unordered_three_low():
    x = [3, 1, 2]
    p = 0.2
    assert s.quantile(x, p) == 1
    
def test_quantile_unordered_three_mid():
    x = [3, 1, 2]
    p = 0.5
    assert s.quantile(x, p) == 2
    
def test_quantile_unordered_three_high():
    x = [3, 1, 2]
    p = 0.8
    assert s.quantile(x, p) == 3
    
def test_quantile_unordered_ten_seven():
    x = [3, 6, 1, 10, 7, 4, 9, 2, 5, 8]
    p = 0.6
    assert s.quantile(x, p) == 7
    
def test_quantile_unordered_ten_two():
    x = [3, 6, 1, 10, 7, 4, 9, 2, 5, 8]
    p = 0.1
    assert s.quantile(x, p) == 2
    

# TEST MODE
def test_mode_null():
    x = []
    assert s.mode(x) == None

def test_mode_zero():
    x = [0]
    assert s.mode(x) == [0]

def test_mode_normal():
    x = [1, 1, 1, 2, 2, 3, 3, 3, 3]
    assert s.mode(x) == [3]
    
def test_mode_multiple():
    x = [0, 0, 1, 1]
    assert s.mode(x) == [0, 1]
    
    
# TES RANGE
def test_range_null():
    x = []
    assert s.range(x) == None

def test_range_single():
    x=[100]
    assert s.range(x) == 0
    
def test_range_positive():
    x = [1, 50, 100]
    assert s.range(x) == 99
    
def test_range_negative():
    x = [-100, 0, 100]
    assert s.range(x) == 200
    

# TEST DE_MEAN
def test_de_mean_null():
    x = []
    assert s.subtract_mean(x) == None

def test_de_mean_single():
    x = [100]
    assert s.subtract_mean(x) == [0]
    
def test_de_mean_three():
    x = [1, 2, 3]
    assert s.subtract_mean(x) == [-1, 0, 1]

def test_de_mean_five():
    x = [0, 5, 10, 15, 20]
    assert s.subtract_mean(x) == [-10, -5, 0, 5, 10]
    

# TEST VARIANCE
def test_variance_null():
    x = []
    assert s.variance(x) == None

def test_variance_single():
    x = [100]
    assert s.variance(x) == 0
    
def test_variance_three():
    x = [1, 2, 3]
    assert s.variance(x) == 1

def test_variance_five():
    x = [0, 5, 10, 15, 20]
    assert s.variance(x) == 62.5
    

# TEST STANDARD DEVIATION
def test_std_null():
    x = []
    assert s.std(x) == None

def test_std_single():
    x = [100]
    assert s.std(x) == 0
    
def test_std_three():
    x = [1, 2, 3]
    assert s.std(x) == 1

def test_std_five():
    x = [0, 5, 10, 15, 20]
    assert s.std(x) == pytest.approx(7.905694)
    

# TEST INTERQUARTILE_RANGE
def test_iqr_null():
    x = []
    assert s.iqr(x) == None

def test_iqr_single():
    x = [100]
    assert s.iqr(x) == 0
    
def test_iqr_ten():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert s.iqr(x) == 5
    

# TEST COVARIANCE
def test_covariance_null():
    x = []
    y = []
    assert s.cov(x, y) == None
    
def test_covariance_single():
    x = [100]
    y = [100]
    assert s.cov(x, y) == 0
    
def test_covariance_multi():
    x = [1, 2, 3, 4, 5]
    y = [6, 7, 8, 9, 10]
    assert s.cov(x, y) == 2.5
    
def test_covariance_multi_reversed():
    x = [1, 2, 3, 4, 5]
    y = [10, 9, 8, 7, 6]
    assert s.cov(x, y) == -2.5
    
def test_covariance_none():
    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    assert s.cov(x, y) == 0
    
def test_covariance_real():
    x = [0, 2, 3, 4]
    y = [0, 1, 5, 10]
    assert s.cov(x, y) == pytest.approx(7)
    

# TEST CORRELATION
def test_correlation_null():
    x = []
    y = []
    assert s.corr(x, y) == None
    
def test_correlation_single():
    x = [100]
    y = [100]
    assert s.corr(x, y) == 0
    
def test_correlation_multi():
    x = [1, 2, 3, 4, 5]
    y = [6, 7, 8, 9, 10]
    assert s.corr(x, y) == pytest.approx(1)
    
def test_correlation_multi_reversed():
    x = [1, 2, 3, 4, 5]
    y = [10, 9, 8, 7, 6]
    assert s.corr(x, y) == pytest.approx(-1)
    
def test_correlation_none():
    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]
    assert s.corr(x, y) == 0
    
def test_correlation_real():
    x = [0, 2, 3, 4]
    y = [0, 1, 5, 10]
    assert s.corr(x, y) == pytest.approx(0.901611)


if __name__ == '__main__':
    pass