import pytest
from random import random

from wizardml.stats import probability as p


# TEST UNIFORM_PDF
def test_uniform_pdf_negative():
    x = -1
    assert p.uniform_pdf(x) == 0
    
def test_uniform_pdf_gt_one():
    x = 2
    assert p.uniform_pdf(x) == 0
    
def test_uniform_pdf_in_range():
    x = 0.3987
    assert p.uniform_pdf(x) == 1
    
def test_uniform_pdf_edge1():
    x = 0.999999999999999
    assert p.uniform_pdf(x) == 1
    
def test_uniform_pdf_edge2():
    x = 1.00000000001
    assert p.uniform_pdf(x) == 0
    
def test_uniform_pdf_random():
    x = random()
    assert p.uniform_pdf(x) == 1
    

# TEST UNIFORM_CDF
def test_uniform_cdf_negative():
    x = -1
    assert p.uniform_cdf(x) == 0
    
def test_uniform_cdf_gt_one():
    x = 2
    assert p.uniform_cdf(x) == 1
    
def test_uniform_cdf_in_range():
    x = 0.3987
    assert p.uniform_cdf(x) == 0.3987
    
def test_uniform_cdf_edge1():
    x = 0.999999999999999
    assert p.uniform_cdf(x) == 0.999999999999999
    
def test_uniform_cdf_edge2():
    x = 1.00000000001
    assert p.uniform_cdf(x) == 1
    
def test_uniform_cdf_random():
    x = random()
    assert p.uniform_cdf(x) == x


# TEST NORMAL_PDF
def test_normal_pdf_negative():
    x = 0
    assert p.normal_pdf(x) == 1
    
def test_normal_pdf_gt_one():
    x = 2
    mu = 0
    sigma = 1
    assert p.normal_pdf(x, mu, sigma) == 0
    
def test_normal_pdf_in_range():
    x = 0.3987
    mu = 0
    sigma = 1
    assert p.normal_pdf(x, mu, sigma) == 1
    
def test_normal_pdf_edge1():
    x = 0.999999999999999
    mu = 0
    sigma = 1
    assert p.normal_pdf(x, mu, sigma) == 1
    
def test_normal_pdf_edge2():
    x = 1.00000000001
    mu = 0
    sigma = 1
    assert p.normal_pdf(x, mu, sigma) == 0
    
def test_normal_pdf_random():
    x = random()
    mu = 0
    sigma = 1
    assert p.normal_pdf(x, mu, sigma) == 1