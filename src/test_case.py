import pytest
from tools import *


@pytest.mark.parametrize("name,rate,expected",
                         [("James", 10, "This will be the last 299.997  payment for the customer James"),
                          pytest.param("James", 10, "this can cauz eisse", marks=pytest.mark.xfail)])
def test_process_data(name, rate, expected):
    result = monthly_payment.invoke(name, rate)
    assert result == expected
