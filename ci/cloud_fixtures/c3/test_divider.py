from divider import divide
def test_normal():
    assert divide(10, 2) == 5
def test_zero_division():
    assert divide(10, 0) is None
def test_negative():
    assert divide(-10, 2) == -5
