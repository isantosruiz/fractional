import pytest
import sympy as sp
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from fractional import FractionalDerivative

def test_fractional_derivative_creation():
    x = sp.Symbol('x')
    fd = FractionalDerivative(x**2, x, 0.5)
    assert fd.alpha == sp.Rational(1, 2)

    with pytest.raises(ValueError):
        FractionalDerivative(x**2, 2, 0.5)

def test_fractional_derivative_linearity():
    x = sp.Symbol('x')
    alpha = sp.Rational(1, 2)
    expr_sum = x**2 + sp.exp(x)
    fd_sum = FractionalDerivative(expr_sum, x, alpha).doit()
    expected_sum = FractionalDerivative(x**2, x, alpha).doit() + FractionalDerivative(sp.exp(x), x, alpha).doit()
    assert fd_sum == expected_sum

def test_fractional_derivative_cases_base():
    x = sp.Symbol('x')
    assert FractionalDerivative(x**3, x, 0).doit() == x**3
    assert FractionalDerivative(x**3, x, 1).doit() == 3*x**2

def test_fractional_derivative_powers():
    x = sp.Symbol('x')
    alpha = sp.Rational(1, 2)
    fd_pow = FractionalDerivative(x**3, x, alpha).doit()
    assert fd_pow == (gamma(4) / gamma(sp.Rational(7, 2))) * x**(sp.Rational(5, 2))

def test_fractional_derivative_trig():
    x = sp.Symbol('x')
    alpha = sp.Rational(1, 2)
    fd_sin = FractionalDerivative(sp.sin(3*x), x, alpha).doit()
    expected_sin = (3 * x**(sp.Rational(1, 2)) / gamma(sp.Rational(3, 2))) * \
                   hyper([sp.S.One], [sp.Rational(3, 4), sp.Rational(5, 4)], -9 * x**2 / 4)
    assert fd_sin == expected_sin

def test_fractional_derivative_evalf():
    x = sp.Symbol('x')
    fd = FractionalDerivative(x**3 + sp.exp(2*x), x, 0.5)
    val_15 = fd.subs(x, 1.5).evalf(15)
    assert isinstance(val_15, sp.Float)
    assert abs(val_15 - 33.3803544767048) < 1e-10
