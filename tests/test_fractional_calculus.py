import pytest
import sympy as sp
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from fractional import FractionalDerivative


def rl_half_from_definition(expr, x):
    """Construct D^(1/2) from the defining Riemann-Liouville integral."""
    t = sp.Symbol('t', positive=True)
    fractional_integral = sp.integrate(
        expr.subs(x, t) / sp.sqrt(x - t),
        (t, 0, x),
    )
    return sp.diff(fractional_integral, x) / sp.sqrt(sp.pi)


def rl_integral_from_definition(expr, x, beta):
    """Construct the Riemann-Liouville integral of positive order beta."""
    t = sp.Symbol('t', positive=True)
    return sp.integrate(
        (x - t)**(beta - 1) * expr.subs(x, t),
        (t, 0, x),
    ) / gamma(beta)


def test_fractional_derivative_creation():
    x = sp.Symbol('x')
    fd = FractionalDerivative(x**2, x, 0.5)
    assert fd.alpha == sp.Rational(1, 2)

    with pytest.raises(ValueError):
        FractionalDerivative(x**2, 2, 0.5)
    assert FractionalDerivative(x**2, x, -sp.Rational(1, 2)).alpha == -sp.Rational(1, 2)
    assert FractionalDerivative(x**2, x, sp.I).alpha == sp.I
    with pytest.raises(ValueError, match='order must be finite'):
        FractionalDerivative(x**2, x, sp.oo)


def test_symbolic_integer_order_remains_classical():
    x = sp.Symbol('x')
    n = sp.Symbol('n', integer=True, nonnegative=True)
    result = FractionalDerivative(x**4, x, n).doit()

    assert result == sp.Derivative(x**4, (x, n), evaluate=False)
    assert result.subs(n, 0).doit() == x**4
    assert result.subs(n, 2).doit() == 12*x**2
    assert result.subs(n, 5).doit() == 0


def test_unknown_symbolic_order_avoids_singular_closed_forms():
    x = sp.Symbol('x')
    alpha = sp.Symbol('alpha', positive=True)
    fd = FractionalDerivative(sp.exp(x), x, alpha)

    assert fd.doit() == fd
    assert fd.subs(alpha, 1).doit() == sp.exp(x)

    noninteger_alpha = sp.Symbol(
        'noninteger_alpha',
        positive=True,
        integer=False,
    )
    result = FractionalDerivative(sp.exp(x), x, noninteger_alpha).doit()
    assert result.has(sp.hyper)


def test_fractional_derivative_constants():
    x = sp.Symbol('x', positive=True)
    c = sp.Symbol('c')
    alpha = sp.Rational(1, 2)

    assert FractionalDerivative(2, x, alpha).doit() == 2 / sp.sqrt(sp.pi*x)
    assert FractionalDerivative(c, x, alpha).doit() == c / sp.sqrt(sp.pi*x)
    assert FractionalDerivative(2, x, 0).doit() == 2
    assert FractionalDerivative(2, x, 1).doit() == 0


def test_fractional_derivative_linearity():
    x = sp.Symbol('x', positive=True)
    alpha = sp.Rational(1, 2)
    expr_sum = x**2 + sp.exp(x) + 2
    fd_sum = FractionalDerivative(expr_sum, x, alpha).doit()
    expected_sum = (
        FractionalDerivative(x**2, x, alpha).doit()
        + FractionalDerivative(sp.exp(x), x, alpha).doit()
        + FractionalDerivative(2, x, alpha).doit()
    )
    assert fd_sum == expected_sum


def test_fractional_derivative_cases_base():
    x = sp.Symbol('x')
    assert FractionalDerivative(x**3, x, 0).doit() == x**3
    assert FractionalDerivative(x**3, x, 1).doit() == 3*x**2


def test_fractional_derivative_powers():
    x = sp.Symbol('x', positive=True)
    alpha = sp.Rational(1, 2)
    fd_pow = FractionalDerivative(x**3, x, alpha).doit()
    assert fd_pow == (gamma(4) / gamma(sp.Rational(7, 2))) * x**(sp.Rational(5, 2))

    definition = rl_half_from_definition(x**sp.Rational(5, 2), x)
    result = FractionalDerivative(x**sp.Rational(5, 2), x, alpha).doit()
    assert sp.simplify(result - definition) == 0


def test_power_rule_rejects_nonintegrable_lower_bound():
    x = sp.Symbol('x', positive=True)
    alpha = sp.Rational(1, 2)

    # x**(-1/2) is locally integrable at zero and its half derivative is zero.
    assert FractionalDerivative(x**(-alpha), x, alpha).doit() == 0

    for exponent in (-1, -sp.Rational(3, 2), -2):
        with pytest.raises(ValueError, match=r're\(m\) > -1'):
            FractionalDerivative(x**exponent, x, alpha).doit()


def test_negative_orders_are_fractional_integrals():
    x = sp.Symbol('x', positive=True)
    beta = sp.Rational(1, 2)

    cases = (
        sp.Integer(2),
        x**2,
        sp.exp(2*x),
        sp.sin(3*x),
        sp.cos(3*x),
    )
    for expr in cases:
        result = FractionalDerivative(expr, x, -beta).doit()
        definition = rl_integral_from_definition(expr, x, beta)
        assert sp.simplify(sp.hyperexpand(result) - definition) == 0


def test_negative_integer_orders_use_definite_integrals_from_zero():
    x = sp.Symbol('x', positive=True)

    exponential_integral = FractionalDerivative(sp.exp(2*x), x, -1).doit()
    sine_integral = FractionalDerivative(sp.sin(3*x), x, -1).doit()
    cosine_integral = FractionalDerivative(sp.cos(3*x), x, -1).doit()

    assert sp.simplify(sp.hyperexpand(exponential_integral) - (sp.exp(2*x) - 1)/2) == 0
    assert sp.simplify(sp.hyperexpand(sine_integral) - (1 - sp.cos(3*x))/3) == 0
    assert sp.simplify(sp.hyperexpand(cosine_integral) - sp.sin(3*x)/3) == 0

    n = sp.Symbol('n', integer=True, negative=True)
    symbolic_integral = FractionalDerivative(sp.exp(x), x, n).doit()
    assert sp.simplify(sp.hyperexpand(symbolic_integral.subs(n, -1)) - (sp.exp(x) - 1)) == 0


def test_fractional_derivative_exponential_rl_lower_bound_zero():
    x = sp.Symbol('x', positive=True)
    alpha = sp.Rational(1, 2)
    result = FractionalDerivative(sp.exp(2*x), x, alpha).doit()
    expected = x**(-alpha) / gamma(1 - alpha) * hyper(
        [sp.S.One], [1 - alpha], 2*x
    )
    assert result == expected

    rl_definition = rl_half_from_definition(sp.exp(2*x), x)
    assert sp.simplify(sp.hyperexpand(result) - rl_definition) == 0
    assert FractionalDerivative(sp.exp(2*x), x, 1).doit() == 2 * sp.exp(2*x)


def test_fractional_derivative_trig():
    x = sp.Symbol('x', positive=True)
    alpha = sp.Rational(1, 2)
    fd_sin = FractionalDerivative(sp.sin(3*x), x, alpha).doit()
    expected_sin = (3 * x**(sp.Rational(1, 2)) / gamma(sp.Rational(3, 2))) * \
                   hyper([sp.S.One], [sp.Rational(3, 4), sp.Rational(5, 4)], -9 * x**2 / 4)
    assert fd_sin == expected_sin

    for expr in (sp.sin(3*x), sp.cos(3*x)):
        result = FractionalDerivative(expr, x, alpha).doit()
        definition = rl_half_from_definition(expr, x)
        assert sp.simplify(sp.hyperexpand(result) - definition) == 0


def test_fractional_derivative_evalf():
    x = sp.Symbol('x')
    fd = FractionalDerivative(x**3 + sp.exp(2*x), x, 0.5)
    val_15 = fd.subs(x, 1.5).evalf(15)
    expected = sp.Float('33.43465145054713231326955', 15)
    assert isinstance(val_15, sp.Float)
    assert abs(val_15 - expected) < 1e-10


def test_numerical_fallback_for_unsupported_expression():
    x = sp.Symbol('x', positive=True)
    fd = FractionalDerivative(sp.sin(x**2), x, sp.Rational(1, 2))
    expected = sp.Float(
        '1.11361910109605500220019376399120564034553545224124796335',
        55,
    )

    # The reference value comes from applying the power rule term by term to
    # sin(x**2), independently of the quadrature used by the implementation.
    via_eval_at = fd.eval_at(1, 50)
    via_substitution = fd.subs(x, 1).evalf(50)
    via_evalf_subs = fd.evalf(50, subs={x: 1})

    assert abs(via_eval_at - expected) < sp.Float('1e-48')
    assert abs(via_substitution - expected) < sp.Float('1e-48')
    assert abs(via_evalf_subs - expected) < sp.Float('1e-48')


def test_numerical_fallback_above_first_order():
    x = sp.Symbol('x', positive=True)
    fd = FractionalDerivative(sp.sin(x**2), x, sp.Rational(3, 2))
    expected = sp.Float(
        '0.20776442709211058427714221471809731109199722541712',
        50,
    )
    assert abs(fd.eval_at(1, 40) - expected) < sp.Float('1e-38')


def test_numerical_fallback_includes_boundary_terms():
    x = sp.Symbol('x', positive=True)
    fd = FractionalDerivative(sp.cos(x**2), x, sp.Rational(1, 2))
    expected = sp.Float(
        '-0.35252842821347457367224835340845791197843207457842',
        50,
    )

    # cos(x**2) equals 1 at the lower bound, so this exercises the nonzero
    # Riemann-Liouville boundary contribution as well as the integral.
    assert abs(fd.eval_at(1, 40) - expected) < sp.Float('1e-38')


@pytest.mark.parametrize(
    ('alpha', 'expected'),
    (
        (
            -sp.Rational(1, 2),
            '0.54016438577284366558644214937484146535362756688707',
        ),
        (
            -sp.Integer(1),
            '0.31026830172338110180815242316539650757450938883245',
        ),
        (
            -sp.Rational(3, 2),
            '0.16361075978445647055255649378045243288477242986673',
        ),
    ),
)
def test_numerical_fallback_for_negative_order(alpha, expected):
    x = sp.Symbol('x', positive=True)
    fd = FractionalDerivative(sp.sin(x**2), x, alpha)
    expected = sp.Float(expected, 50)

    assert abs(fd.eval_at(1, 40) - expected) < sp.Float('1e-38')
    assert abs(fd.subs(x, 1).evalf(40) - expected) < sp.Float('1e-38')


def test_numerical_fallback_domain_errors():
    x = sp.Symbol('x', positive=True)
    fd = FractionalDerivative(sp.sin(x**2), x, sp.Rational(1, 2))

    with pytest.raises(ValueError, match='evaluation point must be positive'):
        fd.eval_at(0)
    with pytest.raises(ValueError, match='evaluation point must be positive'):
        fd.eval_at(-1)

    complex_order = FractionalDerivative(sp.sin(x**2), x, sp.I)
    with pytest.raises(ValueError, match='finite real'):
        complex_order.eval_at(1)
