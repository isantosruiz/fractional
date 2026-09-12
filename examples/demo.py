#!/usr/bin/env python3
"""Small executable tour of the ``fractional`` package."""

import sympy as sp

from fractional import FractionalDerivative


def fractional_derivative(expr, x, alpha, x0=0):
    """Evaluate and display one symbolic fractional derivative."""
    result = FractionalDerivative(expr, x, alpha, x0=x0).doit()
    print(f"\nD^{alpha} of {expr}, with x0={x0}:")
    sp.pprint(result)
    return result


def main():
    # Positive x keeps real-valued fractional powers on their principal branch.
    x = sp.Symbol("x", positive=True)
    alpha = sp.Rational(1, 2)

    print("Riemann-Liouville derivatives with lower bound a = 0")

    constant = fractional_derivative(sp.Integer(1), x, alpha)
    power = fractional_derivative(x**2, x, alpha)
    exponential = fractional_derivative(sp.exp(2 * x), x, alpha)
    sine = fractional_derivative(sp.sin(3 * x), x, alpha)

    # A composite expression demonstrates linearity.
    composite_expr = x**2 + sp.exp(2 * x) + sp.sin(3 * x)
    composite = fractional_derivative(composite_expr, x, alpha)

    # Integer orders fall back to SymPy's ordinary derivative.
    integer_order = fractional_derivative(sp.exp(2 * x), x, sp.Integer(1))

    # Negative orders are Riemann-Liouville fractional integrals.
    first_integral = fractional_derivative(sp.exp(2 * x), x, sp.Integer(-1))
    custom_integral = fractional_derivative(
        sp.exp(2 * x),
        x,
        sp.Integer(-1),
        x0=1,
    )

    # Basic symbolic checks.
    assert constant == 1 / sp.sqrt(sp.pi * x)
    assert power == 8 * x ** sp.Rational(3, 2) / (3 * sp.sqrt(sp.pi))
    assert sp.simplify(composite - (power + exponential + sine)) == 0
    assert integer_order == 2 * sp.exp(2 * x)
    assert first_integral == (sp.exp(2*x) - 1)/2
    assert custom_integral == (sp.exp(2*x) - sp.exp(2))/2

    # The half-derivative of exp(2x) also has a closed form involving erf.
    exponential_closed_form = (
        1 / sp.sqrt(sp.pi * x)
        + sp.sqrt(2) * sp.exp(2 * x) * sp.erf(sp.sqrt(2 * x))
    )
    assert sp.simplify(sp.hyperexpand(exponential) - exponential_closed_form) == 0

    print("\nNumerical value of D^(1/2) exp(2x) at x = 1:")
    sp.pprint(exponential.subs(x, 1).evalf(20))

    # sin(x**2) has no dedicated symbolic rule, so eval_at evaluates the
    # Riemann-Liouville definition numerically with the requested precision.
    unsupported = FractionalDerivative(sp.sin(x**2), x, alpha)
    print("\nNumerical fallback for D^(1/2) sin(x**2) at x = 1:")
    sp.pprint(unsupported.eval_at(1, 30))

    print("\nNegative half-order (fractional integral) at x = 1:")
    sp.pprint(FractionalDerivative(sp.sin(x**2), x, -alpha).eval_at(1, 30))

    print("\nAll symbolic and numerical checks passed.")


if __name__ == "__main__":
    main()
