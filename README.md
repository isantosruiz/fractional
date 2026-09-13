# Fractional 🧠📊

A native, high-precision symbolic Python package extending **SymPy** to compute fractional derivatives and integrals using the **Riemann-Liouville** formulation. The lower bound `x0` is configurable and defaults to zero.

## Features
- **Pure Symbolic Matching (`.doit()`)**: Computes analytical exact fractional derivatives for power functions ($`x^m`$, including negative exponents), exponentials ($`e^{bx}`$) using the confluent hypergeometric function (₁F₁), and trigonometric functions ($`\sin(wx), \cos(wx)`$) using generalized hypergeometric functions (₁F₂).
- **Configurable Lower Bound**: Accepts `x0` as the lower terminal while preserving `x0=0` as the backward-compatible default.
- **Negative Orders**: Interprets $`D^{-\beta}_{x_0+}`$ as the Riemann-Liouville fractional integral $`I^\beta_{x_0+}`$, symbolically and numerically.
- **Linearity & Constants Preservation**: Handles sums, scalar factors, and arbitrary constants.
- **Precision-Controlled Numerical Fallback**: Evaluates smooth expressions without a symbolic rule directly from the Riemann-Liouville definition using adaptive `mpmath` quadrature. It supports both `.eval_at(...)` and SymPy's `.subs(...).evalf(...)` workflow.

## Installation & Development

Clone the repository and install it in editable mode with development dependencies:

```bash
git clone https://github.com/isantosruiz/fractional.git
cd fractional
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running Tests
Run the complete test suite with:
```bash
pytest
```

Run the executable demonstration with:

```bash
python examples/demo.py
```

## Quick Example
```python
import sympy as sp
from fractional import FractionalDerivative

x = sp.Symbol('x')
# Half-derivative of x^2 + sin(x)
fd = FractionalDerivative(x**2 + sp.sin(x), x, 0.5)

print(fd.doit())
```

Pass `x0` as a fourth argument or keyword to choose a different lower bound:

```python
first_integral = FractionalDerivative(sp.exp(2*x), x, -1, x0=1)
print(first_integral.doit())
# exp(2*x)/2 - exp(2)/2
```

For a non-integer order $`\alpha`$, the exponential rule respects the selected
lower bound:

$$
D_{x_0+}^{\alpha} e^{bx}
= \frac{e^{b x_0}(x-x_0)^{-\alpha}}{\Gamma(1-\alpha)}
{}_1F_1\left(1;1-\alpha;b(x-x_0)\right).
$$

The simpler expression $`b^\alpha e^{bx}`$ belongs to a different choice of
fractional operator or boundary conditions and is not used here.

## Mathematical Domain

For real orders, a positive value represents a derivative, zero is the identity,
and a negative value represents a fractional integral:

$$
D^{-\beta}_{x_0+}f(x)=I^\beta_{x_0+}f(x)
=\frac{1}{\Gamma(\beta)}\int_{x_0}^x(x-t)^{\beta-1}f(t)\mathrm{d}t,
\qquad \beta>0.
$$

A concrete positive integer order is evaluated with SymPy's ordinary
derivative. An integer-valued symbolic order known to be nonnegative is kept as
an unevaluated classical `Derivative`, so substituting a specific integer later
remains mathematically correct. Finite complex orders are supported in symbolic
closed forms when their assumptions make the expression unambiguous; numerical
quadrature currently requires a real order.

Concrete negative integer orders are constructed as definite integrals from
`x0` and simplified by SymPy. Therefore elementary antiderivatives remain
elementary:

```python
FractionalDerivative(sp.exp(2*x), x, -1, x0=1).doit()
# exp(2*x)/2 - exp(2)/2
```

For the shifted power rule $`(x-x_0)^m`$, the defining integral at the lower
terminal requires $`\operatorname{Re}(m)>-1`$. A power known not to satisfy that
condition raises `ValueError` instead of returning a value obtained only by
analytic continuation. When the integrality of a symbolic order is unknown,
closed forms with singular integer parameters are deliberately left
unevaluated.

## Numerical Fallback

When no symbolic rule matches, evaluate at a real point greater than `x0` with
the requested number of decimal digits:

```python
fd = FractionalDerivative(sp.sin(x**2), x, sp.Rational(1, 2), x0=sp.Rational(1, 4))

print(fd.eval_at(1, 30))
# 1.11563736176290459323947882831

print(fd.subs(x, 1).evalf(30))
# 1.11563736176290459323947882831
```

For positive orders, the numerical fallback uses the equivalent Caputo integral
together with the lower-boundary terms required by the Riemann-Liouville
definition. For negative orders, it evaluates the fractional integral directly.
It currently requires a finite real order, a finite real `x0`, and a real
evaluation point strictly greater than `x0`; positive derivative orders also
require finite initial derivatives at the lower bound.
