# Fractional 🧠📊

A native, high-precision symbolic Python package extending **SymPy** to compute fractional derivatives and integrals using the **Riemann-Liouville** formulation (with a lower bound of a=0).

## Features
- **Pure Symbolic Matching (`.doit()`)**: Computes analytical exact fractional derivatives for power functions (\(x^m\), including negative exponents), exponentials (\(e^{bx}\)) using the confluent hypergeometric function (₁F₁), and trigonometric functions (\(\sin(wx), \cos(wx)\)) using generalized hypergeometric functions (₁F₂).
- **Negative Orders**: Interprets \(D^{-\beta}_{0+}\) as the Riemann-Liouville fractional integral \(I^\beta_{0+}\), symbolically and numerically.
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

For a non-integer order \(\alpha\), the exponential rule respects the fixed
lower bound at zero:

\[
D_{0+}^{\alpha} e^{bx}
= \frac{x^{-\alpha}}{\Gamma(1-\alpha)}
{}_1F_1\left(1;1-\alpha;bx\right).
\]

The simpler expression \(b^\alpha e^{bx}\) belongs to a different choice of
fractional operator or boundary conditions and is not used here.

## Mathematical Domain

For real orders, a positive value represents a derivative, zero is the identity,
and a negative value represents a fractional integral:

\[
D^{-\beta}_{0+}f(x)=I^\beta_{0+}f(x)
=\frac{1}{\Gamma(\beta)}\int_0^x(x-t)^{\beta-1}f(t)\,dt,
\qquad \beta>0.
\]

A concrete positive integer order is evaluated with SymPy's ordinary
derivative. An integer-valued symbolic order known to be nonnegative is kept as
an unevaluated classical `Derivative`, so substituting a specific integer later
remains mathematically correct. Finite complex orders are supported in symbolic
closed forms when their assumptions make the expression unambiguous; numerical
quadrature currently requires a real order.

Concrete negative integer orders are constructed as definite integrals from
zero and simplified by SymPy. Therefore elementary antiderivatives remain
elementary:

```python
FractionalDerivative(sp.exp(2*x), x, -1).doit()
# exp(2*x)/2 - 1/2
```

For the power rule \(x^m\), the defining integral at the fixed lower bound zero
requires \(\operatorname{Re}(m)>-1\). A power known not to satisfy that
condition raises `ValueError` instead of returning a value obtained only by
analytic continuation. When the integrality of a symbolic order is unknown,
closed forms with singular integer parameters are deliberately left
unevaluated.

## Numerical Fallback

When no symbolic rule matches, evaluate at a positive real point with the
requested number of decimal digits:

```python
fd = FractionalDerivative(sp.sin(x**2), x, sp.Rational(1, 2))

print(fd.eval_at(1, 30))
# 1.11361910109605500220019376399

print(fd.subs(x, 1).evalf(30))
# 1.11361910109605500220019376399
```

For positive orders, the numerical fallback uses the equivalent Caputo integral
together with the lower-boundary terms required by the Riemann-Liouville
definition. For negative orders, it evaluates the fractional integral directly.
It currently requires a finite real order and a positive real evaluation point;
positive derivative orders also require finite initial derivatives at the lower
bound.
