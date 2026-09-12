# Fractional 🧠📊

A native, high-precision symbolic Python package extending **SymPy** to compute fractional derivatives using the **Riemann-Liouville** formulation (with a lower bound of a=0).

## Features
- **Pure Symbolic Matching (`.doit()`)**: Computes analytical exact fractional derivatives for power functions (\(x^m\), including negative exponents), exponentials (\(e^{bx}\)) using the confluent hypergeometric function (₁F₁), and trigonometric functions (\(\sin(wx), \cos(wx)\)) using generalized hypergeometric functions (₁F₂).
- **Linearity & Constants Preservation**: Handles nested algebraic structures seamlessly.
- **Arbitrary Precision Numerical Fallback (`.evalf()`)**: Inherits `mpmath` under the hood to evaluate solutions numerically with exact bit-depth control using Grünwald-Letnikov approximations.

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
Ensure everything works perfectly with:
```bash
pytest
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
