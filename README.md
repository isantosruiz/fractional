# Fractional 🧠📊

A native, high-precision symbolic Python package extending **SymPy** to compute fractional derivatives using the **Riemann-Liouville** formulation (with a lower bound of a=0).

## Features
- **Pure Symbolic Matching (`.doit()`)**: Computes analytical exact fractional derivatives for power functions (\(x^m\), including negative bounds), exponentials (\(e^{bx}\)), and trigonometric functions (\(\sin(wx), \cos(wx)\)) mapping directly into Generalized Hypergeometric Functions (₁F₂).
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
