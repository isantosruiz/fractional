import sympy as sp
import mpmath
from sympy.core.evalf import dps_to_prec, prec_to_dps


def _real_mpf(value, dps, name):
    """Convert a finite real SymPy value to an mpmath number."""
    numeric = sp.N(value, dps)
    if (
        numeric.is_number is not True
        or numeric.is_real is not True
        or numeric.is_finite is not True
    ):
        raise ValueError(f"{name} must be a finite real number.")
    return mpmath.mpf(str(numeric))


def _mpmath_value(value, dps):
    """Convert a real or complex numeric value without passing through float."""
    if isinstance(value, (mpmath.mpf, mpmath.mpc)):
        return value

    numeric = sp.N(value, dps)
    if numeric.is_number is not True or numeric.is_finite is not True:
        raise ValueError("The expression could not be evaluated numerically.")
    if numeric.is_real is True:
        return mpmath.mpf(str(numeric))

    real, imag = numeric.as_real_imag()
    return mpmath.mpc(str(sp.N(real, dps)), str(sp.N(imag, dps)))


def _sympy_float(value, prec):
    """Convert an mpmath value to a SymPy number at binary precision ``prec``."""
    value = mpmath.mpmathify(value)
    if isinstance(value, mpmath.mpc):
        return (
            sp.Float(value.real, precision=prec)
            + sp.I * sp.Float(value.imag, precision=prec)
        )
    return sp.Float(value, precision=prec)


class FractionalDerivative(sp.Expr):
    r"""
    Symbolically represents the Riemann-Liouville fractional derivative.

    Mathematical Explanation
    ========================
    Unlike classical calculus, where the order of differentiation $n$ belongs
    to positive integers ($\mathbb{N}$), fractional calculus generalizes the
    operator to arbitrary orders. Positive orders represent derivatives and
    negative real orders represent Riemann-Liouville fractional integrals.

    This subroutine uses the **Riemann-Liouville (R-L)** definition with a fixed lower
    bound at $a = 0$. For a given function $f(x)$ and order $\alpha > 0$ 
    (where $n = \lceil \alpha \rceil$), the operator is formally defined as:

    .. math::
        D^\alpha f(x) = \frac{1}{\Gamma(n - \alpha)} \frac{d^n}{dx^n} \int_{0}^{x} (x - t)^{n - \alpha - 1} f(t) \, dt

    For a negative real order $\alpha=-\beta$, with $\beta>0$, the same
    operator represents the Riemann-Liouville fractional integral:

    .. math::
        D^{-\beta} f(x) = I^\beta f(x)
        = \frac{1}{\Gamma(\beta)} \int_0^x (x-t)^{\beta-1}f(t)\,dt

    Where $\Gamma(z)$ is the Euler Gamma function, which extends factorials to real numbers.

    Operator Properties:
    --------------------
    1. **Memory Effect (Non-Locality):** The derivative at point $x$ does not depend 
       solely on the local neighborhood of $x$, but on the entire history of the function 
       over the closed interval $[0, x]$.
    2. **Linearity:** It satisfies $D^\alpha [c_1 f(x) + c_2 g(x)] = c_1 D^\alpha f(x) + c_2 D^\alpha g(x)$.
    3. **Derivative of a Constant:** Unlike classical calculus, the fractional derivative 
       of a constant **is not zero** under the Riemann-Liouville framework. 
       For $f(x) = 1$, the operator yields $D^\alpha(1) = \frac{x^{-\alpha}}{\Gamma(1-\alpha)}$.

    Implemented Closed Forms (Symbolic Mapping)
    ===========================================
    To guarantee analytical exact solutions inside SymPy, the ``.doit()`` method uses
    pattern matching based on the following foundational rules:

    * **Powers ($x^m$):**
      .. math::
          D^\alpha (x^m) = \frac{\Gamma(m+1)}{\Gamma(m + 1 - \alpha)} x^{m - \alpha}
      The classical integral additionally requires convergence at the lower
      bound (for example, $\operatorname{Re}(m)>-1$).

    * **Exponentials ($e^{bx}$):**
      .. math::
          D^\alpha (e^{bx}) = \frac{x^{-\alpha}}{\Gamma(1-\alpha)}
          \,{}_1F_1\left(1; 1-\alpha; bx\right)
      for non-integer $\alpha$. The boundary contribution at $x=0$ is essential;
      consequently, $b^\alpha e^{bx}$ is not the Riemann-Liouville derivative
      with the fixed lower bound used by this class.

    * **Trigonometric Functions ($\sin(wx)$ and $\cos(wx)$):**
      Due to the lower bound at $a=0$, analytical solutions require the use 
      of Generalized Hypergeometric Functions $_1F_2$:
      .. math::
          D^\alpha (\sin(wx)) = \frac{w x^{1-\alpha}}{\Gamma(2-\alpha)} \, _1F_2\left(1; 1-\frac{\alpha}{2}, \frac{3}{2}-\frac{\alpha}{2}; -\frac{w^2x^2}{4}\right)
      .. math::
          D^\alpha (\cos(wx)) = \frac{x^{-\alpha}}{\Gamma(1-\alpha)} \, _1F_2\left(1; \frac{1}{2}-\frac{\alpha}{2}, 1-\frac{\alpha}{2}; -\frac{w^2x^2}{4}\right)

    Parameters
    ==========
    expr : Expr
        The mathematical expression or scalar function to differentiate.
    x : Symbol
        The independent variable with respect to which differentiation is performed.
    alpha : Expr or number
        The operator order. Positive values represent derivatives, zero is the
        identity, and negative real values represent fractional integrals.
        Abstract and complex symbolic orders are accepted, but symbolic closed
        forms that are singular at integer parameters require sufficient
        assumptions.

    Examples
    ========
    >>> import sympy as sp
    >>> from fractional import FractionalDerivative
    >>> x = sp.Symbol('x')

    1. Half-derivative ($\alpha = 1/2$) of a power function:
    >>> fd_pow = FractionalDerivative(x**2, x, 0.5)
    >>> fd_pow.doit()
    8*x**(3/2)/(3*sqrt(pi))

    2. Evaluating the memory effect on a constant (f(x) = 1):
    >>> fd_const = FractionalDerivative(1, x, 0.5)
    >>> fd_const.doit()
    1/(sqrt(pi)*sqrt(x))

    3. Exact hypergeometric form for trigonometric functions:
    >>> fd_trig = FractionalDerivative(sp.sin(x), x, 0.5)
    >>> fd_trig.doit()
    2*sqrt(x)*hyper((1,), (3/4, 5/4), -x**2/4)/sqrt(pi)

    4. High-precision numerical evaluation of a resolved closed form:
    >>> expr_eval = FractionalDerivative(sp.exp(x), x, 0.5).subs(x, 1)
    >>> expr_eval.evalf(25)
    2.854887835850994517897617

    5. Numerical evaluation from the Riemann-Liouville definition when no
       symbolic rule is available:
    >>> fd_numeric = FractionalDerivative(sp.sin(x**2), x, 0.5)
    >>> fd_numeric.eval_at(1, 15)
    1.11361910109605

    6. A negative order represents a fractional integral:
    >>> FractionalDerivative(sp.exp(x), x, -1).doit()
    exp(x) - 1

    See Also
    ========
    sympy.core.function.Derivative : Classical integer-order differential operator.
    """
    is_Derivative = True

    def __new__(cls, expr, x, alpha):
        expr = sp.sympify(expr)
        x = sp.sympify(x)
        alpha = sp.sympify(alpha).nsimplify()

        if not isinstance(x, sp.Symbol):
            raise ValueError("The second argument must be a symbol.")

        if alpha.is_number and alpha.is_finite is not True:
            raise ValueError("The operator order must be finite.")

        return sp.Expr.__new__(cls, expr, x, alpha)

    @property
    def expr(self): return self.args[0]
    @property
    def x(self): return self.args[1]
    @property
    def alpha(self): return self.args[2]

    def _eval_subs(self, old, new):
        if old == self.x:
            resolved = self.doit()
            if resolved != self:
                return resolved.subs(old, new)
            return _FractionalDerivativeAt(self, new)
        return None

    def eval_at(self, point, n=15):
        """Numerically evaluate the operator at a positive real point.

        Closed forms are evaluated by SymPy. Otherwise, the implementation
        evaluates the Riemann-Liouville derivative or integral definition.
        """
        point = sp.sympify(point)
        resolved = self.doit()
        if resolved != self:
            return resolved.subs(self.x, point).evalf(n)
        return self._eval_at(point, dps_to_prec(n))

    def _eval_at(self, point, prec):
        dps = prec_to_dps(prec) + 10

        with mpmath.workprec(prec + 32):
            x_value = _real_mpf(point, dps, "The evaluation point")
            alpha_value = _real_mpf(self.alpha, dps, "The operator order")

            if x_value <= 0:
                raise ValueError("The evaluation point must be positive for lower bound 0.")
            def fractional_integral(expression, order):
                numeric_function = sp.lambdify(
                    self.x,
                    expression,
                    modules="mpmath",
                )

                # t = x*(1-u**(1/order)) removes the endpoint power kernel.
                def transformed_integrand(u):
                    t = x_value * (1 - u ** (1 / order))
                    return _mpmath_value(numeric_function(t), dps)

                return (
                    x_value**order
                    / (order * mpmath.gamma(order))
                    * mpmath.quad(transformed_integrand, [0, 1])
                )

            if alpha_value < 0:
                result = fractional_integral(self.expr, -alpha_value)
                return _sympy_float(result, prec)
            if alpha_value == 0:
                value = self.expr.subs(self.x, point)
                return sp.N(value, prec_to_dps(prec))

            n = int(mpmath.ceil(alpha_value))
            beta = n - alpha_value
            if beta == 0:
                value = sp.diff(self.expr, self.x, n).subs(self.x, point)
                return sp.N(value, prec_to_dps(prec))

            # Riemann-Liouville = Caputo + lower-boundary contributions.
            boundary = mpmath.mpmathify(0)
            for k in range(n):
                derivative_at_zero = sp.diff(self.expr, self.x, k).subs(self.x, 0)
                initial_value = _mpmath_value(derivative_at_zero, dps)
                boundary += (
                    initial_value
                    * x_value ** (k - alpha_value)
                    / mpmath.gamma(k + 1 - alpha_value)
                )

            nth_derivative = sp.diff(self.expr, self.x, n)
            result = boundary + fractional_integral(nth_derivative, beta)
            return _sympy_float(result, prec)

    def doit(self, **hints):
        expr = self.expr.doit(**hints)
        x = self.x
        alpha = self.alpha

        if alpha == 0:
            return expr
        if alpha.is_integer is True:
            if alpha.is_nonnegative is True:
                if alpha.is_number:
                    return sp.diff(expr, x, int(alpha))
                return sp.Derivative(expr, (x, alpha), evaluate=False)
            if alpha.is_negative is not True:
                return self
            order = -alpha
            t = sp.Dummy('t', positive=True)
            integral = sp.Integral(
                (x - t)**(order - 1) * expr.xreplace({x: t}),
                (t, 0, x),
            ) / sp.gamma(order)
            if alpha.is_number:
                return sp.simplify(integral.doit(**hints))
            return integral

        # A constant c has D^alpha(c) = c*x^(-alpha)/Gamma(1-alpha).
        if not expr.has(x):
            return expr * x**(-alpha) / sp.gamma(1 - alpha)

        # 1. Linearity: Addition
        if expr.is_Add:
            return sp.Add(*[FractionalDerivative(arg, x, alpha).doit(**hints) for arg in expr.args])

        # 2. Linearity: Multiplication by Constants
        if expr.is_Mul:
            coeff, remaining = expr.as_independent(x)
            if coeff != 1:
                return coeff * FractionalDerivative(remaining, x, alpha).doit(**hints)

        # 3. Power Rule: x^m
        m = sp.Wild('m', exclude=[x])
        match_pow = expr.match(x**m)
        
        if expr == x:
            match_pow = {m: sp.S.One}

        if match_pow:
            m_val = match_pow[m]
            integrable = sp.ask(sp.Q.positive(sp.re(m_val) + 1))
            if integrable is False:
                raise ValueError(
                    "A power x**m must satisfy re(m) > -1 for the "
                    "Riemann-Liouville integral with lower bound 0."
                )
            return (sp.gamma(m_val + 1) / sp.gamma(m_val + 1 - alpha)) * (x**(m_val - alpha))

        # 4. Exponential Rule: exp(b*x)
        b = sp.Wild('b', exclude=[x])
        match_exp = expr.match(sp.exp(b*x))
        if expr == sp.exp(x):
            match_exp = {b: sp.S.One}
            
        if match_exp:
            if alpha.is_integer is not False and alpha.is_negative is not True:
                return self
            b_val = match_exp[b]
            return (x**(-alpha) / sp.gamma(1 - alpha)) * sp.hyper(
                [sp.S.One], [1 - alpha], b_val*x
            )

        # 5. Sine Rule: sin(w*x)
        w = sp.Wild('w', exclude=[x])
        match_sin = expr.match(sp.sin(w*x))
        if expr == sp.sin(x):
            match_sin = {w: sp.S.One}
            
        if match_sin:
            if alpha.is_integer is not False and alpha.is_negative is not True:
                return self
            w_val = match_sin[w]
            arg_hyper = -(w_val**2)*(x**2)/4
            return (w_val * x**(1 - alpha) / sp.gamma(2 - alpha)) * sp.hyper([sp.S.One], [1 - alpha/2, sp.Rational(3,2) - alpha/2], arg_hyper)

        # 6. Cosine Rule: cos(w*x)
        match_cos = expr.match(sp.cos(w*x))
        if expr == sp.cos(x):
            match_cos = {w: sp.S.One}
            
        if match_cos:
            if alpha.is_integer is not False and alpha.is_negative is not True:
                return self
            w_val = match_cos[w]
            arg_hyper = -(w_val**2)*(x**2)/4
            return (x**(-alpha) / sp.gamma(1 - alpha)) * sp.hyper([sp.S.One], [sp.Rational(1,2) - alpha/2, 1 - alpha/2], arg_hyper)

        return self

    def _eval_evalf(self, prec):
        resolved = self.doit()
        if resolved != self:
            return resolved.evalf(prec)
        return self


class _FractionalDerivativeAt(sp.Expr):
    """Internal unevaluated substitution used by numerical fallback."""

    def __new__(cls, derivative, point):
        return sp.Expr.__new__(cls, derivative, sp.sympify(point))

    @property
    def derivative(self):
        return self.args[0]

    @property
    def point(self):
        return self.args[1]

    @property
    def free_symbols(self):
        derivative = self.derivative
        parameters = derivative.expr.free_symbols | derivative.alpha.free_symbols
        return (parameters - {derivative.x}) | self.point.free_symbols

    def doit(self, **hints):
        resolved = self.derivative.doit(**hints)
        if resolved != self.derivative:
            return resolved.subs(self.derivative.x, self.point).doit(**hints)
        return self

    def _eval_evalf(self, prec):
        try:
            return self.derivative._eval_at(self.point, prec)
        except (TypeError, ValueError, ZeroDivisionError):
            return self
