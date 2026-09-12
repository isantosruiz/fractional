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

    This subroutine uses the **Riemann-Liouville (R-L)** definition with lower
    bound $x_0$, which defaults to zero. For a given function $f(x)$ and order $\alpha > 0$
    (where $n = \lceil \alpha \rceil$), the operator is formally defined as:

    .. math::
        D_{x_0+}^\alpha f(x) = \frac{1}{\Gamma(n - \alpha)} \frac{d^n}{dx^n} \int_{x_0}^{x} (x - t)^{n - \alpha - 1} f(t) \, dt

    For a negative real order $\alpha=-\beta$, with $\beta>0$, the same
    operator represents the Riemann-Liouville fractional integral:

    .. math::
        D_{x_0+}^{-\beta} f(x) = I_{x_0+}^\beta f(x)
        = \frac{1}{\Gamma(\beta)} \int_{x_0}^x (x-t)^{\beta-1}f(t)\,dt

    Where $\Gamma(z)$ is the Euler Gamma function, which extends factorials to real numbers.

    Operator Properties:
    --------------------
    1. **Memory Effect (Non-Locality):** The derivative at point $x$ does not depend 
       solely on the local neighborhood of $x$, but on the entire history of the function 
       over the closed interval $[x_0, x]$.
    2. **Linearity:** It satisfies $D^\alpha [c_1 f(x) + c_2 g(x)] = c_1 D^\alpha f(x) + c_2 D^\alpha g(x)$.
    3. **Derivative of a Constant:** Unlike classical calculus, the fractional derivative 
       of a constant **is not zero** under the Riemann-Liouville framework. 
       For $f(x) = 1$, the operator yields
       $D_{x_0+}^\alpha(1) = \frac{(x-x_0)^{-\alpha}}{\Gamma(1-\alpha)}$.

    Implemented Closed Forms (Symbolic Mapping)
    ===========================================
    To guarantee analytical exact solutions inside SymPy, the ``.doit()`` method uses
    pattern matching based on the following foundational rules:

    * **Shifted powers ($(x-x_0)^m$):**
      .. math::
          D_{x_0+}^\alpha ((x-x_0)^m) = \frac{\Gamma(m+1)}{\Gamma(m + 1 - \alpha)} (x-x_0)^{m - \alpha}
      The classical integral additionally requires convergence at the lower
      bound (for example, $\operatorname{Re}(m)>-1$).

    * **Exponentials ($e^{bx}$):**
      .. math::
          D_{x_0+}^\alpha (e^{bx}) = \frac{e^{b x_0}(x-x_0)^{-\alpha}}{\Gamma(1-\alpha)}
          \,{}_1F_1\left(1; 1-\alpha; b(x-x_0)\right)
      for non-integer $\alpha$. The boundary contribution at $x=x_0$ is essential;
      consequently, $b^\alpha e^{bx}$ is not the Riemann-Liouville derivative
      with the finite lower bound used by this class.

    * **Trigonometric Functions ($\sin(wx)$ and $\cos(wx)$):**
      Analytical solutions use the corresponding zero-based $_1F_2$ formulas
      after translating the variable by $x_0$.

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
    x0 : Expr or number, optional
        The lower terminal of the Riemann-Liouville operator. It must not depend
        on ``x`` and defaults to zero.

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

    7. The lower terminal can be customized:
    >>> FractionalDerivative(sp.exp(x), x, -1, x0=2).doit()
    exp(x) - exp(2)

    See Also
    ========
    sympy.core.function.Derivative : Classical integer-order differential operator.
    """
    is_Derivative = True

    def __new__(cls, expr, x, alpha, x0=0):
        expr = sp.sympify(expr)
        x = sp.sympify(x)
        alpha = sp.sympify(alpha).nsimplify()
        x0 = sp.sympify(x0).nsimplify()

        if not isinstance(x, sp.Symbol):
            raise ValueError("The second argument must be a symbol.")

        if alpha.is_number and alpha.is_finite is not True:
            raise ValueError("The operator order must be finite.")

        if x0.has(x):
            raise ValueError("The lower bound x0 must not depend on x.")
        if (
            x0.is_real is False
            or x0.is_finite is False
            or (
                x0.is_number
                and (x0.is_real is not True or x0.is_finite is not True)
            )
        ):
            raise ValueError("The lower bound x0 must be finite and real.")

        # Keep the original three-argument structure when x0 is zero so that
        # existing expressions retain their representation and hash.
        if x0 == 0:
            return sp.Expr.__new__(cls, expr, x, alpha)
        return sp.Expr.__new__(cls, expr, x, alpha, x0)

    @property
    def expr(self): return self.args[0]
    @property
    def x(self): return self.args[1]
    @property
    def alpha(self): return self.args[2]
    @property
    def x0(self): return self.args[3] if len(self.args) == 4 else sp.S.Zero

    def _eval_subs(self, old, new):
        if old == self.x:
            resolved = self.doit()
            if resolved != self:
                return resolved.subs(old, new)
            return _FractionalDerivativeAt(self, new)
        return None

    def eval_at(self, point, n=15):
        """Numerically evaluate the operator at a real point greater than ``x0``.

        Closed forms are evaluated by SymPy. Otherwise, the implementation
        evaluates the Riemann-Liouville derivative or integral definition.
        """
        point = sp.sympify(point)
        validation_dps = n + 10
        point_value = _real_mpf(point, validation_dps, "The evaluation point")
        x0_value = _real_mpf(self.x0, validation_dps, "The lower bound x0")
        if point_value <= x0_value:
            raise ValueError("The evaluation point must be greater than x0.")

        resolved = self.doit()
        if resolved != self:
            return resolved.subs(self.x, point).evalf(n)
        return self._eval_at(point, dps_to_prec(n))

    def _eval_at(self, point, prec):
        dps = prec_to_dps(prec) + 10

        with mpmath.workprec(prec + 32):
            x_value = _real_mpf(point, dps, "The evaluation point")
            x0_value = _real_mpf(self.x0, dps, "The lower bound x0")
            alpha_value = _real_mpf(self.alpha, dps, "The operator order")

            if x_value <= x0_value:
                raise ValueError("The evaluation point must be greater than x0.")

            interval = x_value - x0_value

            def fractional_integral(expression, order):
                numeric_function = sp.lambdify(
                    self.x,
                    expression,
                    modules="mpmath",
                )

                # t = x - (x-x0)*u**(1/order) removes the endpoint power kernel.
                def transformed_integrand(u):
                    t = x_value - interval * u ** (1 / order)
                    return _mpmath_value(numeric_function(t), dps)

                return (
                    interval**order
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
                derivative_at_x0 = sp.diff(self.expr, self.x, k).subs(
                    self.x,
                    self.x0,
                )
                initial_value = _mpmath_value(derivative_at_x0, dps)
                boundary += (
                    initial_value
                    * interval ** (k - alpha_value)
                    / mpmath.gamma(k + 1 - alpha_value)
                )

            nth_derivative = sp.diff(self.expr, self.x, n)
            result = boundary + fractional_integral(nth_derivative, beta)
            return _sympy_float(result, prec)

    def doit(self, **hints):
        expr = self.expr.doit(**hints)
        x = self.x
        alpha = self.alpha
        x0 = self.x0
        distance = x - x0

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
                (t, x0, x),
            ) / sp.gamma(order)
            if alpha.is_number:
                return sp.simplify(integral.doit(**hints))
            return integral

        # A constant c has D^alpha(c) = c*(x-x0)^(-alpha)/Gamma(1-alpha).
        if not expr.has(x):
            return expr * distance**(-alpha) / sp.gamma(1 - alpha)

        # 1. Linearity: Addition
        if expr.is_Add:
            return sp.Add(*[
                FractionalDerivative(arg, x, alpha, x0).doit(**hints)
                for arg in expr.args
            ])

        # 2. Linearity: Multiplication by Constants
        if expr.is_Mul:
            coeff, remaining = expr.as_independent(x)
            if coeff != 1:
                return coeff * FractionalDerivative(
                    remaining,
                    x,
                    alpha,
                    x0,
                ).doit(**hints)

        # Translate ordinary polynomials into powers of (x-x0). This preserves
        # the exact power rule for a custom lower terminal.
        if x0 != 0 and expr.is_polynomial(x):
            shifted_x = sp.Dummy('shifted_x', positive=True)
            shifted_expr = sp.expand(expr.xreplace({x: shifted_x + x0}))
            shifted_operator = FractionalDerivative(shifted_expr, shifted_x, alpha)
            shifted_result = shifted_operator.doit(**hints)
            if not shifted_result.has(FractionalDerivative):
                return sp.simplify(shifted_result.xreplace({shifted_x: distance}))

        # 3. Power Rule: (x-x0)^m
        m = sp.Wild('m', exclude=[x])
        match_pow = expr.match(distance**m)
        
        if expr == distance:
            match_pow = {m: sp.S.One}

        if match_pow:
            m_val = match_pow[m]
            integrable = sp.ask(sp.Q.positive(sp.re(m_val) + 1))
            if integrable is False:
                raise ValueError(
                    "A shifted power (x-x0)**m must satisfy re(m) > -1 for the "
                    "Riemann-Liouville integral at its lower bound."
                )
            return (
                sp.gamma(m_val + 1)
                / sp.gamma(m_val + 1 - alpha)
                * distance**(m_val - alpha)
            )

        # 4. Exponential Rule: exp(b*x)
        b = sp.Wild('b', exclude=[x])
        match_exp = expr.match(sp.exp(b*x))
        if expr == sp.exp(x):
            match_exp = {b: sp.S.One}
            
        if match_exp:
            if alpha.is_integer is not False and alpha.is_negative is not True:
                return self
            b_val = match_exp[b]
            return (
                sp.exp(b_val*x0)
                * distance**(-alpha)
                / sp.gamma(1 - alpha)
                * sp.hyper([sp.S.One], [1 - alpha], b_val*distance)
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
            arg_hyper = -(w_val**2)*(distance**2)/4
            shifted_sine = (
                w_val * distance**(1 - alpha)
                / sp.gamma(2 - alpha)
                * sp.hyper(
                    [sp.S.One],
                    [1 - alpha/2, sp.Rational(3, 2) - alpha/2],
                    arg_hyper,
                )
            )
            shifted_cosine = (
                distance**(-alpha)
                / sp.gamma(1 - alpha)
                * sp.hyper(
                    [sp.S.One],
                    [sp.Rational(1, 2) - alpha/2, 1 - alpha/2],
                    arg_hyper,
                )
            )
            return (
                sp.cos(w_val*x0) * shifted_sine
                + sp.sin(w_val*x0) * shifted_cosine
            )

        # 6. Cosine Rule: cos(w*x)
        match_cos = expr.match(sp.cos(w*x))
        if expr == sp.cos(x):
            match_cos = {w: sp.S.One}
            
        if match_cos:
            if alpha.is_integer is not False and alpha.is_negative is not True:
                return self
            w_val = match_cos[w]
            arg_hyper = -(w_val**2)*(distance**2)/4
            shifted_sine = (
                w_val * distance**(1 - alpha)
                / sp.gamma(2 - alpha)
                * sp.hyper(
                    [sp.S.One],
                    [1 - alpha/2, sp.Rational(3, 2) - alpha/2],
                    arg_hyper,
                )
            )
            shifted_cosine = (
                distance**(-alpha)
                / sp.gamma(1 - alpha)
                * sp.hyper(
                    [sp.S.One],
                    [sp.Rational(1, 2) - alpha/2, 1 - alpha/2],
                    arg_hyper,
                )
            )
            return (
                sp.cos(w_val*x0) * shifted_cosine
                - sp.sin(w_val*x0) * shifted_sine
            )

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
        parameters = (
            derivative.expr.free_symbols
            | derivative.alpha.free_symbols
            | derivative.x0.free_symbols
        )
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
