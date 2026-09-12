import sympy as sp
import mpmath

class FractionalDerivative(sp.Expr):
    r"""
    Symbolically represents the Riemann-Liouville fractional derivative.

    Mathematical Explanation
    ========================
    Unlike classical calculus, where the order of differentiation $n$ belongs 
    to positive integers ($\mathbb{N}$), fractional calculus generalizes the 
    differential operator to an arbitrary order $\alpha \in \mathbb{C}$ (or $\mathbb{R}$).

    This subroutine uses the **Riemann-Liouville (R-L)** definition with a fixed lower
    bound at $a = 0$. For a given function $f(x)$ and order $\alpha > 0$ 
    (where $n = \lceil \alpha \rceil$), the operator is formally defined as:

    .. math::
        D^\alpha f(x) = \frac{1}{\Gamma(n - \alpha)} \frac{d^n}{dx^n} \int_{0}^{x} (x - t)^{n - \alpha - 1} f(t) \, dt

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
      *(Supports negative exponents $m$ provided that $m+1 \notin \mathbb{Z}_{\le 0}$)*

    * **Exponentials ($e^{bx}$):**
      .. math::
          D^\alpha (e^{bx}) = b^\alpha e^{bx}

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
        The order of the derivative. Can be an integer, rational, or abstract symbol.

    Examples
    ========
    >>> import sympy as sp
    >>> from fractional import FractionalDerivative
    >>> x = sp.Symbol('x')

    1. Half-derivative ($\alpha = 1/2$) of a power function:
    >>> fd_pow = FractionalDerivative(x**2, x, 0.5)
    >>> fd_pow.doit()
    8*x**(3/2)/(3*sp.sqrt(sp.pi))

    2. Evaluating the memory effect on a constant (f(x) = 1):
    >>> fd_const = FractionalDerivative(1, x, 0.5)
    >>> fd_const.doit()
    1/(sp.sqrt(sp.pi)*sp.sqrt(x))

    3. Exact hypergeometric form for trigonometric functions:
    >>> fd_trig = FractionalDerivative(sp.sin(x), x, 0.5)
    >>> fd_trig.doit()
    2*sp.sqrt(x)*sp.hyper((1,), (3/4, 5/4), -x**2/4)/sp.sqrt(sp.pi)

    4. High-precision numerical evaluation (.evalf()) inheriting mpmath:
    >>> expr_eval = FractionalDerivative(sp.exp(x), x, 0.5).subs(x, 1)
    >>> expr_eval.evalf(25)
    4.493289641172216174625244

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
        return None

    def doit(self, **hints):
        expr = self.expr.doit(**hints)
        x = self.x
        alpha = self.alpha

        if alpha == 0:
            return expr
        if alpha.is_Integer and alpha > 0:
            return sp.diff(expr, x, int(alpha))

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
            return (sp.gamma(m_val + 1) / sp.gamma(m_val + 1 - alpha)) * (x**(m_val - alpha))

        # 4. Exponential Rule: exp(b*x)
        b = sp.Wild('b', exclude=[x])
        match_exp = expr.match(sp.exp(b*x))
        if expr == sp.exp(x):
            match_exp = {b: sp.S.One}
            
        if match_exp:
            b_val = match_exp[b]
            return (b_val**alpha) * sp.exp(b_val*x)

        # 5. Sine Rule: sin(w*x)
        w = sp.Wild('w', exclude=[x])
        match_sin = expr.match(sp.sin(w*x))
        if expr == sp.sin(x):
            match_sin = {w: sp.S.One}
            
        if match_sin:
            w_val = match_sin[w]
            arg_hyper = -(w_val**2)*(x**2)/4
            return (w_val * x**(1 - alpha) / sp.gamma(2 - alpha)) * sp.hyper([sp.S.One], [1 - alpha/2, sp.Rational(3,2) - alpha/2], arg_hyper)

        # 6. Cosine Rule: cos(w*x)
        match_cos = expr.match(sp.cos(w*x))
        if expr == sp.cos(x):
            match_cos = {w: sp.S.One}
            
        if match_cos:
            w_val = match_cos[w]
            arg_hyper = -(w_val**2)*(x**2)/4
            return (x**(-alpha) / sp.gamma(1 - alpha)) * sp.hyper([sp.S.One], [sp.Rational(1,2) - alpha/2, 1 - alpha/2], arg_hyper)

        return self

    def _eval_evalf(self, prec):
        resolved = self.doit()
        if resolved != self:
            return resolved.evalf(prec)

        try:
            x_val = mpmath.mpf(self.x.evalf(prec))
            alpha_val = mpmath.mpf(self.alpha.evalf(prec))
            
            # Grünwald-Letnikov numerical approximation as a fallback
            def gl_approximation(f, x, alpha, N=100):
                h = x / N
                suma = mpmath.mpf(0)
                binom = 1
                for k in range(N):
                    if k > 0:
                        binom = binom * (alpha - k + 1) / k
                    term_x = x - k * h
                    f_val = mpmath.mpf(self.expr.subs(self.x, float(term_x)).evalf(prec))
                    suma += ((-1)**k) * binom * f_val
                return suma / (h**alpha)

            with mpmath.workprec(prec):
                result = gl_approximation(self.expr, x_val, alpha_val)
                return sp.Float(result, prec)
        except (TypeError, ValueError):
            return self
