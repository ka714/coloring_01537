#!/usr/bin/env python3
"""Verify conditions (i)-(iv) of the paper using only the Python standard library.
The parameter order is (a,b,mu,c,kappa,theta). JSON children.rho denotes rho_prime.
"""
import json
import multiprocessing as mp
from pathlib import Path
import sys
from fractions import Fraction as F
from math import isqrt
BITS = 72
SCALE = 1 << BITS

def rat(x):
    if isinstance(x, bool) or not isinstance(x, (int, F)):
        raise TypeError('expected an exact rational')
    return F(x)

def ceildiv(a, b):
    return -(-a // b)

# Store [lo/2**72, hi/2**72] with integer endpoints and round outwards.
class Interval:
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi):
        if type(lo) is not int or type(hi) is not int or lo > hi:
            raise ValueError('invalid interval')
        (self.lo, self.hi) = (lo, hi)

    @classmethod
    def exact(cls, x):
        x = rat(x)
        n = x.numerator * SCALE
        d = x.denominator
        return cls(n // d, ceildiv(n, d))

    @classmethod
    def box(cls, a, b):
        (a, b) = (rat(a), rat(b))
        if a > b:
            raise ValueError('reversed box')
        return cls(a.numerator * SCALE // a.denominator, ceildiv(b.numerator * SCALE, b.denominator))

    def __add__(self, other):
        if isinstance(other, Derivatives):
            return NotImplemented
        z = as_interval(other)
        return Interval(self.lo + z.lo, self.hi + z.hi)
    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -as_any(other)

    def __rsub__(self, other):
        return as_any(other) + -self

    def __mul__(self, other):
        if isinstance(other, Derivatives):
            return NotImplemented
        z = as_interval(other)
        v = (self.lo * z.lo, self.lo * z.hi, self.hi * z.lo, self.hi * z.hi)
        return Interval(min(v) // SCALE, ceildiv(max(v), SCALE))
    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Derivatives):
            return NotImplemented
        z = as_interval(other)
        if z.lo <= 0 <= z.hi:
            raise ArithmeticError('division interval contains zero')
        pairs = ((a * SCALE, b) for a in (self.lo, self.hi) for b in (z.lo, z.hi))
        vals = [(a // b, ceildiv(a, b)) for (a, b) in pairs]
        return Interval(min((a for (a, b) in vals)), max((b for (a, b) in vals)))

    def __rtruediv__(self, other):
        return as_any(other) / self

    def __pow__(self, n):
        if n == 2:
            v = (self.lo * self.lo, self.hi * self.hi)
            return Interval(0 if self.lo <= 0 <= self.hi else min(v) // SCALE, ceildiv(max(v), SCALE))
        if n == 3:
            return self * self ** 2
        raise ValueError('only square and cube are used')

    def sqrt(self):
        if self.lo < 0:
            raise ArithmeticError('negative square-root interval')
        a = isqrt(self.lo * SCALE)
        b = isqrt(self.hi * SCALE)
        return Interval(a, b + (b * b < self.hi * SCALE))

    def absmax(self):
        return F(max(abs(self.lo), abs(self.hi)), SCALE)

    def lower(self):
        return F(self.lo, SCALE)

def as_any(x):
    return x if isinstance(x, (Derivatives, Interval)) else Interval.exact(x)

def as_interval(x):
    return x if isinstance(x, Interval) else Interval.exact(x)
ZERO = Interval.exact(0)
ONE = Interval.exact(1)

# v encloses the value, g[i] the first partial, and h[3*i+j] the second partial.
# The variable order is (A, rho, rho_prime).
class Derivatives:
    """Value, gradient and Hessian enclosures in three real variables."""
    __slots__ = ('v', 'g', 'h')

    def __init__(self, v, g=None, h=None):
        self.v = as_interval(v)
        self.g = g if g is not None else (ZERO, ZERO, ZERO)
        self.h = h if h is not None else (ZERO,) * 9

    @classmethod
    def variable(cls, v, axis):
        if axis not in (0, 1, 2):
            raise ValueError('bad derivative axis')
        return cls(v, tuple((ONE if i == axis else ZERO for i in range(3))))

    def __add__(self, other):
        z = as_derivatives(other)
        return Derivatives(self.v + z.v,
            tuple((a + b for (a, b) in zip(self.g, z.g))),
            tuple((a + b for (a, b) in zip(self.h, z.h))))
    __radd__ = __add__

    def __neg__(self):
        return Derivatives(-self.v, tuple((-a for a in self.g)), tuple((-a for a in self.h)))

    def __sub__(self, other):
        return self + -as_derivatives(other)

    def __rsub__(self, other):
        return as_derivatives(other) + -self

    def __mul__(self, other):
        z = as_derivatives(other)
        g = tuple((self.g[i] * z.v + self.v * z.g[i] for i in range(3)))
        h = tuple(
            self.h[3 * i + j] * z.v
            + self.g[i] * z.g[j] + self.g[j] * z.g[i]
            + self.v * z.h[3 * i + j]
            for i in range(3) for j in range(3)
        )
        return Derivatives(self.v * z.v, g, h)
    __rmul__ = __mul__

    def unary(self, v, d1, d2):
        return Derivatives(v,
            tuple((d1 * a for a in self.g)),
            tuple((d1 * self.h[3 * i + j] + d2 * self.g[i] * self.g[j] for i in range(3) for j in range(3))))

    def reciprocal(self):
        inv = ONE / self.v
        return self.unary(inv, -inv ** 2, 2 * inv ** 3)

    def __truediv__(self, other):
        return self * as_derivatives(other).reciprocal()

    def __rtruediv__(self, other):
        return as_derivatives(other) * self.reciprocal()

    def __pow__(self, n):
        if n != 2:
            raise ValueError('only square is used')
        return self.unary(self.v ** 2, 2 * self.v, 2)

    def sqrt(self):
        v = self.v.sqrt()
        return self.unary(v, ONE / (2 * v), -ONE / (4 * self.v * v))

def as_derivatives(x):
    return x if isinstance(x, Derivatives) else Derivatives(x)

def sqrt(x):
    return as_any(x).sqrt()

def min_quadratic(a, g, r):
    """min of a*u^2-g*u, for 0<=u<=r, where g,r>=0."""
    if not r:
        return F(0)
    if a > 0 and g <= 2 * a * r:
        return -g * g / (4 * a)
    return a * r * r - g * r

def taylor_bound(center, whole, radii):
    """Taylor lower bound; all Hessian entries enclose the full closed box."""
    center = as_derivatives(center)
    whole = as_derivatives(whole)
    f = center.v.lower()
    g = [z.absmax() for z in center.g]
    a = [whole.h[3 * i + i].lower() / 2 for i in range(3)]
    # Bound each mixed term by two diagonal terms using AM-GM (README, Section 11).
    for i in range(3):
        for j in range(i + 1, 3):
            c = max(whole.h[3 * i + j].absmax(), whole.h[3 * j + i].absmax())
            if c and radii[i] and radii[j]:
                a[i] -= c * radii[j] / (2 * radii[i])
                a[j] -= c * radii[i] / (2 * radii[j])
    return f + sum((min_quadratic(a[i], g[i], radii[i]) for i in range(3)))
(BETA, ETA) = (F('0.692601'), F('0.000001'))
D = 2 * BETA / (1 - BETA)
D_A = D + 3 * ETA
(A_MIN, A_SPLIT) = (F('1.968'), F('2.15'))
(A_0, SQRT3_OVER2) = (sqrt(D_A + F(1, 4)), sqrt(3) / 2)
(ZETA_1_ZERO, ZETA_2_ZERO, ZETA_3_ZERO, RHO_Q_C_START) = (F(-1, 40), F(1, 50), F(1, 4), F(7, 10))
(RHO_MINUS, RHO_PLUS) = ((4 - D_A) / (2 * D_A + 16), (D_A - 4) / (4 * D_A + 8))
(A_B_MU_MARGIN, C_KAPPA_MARGIN, CONDITION_III_MARGIN) = (F('1e-12'), F('1e-20'), F('1e-18'))

def f_A(A, rho, indices):
    """The paper's f_A(rho); indices selects the zeta_i terms to include.
    Use (1,), (), (2,), or (2,3). Terms vanishing at an endpoint may be included.
    """
    value = (F('.52') * A + F('1.426') + F('2.46') * (A - 2) ** 2) * sqrt(1 - rho * rho)
    if indices == (1,):
        return value - F('.714') * (ZETA_1_ZERO - rho) + F('.826') * (ZETA_1_ZERO - rho) ** 2
    if indices == ():
        return value
    if indices in ((2,), (2, 3)):
        value += -F('.80') * (rho - ZETA_2_ZERO) + F('.45') * (rho - ZETA_2_ZERO) ** 2
        if indices == (2, 3):
            value += F('.39') * (rho - ZETA_3_ZERO) + F('.65') * (rho - ZETA_3_ZERO) ** 2
        return value
    raise ValueError('invalid zeta indices')

def C_Da(rho, case):
    """The paper's C_{D_a}(rho); case selects a formula in README, Section 3."""
    if case == 1:
        return (1 + sqrt(1 + 6 * D_A * (D_A + 1) * (1 - 2 * rho) / (1 - 4 * rho))) / 2
    if case == 2:
        return D_A + 2
    if case == 3:
        return (1 + sqrt(1 + 6 * D_A * (D_A + 1) / (1 + 2 * rho))) / 2
    if case == 4:
        return D_A + 1
    raise ValueError('invalid C case')

# Each row is (rho lower endpoint, upper endpoint, zeta_i index set, C case).
# None means that f_A is not evaluated; comparison uses q_c.
RHO_INTERVALS = [
    (F(-1, 2), ZETA_1_ZERO, (1,), 1),
    (ZETA_1_ZERO, RHO_MINUS, (), 1),
    (RHO_MINUS, F(0), (), 2),
    (F(0), RHO_PLUS, (), 2),
    (RHO_PLUS, ZETA_2_ZERO, (), 3),
    (ZETA_2_ZERO, ZETA_3_ZERO, (2,), 3),
    (ZETA_3_ZERO, RHO_Q_C_START, (2, 3), 4),
    (RHO_Q_C_START, F(1), None, 4),
]
DOMAINS = {
    f'case1.{j}': ('case1', f_indices, C_case, ((A_MIN, A_SPLIT), (lo, hi)))
    for j, (lo, hi, f_indices, C_case) in enumerate(RHO_INTERVALS) if f_indices is not None
}
DOMAINS['case2'] = ('case2', (1,), 1, ((A_MIN, A_SPLIT), (F(-1, 2), F(-1, 2))))
DOMAINS['case2_A0'] = ('case2_A0', (1,), 1, ((A_SPLIT, A_SPLIT), (F(-1, 2), F(-1, 2))))

def f_A_or_A(mode, f_indices, A, rho):
    """Return f_A(A,rho) in Case 1 and A in Case 2.
    In Case 1, divide this value by sqrt(1-rho^2) to obtain the paper's L.
    """
    return f_A(A, rho, f_indices) if mode == 'case1' else A

def condition_iii(mode, f_indices, C_case, A, rho, scaled_parameters):
    """The left-hand side of (iii) minus its right-hand side.
    scaled_parameters[3]+scaled_parameters[5] = mu+theta*(mu+kappa).
    """
    f = f_A_or_A(mode, f_indices, A, rho)
    if mode == 'case1':
        J = C_Da(rho, C_case) + F(1, 4)
        L_squared = f * f / (1 - rho * rho)
        Lambda = 1
    else:
        J = D_A + F(1, 4)
        L_squared = f * f
        Lambda = F('3.999')
    required = J - L_squared
    return (Lambda * (scaled_parameters[3] + scaled_parameters[5]) ** 2 - required - CONDITION_III_MARGIN,)

def f_A_above_q_c(f_indices, C_case, A, rho):
    """Return f_A and f_A^2-q_c^2; positivity of both proves f_A > q_c."""
    f = f_A(A, rho, f_indices)
    return (f, f * f - (1 - rho * rho) * (C_Da(rho, C_case) + F(1, 4)))

def condition_iv(
    mode,
    f_indices,
    child_f_indices,
    child_C_case,
    A,
    rho,
    rho_prime,
    scaled_parameters,
    sigma,
    target,
):
    """Expressions whose positivity proves (iv).
    Replace the absolute value by sigma times its argument and subtract 3e-6.
    Compare with f_A or q_c; for q_c, return both the value and the squared gap.
    """
    (theta, a_scaled, b_scaled, _, c_scaled, _) = scaled_parameters
    one_minus_rho_squared = 1 - rho * rho
    denominator = one_minus_rho_squared if mode == 'case1' else sqrt(one_minus_rho_squared)
    first_term = f_A_or_A(mode, f_indices, A, rho) * (F(1, 2) + rho * rho_prime) / denominator
    value = (
        first_term
        + (-b_scaled + sigma * c_scaled) * (rho_prime + rho / 2)
        / sqrt(one_minus_rho_squared)
        + F(3, 2) * a_scaled
        - SQRT3_OVER2 * theta * (A_0 if mode == 'case2_A0' else A)
        - F('.000003')
    )
    if target == 'f_A':
        return (value - f_A(A, rho_prime, child_f_indices),)
    return (value, value * value - (1 - rho_prime * rho_prime) * (C_Da(rho_prime, child_C_case) + F(1, 4)))

def bounds(fun, box, rule):
    """Return a rigorous lower bound for each expression on an (A,rho,rho_prime) box."""
    box = tuple(box) + ((F(0), F(0)),) * (3 - len(box))
    ranges = [Interval.box(lo, hi) for (lo, hi) in box]
    if rule == 'interval':
        return [as_derivatives(x).v.lower() for x in fun(*ranges)]
    if rule != 'taylor':
        raise ValueError('invalid proof rule')
    centers = [(lo + hi) / 2 for (lo, hi) in box]
    radii = [(hi - lo) / 2 for (lo, hi) in box]
    whole = fun(*[Derivatives.variable(x, i) for (i, x) in enumerate(ranges)])
    center = fun(*[Derivatives.variable(Interval.exact(x), i) for (i, x) in enumerate(centers)])
    return [taylor_bound(x, y, radii) for (x, y) in zip(center, whole)]

def need(condition, message):
    if not condition:
        raise ValueError(message)

def rational(value):
    need(isinstance(value, str), 'expected a rational string')
    return F(value)

def interval(values):
    need(isinstance(values, list) and len(values) == 2, 'expected two interval endpoints')
    (lo, hi) = map(rational, values)
    need(lo <= hi, 'reversed interval')
    return (lo, hi)

def zeta_indices(value):
    """Read the JSON index set; null means that f_A is not evaluated."""
    if value is None:
        return None
    need(type(value) is list and all(type(i) is int for i in value), 'invalid zeta indices')
    indices = tuple(value)
    need(indices in ((1,), (), (2,), (2, 3)), 'invalid zeta indices')
    return indices

def rectangle(row):
    need(type(row) is dict and set(row) == {'root', 'rect', 'parameter', 'rule', 'children'},
        'invalid rectangle fields')
    need(row['root'] in DOMAINS, 'unknown root')
    rect = row['rect']
    need(type(rect) is dict and set(rect) == {'A', 'rho'}, 'invalid rect fields')
    need(isinstance(row['children'], list), 'expected child list')
    return (interval(rect['A']), interval(rect['rho']))

def cover_interval(domain, parts):
    """Check coverage by closed intervals, with no gaps or interior overlaps."""
    (lo, hi) = domain
    if lo == hi:
        need(parts == [domain], 'point requires exactly one matching interval')
        return
    edge = lo
    for (left, right) in sorted(parts):
        need(lo <= left < right <= hi and left == edge, 'interval gap, overlap, or invalid dimension')
        edge = right
    need(edge == hi, 'incomplete interval coverage')

def cover_rectangles(domain, boxes):
    """Check rho coverage between successive A endpoints; Case 2 is a line or point."""
    need(bool(boxes), 'missing rectangles')
    for box in boxes:
        for ((lo, hi), (left, right)) in zip(domain, box):
            need(lo == hi and left == right == lo or lo <= left < right <= hi,
                'rectangle outside domain or wrong dimension')
    if domain[1][0] == domain[1][1]:
        cover_interval(domain[0], [box[0] for box in boxes])
        return
    (starts, ends) = ({}, {})
    for (i, box) in enumerate(boxes):
        starts.setdefault(box[0][0], []).append(i)
        ends.setdefault(box[0][1], []).append(i)
    edges = sorted(set(starts) | set(ends) | set(domain[0]))
    active = set()
    for edge in edges[:-1]:
        active.difference_update(ends.get(edge, ()))
        active.update(starts.get(edge, ()))
        cover_interval(domain[1], [boxes[i][1] for i in active])

def positive(fun, box, rule):
    values = bounds(fun, box, rule)
    need(bool(values) and all((x > 0 for x in values)), 'nonpositive inequality bound')

def check_rectangle(job):
    """Check (i)-(iv) with one fixed tuple, or exclude a Case 1 rectangle."""
    index, row = job
    try:
        box = rectangle(row)
        mode, f_indices, C_case, _ = DOMAINS[row['root']]
        if row['parameter'] is None:
            need(mode == 'case1' and row['children'] == [], 'invalid f_A > q_c rectangle')
            positive(
                lambda A, rho, rho_prime: f_A_above_q_c(f_indices, C_case, A, rho),
                box, row['rule'],
            )
            return

        # Condition (i): signs. Condition (ii): the two norm bounds.
        values = row['parameter']
        need(isinstance(values, list) and len(values) == 6, 'expected six parameters')
        a, b, mu, c, kappa, theta = map(rational, values)
        need(
            a >= 0 and mu >= 0 and c >= 0 and kappa >= 0 and 0 < theta < 14,
            'parameter sign or theta bound',
        )
        need(
            a * a + b * b + mu * mu <= 1 - A_B_MU_MARGIN
            and c * c + kappa * kappa <= F(1, 4) - C_KAPPA_MARGIN,
            'parameter norm bound',
        )
        scaled_parameters = (
            theta, (1 + theta) * a, (1 + theta) * b,
            (1 + theta) * mu, theta * c, theta * kappa,
        )

        # Check condition (iii) and L > 0 on the entire parent rectangle.
        positive(
            lambda A, rho, rho_prime: condition_iii(
                mode, f_indices, C_case, A, rho, scaled_parameters,
            ),
            box, row['rule'],
        )
        need(
            any(
                all(x > 0 for x in bounds(
                    lambda A, rho, rho_prime: (f_A_or_A(mode, f_indices, A, rho),),
                    box, rule,
                ))
                for rule in ('interval', 'taylor')
            ),
            'nonpositive L',
        )

        # Split the required rho_prime range at the formula boundaries.
        top = min(F(1), F(1, 2) - box[1][0])
        domains = [
            (lo, min(hi, top), child_f_indices, child_C_case)
            for lo, hi, child_f_indices, child_C_case in RHO_INTERVALS if lo <= top
        ]
        groups = [[] for _ in domains]
        for child in row['children']:
            need(
                type(child) is dict and set(child) == {
                    'rho', 'f_indices', 'C_case', 'sign', 'target', 'rule',
                },
                'invalid child fields',
            )
            left, right = interval(child['rho'])
            child_f_indices, child_C_case, sign, target = (
                child[key]
                for key in ('f_indices', 'C_case', 'sign', 'target')
            )
            child_f_indices = zeta_indices(child_f_indices)
            need(type(child_C_case) is int, 'C case must be an integer')
            need(type(sign) is int and sign in (-1, 1), 'invalid sign')
            need(
                target in ('q_c', 'f_A')
                and (target == 'q_c' or (mode != 'case2_A0' and child_f_indices is not None)),
                'invalid target',
            )
            matching = [
                j for j, (lo, hi, p, c) in enumerate(domains)
                if (child_f_indices, child_C_case) == (p, c)
                and (lo == hi and left == right == lo or lo <= left < right <= hi)
            ]
            need(len(matching) == 1, 'rho_prime interval outside its formula domain')
            groups[matching[0]].append((left, right))

            # Check condition (iv) on the parent rectangle times this rho_prime interval.
            positive(
                lambda A, rho, rho_prime: condition_iv(
                    mode, f_indices, child_f_indices, child_C_case,
                    A, rho, rho_prime, scaled_parameters, sign, target,
                ),
                box + ((left, right),), child['rule'],
            )
        # Check that these intervals cover every required rho_prime.
        for (lo, hi, child_f_indices, child_C_case), parts in zip(domains, groups):
            cover_interval((lo, hi), parts)
    except (ValueError, TypeError, KeyError, ArithmeticError) as error:
        raise ValueError(f'rectangle {index}: {error}') from error

def check_f_gt_q_c(rows):
    """Check f_A > q_c at A=2.15 and complete coverage of the required intervals."""
    need(isinstance(rows, list), 'expected f_A > q_c list')
    groups = [[] for (lo, hi, f_indices, C_case) in RHO_INTERVALS if f_indices is not None]
    for row in rows:
        need(type(row) is dict and set(row) == {'rho', 'f_indices', 'C_case', 'rule'},
            'invalid f_A > q_c fields')
        (left, right) = interval(row['rho'])
        (f_indices, C_case) = (row['f_indices'], row['C_case'])
        f_indices = zeta_indices(f_indices)
        need(type(C_case) is int, 'C case must be an integer')
        matching = [
            j for j, (lo, hi, p, c) in enumerate(RHO_INTERVALS)
            if p is not None and (f_indices, C_case) == (p, c)
            and lo <= left < right <= hi
        ]
        need(len(matching) == 1, 'f_A > q_c outside its branch domain')
        groups[matching[0]].append((left, right))
        positive(lambda A, rho, rho_prime: f_A_above_q_c(f_indices, C_case, A, rho),
            ((A_SPLIT, A_SPLIT), (left, right)),
            row['rule'])
    for ((lo, hi, f_indices, C_case), parts) in zip(RHO_INTERVALS, groups):
        cover_interval((lo, hi), parts)

def unique_object(pairs):
    out = {}
    for (key, value) in pairs:
        need(key not in out, 'duplicate JSON key: ' + key)
        out[key] = value
    return out

def reject_float(value):
    raise ValueError('floating-point JSON numbers are not permitted')

def main():
    path = Path(__file__).with_name('certificate.json')
    data = json.loads(path.read_text(),
        parse_float=reject_float,
        parse_constant=reject_float,
        object_pairs_hook=unique_object)
    need(type(data) is dict and set(data) == {'rectangles', 'f_gt_q_c'}, 'invalid certificate fields')
    rows = data['rectangles']
    need(isinstance(rows, list), 'expected rectangle list')
    groups = {name: [] for name in DOMAINS}
    for row in rows:
        box = rectangle(row)
        groups[row['root']].append(box)
        if row['root'] == 'case2_A0':
            need(row['parameter'] is not None, 'case2_A0 requires a fixed tuple')
    for (name, (mode, f_indices, C_case, domain)) in DOMAINS.items():
        cover_rectangles(domain, groups[name])
    check_f_gt_q_c(data['f_gt_q_c'])
    print('Coverage and f_A > q_c: PASS', flush=True)
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        for (done, _) in enumerate(pool.imap_unordered(check_rectangle, enumerate(rows), chunksize=32), 1):
            if done % 5000 == 0:
                print(f'{done}/{len(rows)} rectangles: PASS', flush=True)
    print(f"PASS: {len(rows)} rectangles, {sum((len(row['children']) for row in rows))} child intervals.")
if __name__ == '__main__':
    try:
        main()
    except (ValueError, TypeError, KeyError, ArithmeticError, OSError) as error:
        print('FAIL: ' + str(error), file=sys.stderr)
        sys.exit(1)
