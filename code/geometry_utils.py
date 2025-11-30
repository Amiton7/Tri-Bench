import math

DEFAULT_TOL_EQ_REL = 0.03
DEFAULT_TOL_RIGHT_DEG = 2.0


def triangle_angles(a, b, c):
    def safe_acos(x):
        x = max(-1.0, min(1.0, x))
        return math.degrees(math.acos(x))

    A = safe_acos((b * b + c * c - a * a) / (2 * b * c))
    B = safe_acos((a * a + c * c - b * b) / (2 * a * c))
    C = 180.0 - A - B
    return A, B, C


def side_type_from_lengths(AB, BC, CA, tol_rel: float = DEFAULT_TOL_EQ_REL) -> str:
    s = sorted([AB, BC, CA])
    eq01 = abs(s[0] - s[1]) <= tol_rel * max(s[0], s[1])
    eq12 = abs(s[1] - s[2]) <= tol_rel * max(s[1], s[2])
    if eq01 and eq12:
        return "equilateral"
    if eq01 or eq12:
        return "isosceles"
    return "scalene"


def angle_type_from_angles(A, B, C, tol_right: float = DEFAULT_TOL_RIGHT_DEG) -> str:
    mx = max(A, B, C)
    if abs(mx - 90.0) <= tol_right:
        return "right"
    return "obtuse" if mx > 90.0 else "acute"


def derived_metrics_from_triangle(AB, BC, CA, Adeg, Bdeg, Cdeg):
    side_t = side_type_from_lengths(AB, BC, CA)
    angle_t = angle_type_from_angles(Adeg, Bdeg, Cdeg)

    ab_over_ac = AB / CA if CA > 0 else float("nan")
    abs_b_minus_c = abs(Bdeg - Cdeg)
    max_over_min = max(AB, BC, CA) / min(AB, BC, CA) if min(AB, BC, CA) > 0 else float("inf")
    angle_range = max(Adeg, Bdeg, Cdeg) - min(Adeg, Bdeg, Cdeg)

    return {
        "side_type": side_t,
        "angle_type": angle_t,
        "ab_over_ac": ab_over_ac,
        "abs_b_minus_c_deg": abs_b_minus_c,
        "max_over_min_side": max_over_min,
        "angle_range_deg": angle_range,
    }