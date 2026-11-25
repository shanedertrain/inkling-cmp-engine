//! C `%g` number formatting, so that the tools' JSON lines are byte-identical to the C++ tools'
//! (`std::ostringstream` with `precision(n)` and `printf("%.ng")` both print this format).

/// Formats `x` as C's `%.{prec}g`: `prec` significant digits, trailing zeros removed, scientific notation when the
/// decimal exponent is below -4 or at least `prec`.
#[must_use]
pub fn g(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return if x.is_sign_negative() { "-nan" } else { "nan" }.to_owned();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_owned();
    }
    let prec = prec.max(1);
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    // The exponent after rounding to `prec` digits decides the notation (C99 7.21.6.1).
    let sci = format!("{:.*e}", prec - 1, x);
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let limit = i32::try_from(prec).unwrap_or(i32::MAX);
    if exp < -4 || exp >= limit {
        let mantissa = strip_zeros(mantissa);
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.unsigned_abs())
    } else {
        let decimals = usize::try_from(limit - 1 - exp).unwrap_or(0);
        strip_zeros(&format!("{x:.decimals$}")).to_owned()
    }
}

/// Removes trailing zeros after a decimal point, and the point itself when nothing follows it.
fn strip_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::g;

    /// Expected strings are Python's `'%.9g' % x` and `'%.6g' % x` (the C library's output).
    #[test]
    fn matches_c_printf() {
        let cases: &[(f64, &str, &str)] = &[
            (0.1, "0.1", "0.1"),
            (1.0, "1", "1"),
            (-3.5, "-3.5", "-3.5"),
            (1e-5, "1e-05", "1e-05"),
            (123_456_789.0, "123456789", "1.23457e+08"),
            (1_234_567_890.0, "1.23456789e+09", "1.23457e+09"),
            (-2.718_281_9, "-2.7182819", "-2.71828"),
            (1e21, "1e+21", "1e+21"),
            (0.0001, "0.0001", "0.0001"),
            (1.234_567_891e-5, "1.23456789e-05", "1.23457e-05"),
            (99_999_999.95, "100000000", "1e+08"),
            (5e-324, "4.94065646e-324", "4.94066e-324"),
            (-0.0, "-0", "-0"),
            (1.5e300, "1.5e+300", "1.5e+300"),
            (-9.876_543_21e-7, "-9.87654321e-07", "-9.87654e-07"),
            (-11.234_567_891_234_5, "-11.2345679", "-11.2346"),
            (0.5, "0.5", "0.5"),
            (100.0, "100", "100"),
            (1e9, "1e+09", "1e+09"),
            (999_999_999.5, "1e+09", "1e+09"),
        ];
        for &(x, nine, six) in cases {
            assert_eq!(g(x, 9), nine, "%.9g of {x:e}");
            assert_eq!(g(x, 6), six, "%.6g of {x:e}");
        }
    }

    #[test]
    fn non_finite() {
        assert_eq!(g(f64::INFINITY, 9), "inf");
        assert_eq!(g(f64::NEG_INFINITY, 9), "-inf");
        assert_eq!(g(f64::NAN, 6), "nan");
    }
}
