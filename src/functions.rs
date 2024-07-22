use std::f64::consts::{E, PI};

use crate::datastructures::Number;

/*
sum:                x + y
substraction:       x - y
multiplication:     x * y
division:           x / y
power:              x ^ y
+ mod
*/

pub const FN_STR_INV: &'static str = "inv";
pub const FN_STR_SQRT: &'static str = "sqrt";
pub const FN_STR_SIN: &'static str = "sin";
pub const FN_STR_COS: &'static str = "cos";
pub const FN_STR_TAN: &'static str = "tan";
pub const FN_STR_ASIN: &'static str = "arcsin";
pub const FN_STR_ACOS: &'static str = "arccos";
pub const FN_STR_ATAN: &'static str = "arctan";
pub const FN_STR_EXP: &'static str = "exp";
pub const FN_STR_LN: &'static str = "ln";
pub const FN_STR_ABS: &'static str = "abs";

//constants
pub const CONST_STR_PI: &'static str = "pi";
pub const CONST_STR_DEG2RAD: &'static str = "deg2rad";
pub const CONST_STR_RAD2DEG: &'static str = "rad2deg";
pub const CONST_STR_PHI: &'static str = "phi";
pub const CONST_STR_E: &'static str = "e";
pub const CONST_STR_TAU: &'static str = "tau";

pub struct Functions {}
pub struct Constants {}

impl Functions {
    /// Finds the function by the name and evaluates it on the given input.
    ///
    /// If it attemps to evaluate the function outside the bounds of the domain,
    /// it will retun the corresponding error. It will also return an error if
    /// the function s not found.
    pub fn find_and_evaluate(function_name: &str, mut input: Number) -> Result<Number, String> {
        /*function name must be in lowercase and match exacly with the corresponding name.  */

        input.minimize();

        return match function_name {
            FN_STR_SQRT => match input {
                Number::Real(r) => {
                    if r < 0.0 {
                        Err(format!(
                            "The input of sqrt cannot be negative. Input provided: {}",
                            r
                        ))
                    } else {
                        Ok(Number::new_real(r.sqrt()))
                    }
                }
                Number::Rational(num, den) => {
                    if num < 0 {
                        Err(format!(
                            "The input of sqrt cannot be negative. Input provided: {}",
                            num as f64 / den as f64
                        ))
                    } else {
                        match (
                            Number::is_perfect_square(num),
                            Number::is_perfect_square(den as i64),
                        ) {
                            (None, None) => Ok(Number::new_real((num as f64 / den as f64).sqrt())),
                            (None, Some(d)) => {
                                Ok(Number::new_real((num as f64).sqrt() / (d as f64)))
                            }
                            (Some(n), None) => {
                                /*Use rationalitzation for better numerical performance:
                                a/sqrt(b) = a * sqrt(b)/sqrt(b) * sqrt(b) = a * sqrt(b) / b
                                */
                                let d: f64 = den as f64;
                                let rationalized: f64 = n as f64 * d.sqrt() / d;
                                Ok(Number::new_real(rationalized))
                            }
                            (Some(n), Some(d)) => Ok(Number::new_rational(n, d as u64)?),
                        }
                    }
                }
            },
            FN_STR_INV => match input {
                Number::Real(r) => {
                    if r == 0.0 {
                        Err(String::from("Division by 0 is not possible. "))
                    } else {
                        Ok(Number::Real(1.0 / r))
                    }
                }
                Number::Rational(num, den) => {
                    if num == 0 {
                        Err(String::from("Division by 0 is not possible. "))
                    } else {
                        let sign: i64 = num.signum();
                        Ok(Number::Rational((den as i64) * sign, num.abs() as u64))
                    }
                }
            },
            FN_STR_SIN => Ok(Number::new_real(input.get_numerical().sin())),
            FN_STR_COS => Ok(Number::new_real(input.get_numerical().cos())),
            FN_STR_TAN => {
                let x: f64 = input.get_numerical();
                if x / PI % 1.0 as f64 == 0.5 {
                    Err(String::from("The domain of tan(x) does not include values in the form x = PI*(1/2 + n), where n is an integer. "))
                } else {
                    Ok(Number::new_real(x.tan()))
                }
            }
            FN_STR_ASIN => {
                let x: f64 = input.get_numerical();
                //Ok(Number::new_real(input.get_numerical().sin())),
                if -1.0 <= x && x <= 1.0 {
                    // inside domain
                    Ok(Number::new_real(x.asin()))
                } else {
                    Err("The domain of arcsin() is [-1, 1]. ".to_string())
                }
            }
            FN_STR_ACOS => {
                let x: f64 = input.get_numerical();
                //Ok(Number::new_real(input.get_numerical().sin())),
                if -1.0 <= x && x <= 1.0 {
                    // inside domain
                    Ok(Number::new_real(x.acos()))
                } else {
                    Err("The domain of arccos() is [-1, 1]. ".to_string())
                }
            }
            FN_STR_ATAN => Ok(Number::new_real(input.get_numerical().atan())),
            FN_STR_EXP => Ok(Number::new_real(input.get_numerical().exp())),
            FN_STR_LN => {
                let x: f64 = input.get_numerical();

                if x <= 0.0 {
                    Err("The domain of ln() is the positive reals excluding 0. ".to_string())
                } else {
                    Ok(Number::new_real(x.ln()))
                }
            }
            FN_STR_ABS => {
                match input {
                    Number::Real(r) => Ok(Number::new_real(r.abs())),
                    Number::Rational(num, den) => Ok(Number::new_rational(num.abs(), den)
                        .expect("The number was already rational")),
                }
            }

            _ => Err("Function not found. ".to_string()),
        };
    }
}

impl Constants {
    /// Searches a constant by name and returns a [Number] containing it's value.
    ///
    /// If the constant is not found, returns None.
    pub fn get_constant(constant_name: &str) -> Option<Number> {
        //returns Ok(Number) containing the value of the constant or None
        //if the constant is not found

        return match constant_name.to_lowercase().as_ref() {
            CONST_STR_PI => Some(Number::new_real(PI)),
            CONST_STR_DEG2RAD => Some(Number::new_real(PI / 180 as f64)),
            CONST_STR_RAD2DEG => Some(Number::new_real(180 as f64 / PI)),
            CONST_STR_PHI => Some(Number::new_real(
                (1.0 as f64 + (5.0 as f64).sqrt()) / 2.0 as f64,
            )),
            CONST_STR_E => Some(Number::new_real(E)),
            CONST_STR_TAU => Some(Number::new_real(PI * 2 as f64)),
            _ => None,
        };
    }
}
