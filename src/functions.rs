use std::{
    cell::{Ref, RefCell},
    f64::consts::{E, PI},
    rc::Rc,
};

use crate::{datastructures::Number, Element, AST};

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

    pub fn func_derive(input: &AST) -> Result<AST, String> {
        let function_name: &String = if let Element::Function(iden) = &input.value {
            iden
        } else {
            return Err(format!(
                "func_derive must be called with functions. It was called with: \n{}",
                input.to_string()
            ));
        };

        if !input.contains_variable() {
            // the derivative of anything that does not depend on the variable is 0
            return Ok(AST {
                value: Element::Number(Number::Rational(0, 1)),
                children: Vec::new(),
            });
        }

        let ret: AST = match function_name.as_str() {
            FN_STR_INV => {
                // 1/f => f'/(f^2)

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // f^2
                let f_sq: AST = AST {
                    value: Element::Exp,
                    children: vec![
                        input.children[0].clone(),
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(2, 1)))),
                    ],
                };

                // f'/(f^2)
                AST {
                    value: Element::Div,
                    children: vec![Rc::new(RefCell::new(der)), Rc::new(RefCell::new(f_sq))],
                }
            }
            FN_STR_SQRT => {
                // sqrt(f) = f'/(2*sqrt(f))

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                //2*sqrt(f)
                let f2: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(2, 1)))),
                        Rc::new(RefCell::new(input.clone())),
                    ],
                };

                // f'/(2*sqrt(f))
                AST {
                    value: Element::Div,
                    children: vec![Rc::new(RefCell::new(der)), Rc::new(RefCell::new(f2))],
                }
            }
            FN_STR_SIN => {
                // sin(f) => cos(f)*f'

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // cos(f)
                let mut cos: AST = input.clone();
                cos.value = Element::Function(String::from(FN_STR_COS));

                // cos(f)*f'
                AST {
                    value: Element::Mult,
                    children: vec![Rc::new(RefCell::new(cos)), Rc::new(RefCell::new(der))],
                }
            }
            FN_STR_COS => {
                // cos(f) => -sin(f)*f'

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // sin(f)
                let mut sin: AST = input.clone();
                sin.value = Element::Function(String::from(FN_STR_SIN));

                // -sin(f)
                let minus_sin: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(-1, 1)))),
                        Rc::new(RefCell::new(sin)),
                    ],
                };

                // -sin(f)*f'
                AST {
                    value: Element::Mult,
                    children: vec![Rc::new(RefCell::new(minus_sin)), Rc::new(RefCell::new(der))],
                }
            }
            FN_STR_TAN => {
                // tan(f) => (1 + tan(f)^2) * f'

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // tan(f)^2
                let tan_sq: AST = AST {
                    value: Element::Exp,
                    children: vec![
                        Rc::new(RefCell::new(input.clone())),
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(2, 1)))),
                    ],
                };

                // tan(f)^2 + 1
                let tan_sq_plus_1: AST = AST {
                    value: Element::Add,
                    children: vec![
                        Rc::new(RefCell::new(tan_sq)),
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(1, 1)))),
                    ],
                };

                // (1 + tan(f)^2) * f'

                AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(tan_sq_plus_1)),
                        Rc::new(RefCell::new(der)),
                    ],
                }
            }
            FN_STR_ASIN => {
                // arcsin(f) => f'/sqrt(1-f^2)

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // f^2
                let f_sq: AST = AST {
                    value: Element::Exp,
                    children: vec![
                        input.children[0].clone(),
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(2, 1)))),
                    ],
                };

                // 1-f^2
                let one_minus_f_sq: AST = AST {
                    value: Element::Sub,
                    children: vec![
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(1, 1)))),
                        Rc::new(RefCell::new(f_sq)),
                    ],
                };

                // sqrt(1-f^2)
                let sqrt_fn: AST = AST {
                    value: Element::Function(String::from(FN_STR_SQRT)),
                    children: vec![Rc::new(RefCell::new(one_minus_f_sq))],
                };

                // f'/sqrt(1-f^2)
                AST {
                    value: Element::Div,
                    children: vec![Rc::new(RefCell::new(der)), Rc::new(RefCell::new(sqrt_fn))],
                }
            }
            FN_STR_ACOS => {
                // arccos(f) => -f'/sqrt(1-f^2)

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // f^2
                let f_sq: AST = AST {
                    value: Element::Exp,
                    children: vec![
                        input.children[0].clone(),
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(2, 1)))),
                    ],
                };

                // 1-f^2
                let one_minus_f_sq: AST = AST {
                    value: Element::Sub,
                    children: vec![
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(1, 1)))),
                        Rc::new(RefCell::new(f_sq)),
                    ],
                };

                // sqrt(1-f^2)
                let sqrt_fn: AST = AST {
                    value: Element::Function(String::from(FN_STR_SQRT)),
                    children: vec![Rc::new(RefCell::new(one_minus_f_sq))],
                };

                // f'/sqrt(1-f^2)
                let arcsin_der: AST = AST {
                    value: Element::Div,
                    children: vec![Rc::new(RefCell::new(der)), Rc::new(RefCell::new(sqrt_fn))],
                };

                AST {
                    value: Element::Neg,
                    children: vec![Rc::new(RefCell::new(arcsin_der))],
                }
            }
            FN_STR_ATAN => {
                // arctan(f) => f'/1+f^2

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // f^2
                let f_sq: AST = AST {
                    value: Element::Exp,
                    children: vec![
                        input.children[0].clone(),
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(2, 1)))),
                    ],
                };

                // 1+f^2
                let one_plus_f_sq: AST = AST {
                    value: Element::Add,
                    children: vec![
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(1, 1)))),
                        Rc::new(RefCell::new(f_sq)),
                    ],
                };

                AST {
                    value: Element::Div,
                    children: vec![
                        Rc::new(RefCell::new(der)),
                        Rc::new(RefCell::new(one_plus_f_sq)),
                    ],
                }
            }
            FN_STR_EXP => {
                // exp(f) => exp(f) * f'

                // f'
                let der: AST = input.children[0].borrow().derive()?;
                

                // exp(f) * f'
                AST {
                    value: Element::Mult,
                    children: vec![Rc::new(RefCell::new(input.clone())), Rc::new(RefCell::new(der))],
                }
            }
            FN_STR_LN => {
                // ln(f) => f'/f

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // f'/f
                AST {
                    value: Element::Div,
                    children: vec![Rc::new(RefCell::new(der)), input.children[0].clone()],
                }
            }
            FN_STR_ABS => {
                // |f| =>  f / ||

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                todo!();
            }
            _ => {
                return Err(String::from(
                    "Trying to derive a function that does not exist / unimplemented. \n",
                ));
            }
        };

        return Ok(ret);
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
