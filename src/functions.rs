use std::{
    cell::RefCell,
    f64::consts::{E, PI},
    mem::transmute,
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
pub const FN_STR_FLOOR: &'static str = "floor";
pub const FN_STR_CEIL: &'static str = "ceil";

//constants
pub const CONST_STR_PI: &'static str = "pi";
pub const CONST_STR_DEG2RAD: &'static str = "deg2rad";
pub const CONST_STR_RAD2DEG: &'static str = "rad2deg";
pub const CONST_STR_PHI: &'static str = "phi";
pub const CONST_STR_E: &'static str = "e";
pub const CONST_STR_TAU: &'static str = "tau";

pub const LIST_CONST_STR: [&'static str; 6] = [
    CONST_STR_PI,
    CONST_STR_DEG2RAD,
    CONST_STR_RAD2DEG,
    CONST_STR_PHI,
    CONST_STR_E,
    CONST_STR_TAU,
];

pub const LIST_CONST_VAUE_STR: [(&'static str, Number); 6] = [
    (CONST_STR_PI, Number::Real(PI)),
    (CONST_STR_DEG2RAD, Number::Real(PI / (180 as f64))),
    (CONST_STR_RAD2DEG, Number::Real(180 as f64 / PI)),
    (
        CONST_STR_PHI,
        Number::Real(unsafe { transmute::<u64, f64>(0x3ff9e3779b97f4a8) }),
    ),
    (CONST_STR_E, Number::Real(E)),
    (CONST_STR_TAU, Number::Real(PI * 2 as f64)),
];

//const phi: f64 = f64::from_bits(0x3ff9e3779b97f4a8);
//const phi: f64 = unsafe { transmute::<u64, f64>(0x3ff9e3779b97f4a8) } ;

pub struct Functions {}
pub struct Constants {}

impl Functions {
    /// Finds the function by the name and evaluates it on the given input.
    ///
    /// If it attemps to evaluate the function outside the bounds of the domain,
    /// it will retun the corresponding error. It will also return an error if
    /// the function s not found. The function name must be in lowercase and
    /// match exacly with the corresponding name.
    pub fn find_and_evaluate(function_name: &str, mut input: Number) -> Result<Number, String> {
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
                            (None, Some(sqrt_d)) => {
                                Ok(Number::new_real((num as f64).sqrt() / (sqrt_d as f64)))
                            }
                            (Some(sqrt_n), None) => {
                                /*
                                Use rationalitzation for better numerical performance:
                                x = a*a/b => sqrt(x) = sqrt(a * a)/sqrt(b)
                                a/sqrt(b) = a * sqrt(b)/sqrt(b) * sqrt(b) = a * sqrt(b) / b
                                */

                                let f_den: f64 = den as f64;
                                let rationalized: f64 = (sqrt_n as f64 * f_den.sqrt()) / f_den;
                                Ok(Number::Real(rationalized))
                            }
                            (Some(sqrt_n), Some(sqrt_d)) => Ok(Number::Rational(sqrt_n, sqrt_d as u64)),
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
                if (x / PI % 1.0 as f64 - 0.5).abs() < f64::EPSILON * 32.0 {
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
                    Number::Rational(num, den) => Ok(Number::Rational(num.abs(), den)),
                }
            }

            _ => Err("Function not found. ".to_string()),
        };
    }

    /// Derivate a function
    ///
    /// The current [AST] node must have the [AST::value] with the variant [Element::Function]
    /// with a regognized function. Some parts of the expression may be evaluated, therefore, if the expression
    /// is invalid, the function may return an error.
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
                // 1/f => -f'/(f^2)

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // f^2
                let f_sq: AST = AST {
                    value: Element::Exp,
                    children: vec![
                        Rc::clone(&input.children[0]),
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(2, 1)))),
                    ],
                };

                // -f'
                let minus_der: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(AST::from_number(Number::Rational(-1, 1)))),
                        Rc::new(RefCell::new(der)),
                    ],
                };

                // f'/(f^2)
                AST {
                    value: Element::Div,
                    children: vec![
                        Rc::new(RefCell::new(minus_der)),
                        Rc::new(RefCell::new(f_sq)),
                    ],
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
                        Rc::new(RefCell::new(input.deep_copy())),
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
                let mut cos: AST = input.deep_copy();
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
                let mut sin: AST = input.deep_copy();
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
                        Rc::new(RefCell::new(input.deep_copy())),
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
                        Rc::clone(&input.children[0]),
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
                        Rc::clone(&input.children[0]),
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
                        Rc::clone(&input.children[0]),
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
                    children: vec![
                        Rc::new(RefCell::new(input.deep_copy())),
                        Rc::new(RefCell::new(der)),
                    ],
                }
            }
            FN_STR_LN => {
                // ln(f) => f'/f

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // f'/f
                AST {
                    value: Element::Div,
                    children: vec![Rc::new(RefCell::new(der)), Rc::clone(&input.children[0])],
                }
            }
            FN_STR_ABS => {
                // |f| =>  f / |f| * f'

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // f
                let inner: Rc<RefCell<AST>> = Rc::clone(&input.children[0]);

                // f / |f|
                let sign: AST = AST {
                    value: Element::Div,
                    children: vec![inner, Rc::new(RefCell::new(input.deep_copy()))],
                };

                AST {
                    value: Element::Mult,
                    children: vec![Rc::new(RefCell::new(sign)), Rc::new(RefCell::new(der))],
                }
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
            CONST_STR_PI => Some(LIST_CONST_VAUE_STR[0].1.clone()),
            CONST_STR_DEG2RAD => Some(LIST_CONST_VAUE_STR[1].1.clone()),
            CONST_STR_RAD2DEG => Some(LIST_CONST_VAUE_STR[2].1.clone()),
            CONST_STR_PHI => Some(LIST_CONST_VAUE_STR[3].1.clone()),
            CONST_STR_E => Some(LIST_CONST_VAUE_STR[4].1.clone()),
            CONST_STR_TAU => Some(LIST_CONST_VAUE_STR[5].1.clone()),
            _ => None,
        };
    }

    /// If the given number is very close to a constant, returns it's str name
    pub fn is_constant(num: &Number) -> Option<&str> {
        let tolerance: f64 = 0.00001;

        for (s, n) in LIST_CONST_VAUE_STR {
            if num.in_tolerance_range(&n, tolerance) {
                return Some(s); 
            }
        }

        None

        /* 
        if num.in_tolerance_range(&Number::new_real(PI), tolerance) {
            Some(&CONST_STR_PI)
        } else if num.in_tolerance_range(&Number::new_real(PI / 180 as f64), tolerance) {
            Some(&CONST_STR_DEG2RAD)
        } else if num.in_tolerance_range(&Number::new_real(180 as f64 / PI), tolerance) {
            Some(&CONST_STR_RAD2DEG)
        } else if num.in_tolerance_range(
            &Number::new_real((1.0 as f64 + (5.0 as f64).sqrt()) / 2.0 as f64),
            tolerance,
        ) {
            Some(&CONST_STR_PHI)
        } else if num.in_tolerance_range(&Number::new_real(E), tolerance) {
            Some(&CONST_STR_E)
        } else if num.in_tolerance_range(&Number::new_real(PI * 2 as f64), tolerance) {
            Some(&CONST_STR_TAU)
        } else {
            None
        }

        */

    }


}
