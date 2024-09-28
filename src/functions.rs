use core::f64;
use std::{
    cell::RefCell,
    f64::consts::{E, PI},
    fmt::format,
    mem::transmute,
    rc::Rc,
};

use rand::Rng;

use crate::{datastructures::Number, get_ptr, Element, AST, AST_ONE};

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
pub const FN_STR_GAMMA: &'static str = "gamma";
pub const FN_STR_ABS: &'static str = "abs";
pub const FN_STR_FLOOR: &'static str = "floor";
pub const FN_STR_CEIL: &'static str = "ceil";
pub const FN_STR_REAL: &'static str = "real";
pub const FN_STR_RATIONAL: &'static str = "rational";
pub const FN_STR_RATIONAL_2: &'static str = "rat";
pub const FN_STR_RANDOM: &'static str = "rand";
pub const FN_STR_RANDOM_2: &'static str = "random";
pub const FN_STR_RANDOM_3: &'static str = "rnd";

#[derive(Debug, PartialEq, Clone)]
pub enum FnIdentifier {
    Derive,
    Inv,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Arcsin,
    Arccos,
    Arctan,
    Exp,
    Ln,
    Abs,
    Floor,
    Ceil,
    Random,
    Real,
    Rational,
    Gamma,
}

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

const TWO_PI: f64 = f64::consts::PI * 2.0;

pub struct Functions {}
pub struct Constants {}

impl Functions {
    /// Finds the function by the name and evaluates it on the given input.
    ///
    /// If it attemps to evaluate the function outside the bounds of the domain,
    /// it will retun the corresponding error. It will also return an error if
    /// the function s not found. The function name must be in lowercase and
    /// match exacly with the corresponding name.
    pub fn find_and_evaluate(
        function_name: FnIdentifier,
        mut input: Number,
    ) -> Result<Number, String> {
        input.minimize();

        let ret: Number = match function_name {
            FnIdentifier::Derive => {
                return Err(String::from(
                    "Cannot evaluate a derivative. Expand derivative first and then evaluate. \n",
                ))
            }
            FnIdentifier::Inv => match input {
                Number::Real(r) => {
                    if r == 0.0 {
                        return Err(String::from("Division by 0 is not possible. "));
                    } else {
                        Number::Real(1.0 / r)
                    }
                }
                Number::Rational(num, den) => {
                    if num == 0 {
                        return Err(String::from("Division by 0 is not possible. "));
                    } else {
                        let sign: i64 = num.signum();
                        Number::Rational((den as i64) * sign, num.abs() as u64)
                    }
                }
            },
            FnIdentifier::Sqrt => match input {
                Number::Real(r) => {
                    if r < 0.0 {
                        return Err(format!(
                            "The input of sqrt cannot be negative. Input provided: {}",
                            r
                        ));
                    } else {
                        Number::new_real(r.sqrt())
                    }
                }
                Number::Rational(num, den) => {
                    if num < 0 {
                        return Err(format!(
                            "The input of sqrt cannot be negative. Input provided: {}",
                            num as f64 / den as f64
                        ));
                    } else {
                        match (
                            Number::is_perfect_square(num),
                            Number::is_perfect_square(den as i64),
                        ) {
                            (None, None) => Number::new_real((num as f64 / den as f64).sqrt()),
                            (None, Some(sqrt_d)) => {
                                Number::new_real((num as f64).sqrt() / (sqrt_d as f64))
                            }
                            (Some(sqrt_n), None) => {
                                /*
                                Use rationalitzation for better numerical performance:
                                x = a*a/b => sqrt(x) = sqrt(a * a)/sqrt(b)
                                a/sqrt(b) = a * sqrt(b)/sqrt(b) * sqrt(b) = a * sqrt(b) / b
                                */

                                let f_den: f64 = den as f64;
                                let rationalized: f64 = (sqrt_n as f64 * f_den.sqrt()) / f_den;
                                Number::Real(rationalized)
                            }
                            (Some(sqrt_n), Some(sqrt_d)) => Number::Rational(sqrt_n, sqrt_d as u64),
                        }
                    }
                }
            },
            FnIdentifier::Sin => Number::new_real(input.get_numerical().sin()),
            FnIdentifier::Cos => Number::new_real(input.get_numerical().cos()),
            FnIdentifier::Tan => {
                let x: f64 = input.get_numerical();
                if (x / PI % 1.0 as f64 - 0.5).abs() < f64::EPSILON * 32.0 {
                    return Err(String::from("The domain of tan(x) does not include values in the form x = PI*(1/2 + n), where n is an integer. "));
                }
                Number::new_real(x.tan())
            }
            FnIdentifier::Arcsin => {
                let x: f64 = input.get_numerical();
                //Ok(Number::new_real(input.get_numerical().sin())),
                if !(-1.0 <= x && x <= 1.0) {
                    // outside domain
                    return Err("The domain of arcsin() is [-1, 1]. ".to_string());
                }
                Number::new_real(x.asin())
            }
            FnIdentifier::Arccos => {
                let x: f64 = input.get_numerical();
                //Ok(Number::new_real(input.get_numerical().sin())),
                if !(-1.0 <= x && x <= 1.0) {
                    // outside domain
                    return Err("The domain of arccos() is [-1, 1]. ".to_string());
                }
                Number::new_real(x.acos())
            }
            FnIdentifier::Arctan => Number::new_real(input.get_numerical().atan()),
            FnIdentifier::Exp => Number::new_real(input.get_numerical().exp()),
            FnIdentifier::Ln => {
                let x: f64 = input.get_numerical();

                if x <= 0.0 {
                    // outside domain
                    return Err(
                        "The domain of ln() is the positive reals excluding 0. ".to_string()
                    );
                }
                Number::new_real(x.ln())
            }
            FnIdentifier::Abs => match input {
                Number::Real(r) => Number::new_real(r.abs()),
                Number::Rational(num, den) => Number::Rational(num.abs(), den),
            },
            FnIdentifier::Floor => 'floor: {
                match input {
                    Number::Real(r) => Number::Real(r.floor()),
                    Number::Rational(n, d) => {
                        // n = d * q + r
                        // n/d = q + 0.[]

                        if (n.abs() as u64) < d {
                            break 'floor Number::Rational(0, 1);
                        }

                        match n.cmp(&0) {
                            std::cmp::Ordering::Less => {
                                // negative number

                                let q: u64 = (n as u64) / d;
                                if q * d == n as u64 {
                                    //integer
                                    Number::Rational(-n, d)
                                } else {
                                    Number::Rational(-((q + 1_u64) as i64), 1)
                                }
                            }
                            std::cmp::Ordering::Equal => Number::Rational(0, 1),
                            std::cmp::Ordering::Greater => {
                                let q: u64 = n as u64 / d;

                                Number::Rational(q as i64, 1)
                                // ^drop r and return
                            }
                        }
                    }
                }
            }
            FnIdentifier::Ceil => 'ceil: {
                match input {
                    Number::Real(r) => Number::Real(r.ceil()),
                    Number::Rational(n, d) => {
                        match n.cmp(&0) {
                            std::cmp::Ordering::Less => {
                                // negative n,

                                let q: u64 = n as u64 / d;

                                Number::Rational(-(q as i64), 1)
                            }
                            std::cmp::Ordering::Equal => Number::Rational(0, 1),
                            std::cmp::Ordering::Greater => {
                                // n = d * q + r
                                // n/d = q + 0.[]
                                let q: u64 = (n as u64) / d;
                                if q * d == n as u64 {
                                    //integer
                                    break 'ceil Number::Rational(n, d);
                                }

                                Number::Rational((q + 1_u64) as i64, 1)
                            }
                        }
                    }
                }
            }
            FnIdentifier::Random => {
                // Returns a random uniformley distributed random number in [0, 1]
                // ^not sure if 0 and 1 are included

                let mut rand_gen: rand::prelude::ThreadRng = rand::thread_rng();

                Number::Real(rand_gen.gen::<f64>())
            }
            FnIdentifier::Real => {
                // Cast the number to a real representation.

                Number::Real(input.get_numerical())
            }
            FnIdentifier::Rational => {
                // Cast the number to the most accurate rational representation.

                match input {
                    Number::Rational(_, _) => input,
                    Number::Real(r) => match Number::rationalize(r) {
                        Ok(v) => v,
                        Err(v) => v,
                    },
                }
            }
            FnIdentifier::Gamma => 'gamma: {
                /*
                    For the computation:
                We know that:
                    gamma(z + 1) = z * gamma(z)

                 So we will use that formula to reduce the input to a value between
                 0 and 1 (or 1 and 2) and then evaluate a good aproximation. Then we multiply
                 everying and get a good aproxiation.

                 Formula for computing the gamma function: https://en.wikipedia.org/wiki/Stirling%27s_approximation#A_convergent_version_of_Stirling's_formula

                 */

                let is_int: bool = input.is_integer();
                let is_positive: bool = input.is_positive();

                if is_int {
                    if !is_positive {
                        return Err(String::from(
                            "The gamma function cannot be evaluated on negative integers. ",
                        ));
                    }

                    // gamma(x) = (x-1)! for positive integers.

                    let mut integer: u64 = match input {
                        Number::Real(r) => r as u64,
                        Number::Rational(n, d) => n as u64 / d,
                    };

                    integer = integer - 1; // change form gamma to factorial

                    let largest_factorial: u64 = 170;

                    if largest_factorial < integer {
                        // too big to compute
                        // the largest f64 valid number (non-infinite) is:
                        // 1.7976931348623157 * 10^308
                        // therefore if `integer!` is greater then that, we return an error

                        return Err(format!(
                            "Cannot compute gamma({}). Result is too big to store. ",
                            input.as_str()
                        ));
                    }

                    let largest_int_factorial: u64 = 20;

                    if integer <= largest_int_factorial {
                        // Compute with ints

                        if integer < 2 {
                            break 'gamma Number::Rational(1, 1);
                        }

                        let factorial: u64 =
                            (1..=integer).into_iter().reduce(|acc, x| x * acc).unwrap();
                        // ^less than 2^63-1, so safe to cast to i64

                        break 'gamma Number::Rational(factorial as i64, 1);
                    }

                    // use floats

                    let mut factorial: f64 = 1.0;

                    for i in 2..=integer {
                        factorial = factorial * i as f64;
                    }

                    break 'gamma Number::Real(factorial);
                }

                /*
                 ln(gamma(x)) = 1/2 * (ln(2*pi) - ln(x)) + x * (ln(x+1/(12*x - 1/(10*x)))-1)

                 then exponentiate. Done in this way for numerical stability.
                 Precision seems to increase as x grows. (Did I read that this
                 is acurate with an O(1/n) error somewhere (???))

                 todo: pass to ramanujan aprox. for precision ?

                */

                let mut x: f64 = input.get_numerical();

                let correction_term: f64 = {
                    //gamma(z + 1) = z * gamma(z)
                    //gamma(z) = (z - 1) * gamma(z - 1)
                    let mut acc: f64 = 1.0;
                    while x < 1.0 {
                        acc = acc * x;
                        x += 1.0;
                    }
                    acc
                };

                let left_term: f64 = 0.5 * (TWO_PI.ln() - x.ln());
                let denominator: f64 = 12.0 * x - 1.0 / (10.0 * x);
                let right_term: f64 = x * ((x + 1.0 / denominator).ln() - 1.0);

                let ret: f64 = (left_term + right_term).exp();

                println!("Number obtained: {}", x);
                println!("left: {}", left_term);
                println!("right: {}", right_term);
                println!("ret: {}", ret);

                Number::Real(ret / correction_term)
            }
        };

        return Ok(ret);

        /*
        match function_name {
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
                            (Some(sqrt_n), Some(sqrt_d)) => {
                                Ok(Number::Rational(sqrt_n, sqrt_d as u64))
                            }
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
            FN_STR_ABS => match input {
                Number::Real(r) => Ok(Number::new_real(r.abs())),
                Number::Rational(num, den) => Ok(Number::Rational(num.abs(), den)),
            },
            FN_STR_FLOOR => todo!(),
            FN_STR_CEIL => todo!(),
            FN_STR_GAMMA => {
                let is_int: bool = input.is_integer();

                if is_int && !input.is_positive() {
                    panic!()
                }

                todo!()
            }

            _ => Err("Function not found. ".to_string()),
        };
        */
    }

    /// Derivate a function
    ///
    /// The current [AST] node must have the [AST::value] with the variant [Element::Function]
    /// with a regognized function. Some parts of the expression may be evaluated, therefore, if the expression
    /// is invalid, the function may return an error.
    pub fn func_derive(input: &AST) -> Result<AST, String> {
        let function_name: FnIdentifier = if let Element::Function(iden) = &input.value {
            iden.to_owned()
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

        let ret: AST = match function_name {
            FnIdentifier::Inv => {
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
            FnIdentifier::Sqrt => {
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
            FnIdentifier::Sin => {
                // sin(f) => cos(f)*f'

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // cos(f)
                let mut cos: AST = input.deep_copy();
                cos.value = Element::Function(FnIdentifier::Cos);

                // cos(f)*f'
                AST {
                    value: Element::Mult,
                    children: vec![Rc::new(RefCell::new(cos)), Rc::new(RefCell::new(der))],
                }
            }
            FnIdentifier::Cos => {
                // cos(f) => -sin(f)*f'

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // sin(f)
                let mut sin: AST = input.deep_copy();
                sin.value = Element::Function(FnIdentifier::Sin);

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
            FnIdentifier::Tan => {
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
            FnIdentifier::Arcsin => {
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
                    value: Element::Function(FnIdentifier::Sqrt),
                    children: vec![Rc::new(RefCell::new(one_minus_f_sq))],
                };

                // f'/sqrt(1-f^2)
                AST {
                    value: Element::Div,
                    children: vec![Rc::new(RefCell::new(der)), Rc::new(RefCell::new(sqrt_fn))],
                }
            }
            FnIdentifier::Arccos => {
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
                    value: Element::Function(FnIdentifier::Sqrt),
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
            FnIdentifier::Arctan => {
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
            FnIdentifier::Exp => {
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
            FnIdentifier::Ln => {
                // ln(f) => f'/f

                // f'
                let der: AST = input.children[0].borrow().derive()?;

                // f'/f
                AST {
                    value: Element::Div,
                    children: vec![Rc::new(RefCell::new(der)), Rc::clone(&input.children[0])],
                }
            }
            FnIdentifier::Abs => {
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
            FnIdentifier::Ceil | FnIdentifier::Floor => {
                // we simplify the derivative to 0,
                // although it would not be defined at the integers.
                crate::datastructures::AST_ZERO.clone()
            }
            FnIdentifier::Real | FnIdentifier::Rational => {
                // this is only a cast between Number types, so the derivative is the identity.
                AST_ONE.clone()
            }
            FnIdentifier::Gamma => {
                // use classical definition of derivative for a good aplroximation of the origina result.

                let h: f64 = 0.0000001;
                let h_ast: AST = AST::from_number(Number::Real(h));
                let argument: Rc<RefCell<AST>> = Rc::clone(&input.children[0]);

                // gamma(x + h)
                let upper: AST = AST {
                    value: Element::Add,
                    children: vec![
                        get_ptr(argument.borrow().deep_copy()),
                        get_ptr(h_ast.clone()),
                    ],
                }.push_function(FnIdentifier::Gamma);

                // gamma(x+h) - gamma(x)
                let diference: AST = AST {
                    value: Element::Sub,
                    children: vec![get_ptr(upper), get_ptr(argument.borrow().deep_copy().push_function(FnIdentifier::Gamma))],
                };

                // (gamma(x+h) - gamma(x)) / h

                AST {
                    value: Element::Div,
                    children: vec![get_ptr(diference), get_ptr(h_ast)],
                }
            }
            FnIdentifier::Random => {
                return Err(String::from("Random function has no derivative. "))
            }
            FnIdentifier::Derive => return Err(String::from("Cannot derive a derivative. \n")),
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
    }
}

impl FnIdentifier {
    ///Returns the propper [FnIdentifier] variant for the input.
    ///
    /// The string can be in any combination of uppercase and lowercase.
    /// Returns error if there is no match.
    pub fn from_str(input: &str) -> Result<Self, String> {
        let ret: FnIdentifier = match input.to_lowercase().as_str() {
            FN_STR_INV => FnIdentifier::Inv,
            FN_STR_SQRT => FnIdentifier::Sqrt,
            FN_STR_SIN => FnIdentifier::Sin,
            FN_STR_COS => FnIdentifier::Cos,
            FN_STR_TAN => FnIdentifier::Tan,
            FN_STR_ASIN => FnIdentifier::Arcsin,
            FN_STR_ACOS => FnIdentifier::Arccos,
            FN_STR_ATAN => FnIdentifier::Arctan,
            FN_STR_EXP => FnIdentifier::Exp,
            FN_STR_LN => FnIdentifier::Ln,
            FN_STR_ABS => FnIdentifier::Abs,
            FN_STR_CEIL => FnIdentifier::Ceil,
            FN_STR_FLOOR => FnIdentifier::Floor,
            FN_STR_GAMMA => FnIdentifier::Gamma,
            FN_STR_REAL => FnIdentifier::Real,
            FN_STR_RATIONAL | FN_STR_RATIONAL_2 => FnIdentifier::Rational,
            FN_STR_RANDOM | FN_STR_RANDOM_2 | FN_STR_RANDOM_3 => FnIdentifier::Random,
            crate::processing::DERIVE_KEYWORD_1 | crate::processing::DERIVE_KEYWORD_2 => {
                FnIdentifier::Derive
            }
            _ => return Err(format!("Not a valid function name. Recived: {}\n", input)),
        };

        return Ok(ret);
    }
}

impl ToString for FnIdentifier {
    /// Returns a [String] that represents the function.
    fn to_string(&self) -> String {
        match self {
            FnIdentifier::Derive => "der".to_string(),
            FnIdentifier::Inv => FN_STR_INV.to_string(),
            FnIdentifier::Sqrt => FN_STR_SQRT.to_string(),
            FnIdentifier::Sin => FN_STR_SIN.to_string(),
            FnIdentifier::Cos => FN_STR_COS.to_string(),
            FnIdentifier::Tan => FN_STR_TAN.to_string(),
            FnIdentifier::Arcsin => FN_STR_ASIN.to_string(),
            FnIdentifier::Arccos => FN_STR_ACOS.to_string(),
            FnIdentifier::Arctan => FN_STR_ATAN.to_string(),
            FnIdentifier::Exp => FN_STR_EXP.to_string(),
            FnIdentifier::Ln => FN_STR_LN.to_string(),
            FnIdentifier::Abs => FN_STR_ABS.to_string(),
            FnIdentifier::Ceil => FN_STR_CEIL.to_string(),
            FnIdentifier::Floor => FN_STR_FLOOR.to_string(),
            FnIdentifier::Random => FN_STR_RANDOM.to_string(),
            FnIdentifier::Real => FN_STR_REAL.to_string(),
            FnIdentifier::Rational => FN_STR_RATIONAL.to_string(),
            FnIdentifier::Gamma => FN_STR_GAMMA.to_string(),
        }
    }
}
