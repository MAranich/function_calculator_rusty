//! This program allows evaluating numerical expressions, evaluating a given 
//! function and deriving functions. You can create expressions using basic 
//! mathematical operations and evaluate and compute it's derivatives. 
//! 
//! Use the character `x` as the variable to create functions. 
//! 
//! ***
//! ## Operations and syntax:
//!
//! This project supports the following operations:
//!
//! 1) Addition                         (using `+`, as in `"2+5"`)
//! 2) Substraction and negation        (using `-`, as in `"3-5.5"` or in `"-sin(-2)"`)
//! 3) Multiplication                   (using `*`, as in `"4*8"`)
//! 4) Division                         (using `/`, as in `"2/9"`)
//! 5) Modulus                          (using `%`, as in `"19%10"`)
//! 6) Exponentiation                   (using `^` or `**` as in `"2^10"` or `"2**10"`)
//! 7) Factorial                        (using `!`, as in `"6!"`)
//! 8) Square root                      (using `sqrt(x)`)
//! 9) Sinus                            (using `sin(x)`)
//! 10) Cosinus                         (using `cos(x)`)
//! 11) Tangent                         (using `tan(x)`)
//! 12) Arcsinus (inverse sinus)        (using `arcsin(x)`)
//! 13) Arccosinus (inverse cosinus)    (using `arccos(x)`)
//! 14) Arctangent (inverse tangent)    (using `arctan(x)`)
//! 15) Exponential                     (using `exp(x)`)
//! 16) Natural logarithm               (using `ln(x)`)
//! 17) Absolute value                  (using `abs(x)`)
//! 18) Floor function                  (using `floor(x)`)
//! 19) Ceil function                   (using `ceil(x)`)
//! 20) Random  (Uniform [0, 1])        (using `rand(x)`)
//! 21) Gamma                           (using `gamma(x)`)
//!
//! + All the functions can be combined and composed in any way as long as they are
//! mathematically correct and fullfill the syntax requirments.
//!
//! + Some operations have priority over others, such as multiplication over
//! addition. That means that `"2+5*3"` will be evaluated as `"2+(5*3)"`. To overwrite
//! the order parenthesis can be used `()`.
//!
//! + All the trigonometric functions work with radians. In order to use degrees, multiply your
//! value by `DEG2RAD`, for example: `sin(90*DEG2RAD)`
//!
//! + Only real values are supported (no complex values), therefore `"sqrt(-1)"` lies outside
//! the domains of the function and will return an error indicating the invalid
//! evaluation.
//!
//! + Division by 0 is not allowed.
//!
//! + Every parenthesis must be closed. Any of `()`, `{}` and `[]` can be used. 
//! They are equivalent but must to match it's counterpart.
//!
//! + Spaces are ignored, you can add all you want or even remove them completly.
//!
//! + Remember that a [logarithm](https://en.wikipedia.org/wiki/Logarithm#Change_of_base)
//! in any base `b` can be expressed as `log_b(x) = (ln(x)/ln(b))` .
//!
//! ***
//!
//! ## Constants:
//!
//! The program will automatically translate some constants to it's corresponding
//! numerical values.
//!
//! 1)  [x]  PI              (equal to 3.141592653589793)
//! 2)  [x]  RAD2DEG         (equal to 57.29577951308232 = 180 / PI)
//! 3)  [x]  DEG2RAD         (equal to 0.0174532925199433 = PI / 180)
//! 4)  [x]  phi             (equal to 1.618033988749895 = (1+sqrt(5))/2 )
//! 5)  [x]  e               (equal to 2.718281828459045)
//! 6)  [x]  tau             (equal to 6.283185307179586)
//! 7)  [ ]  gravitational   (equal to 0.000000000066743 = 6.6743 * 10^-11 m^3/(kg * s^2), the gravitational constant)
//! 8)  [ ]  plank           (equal to 0.000000000000000000000000000000000662607015 = 6.62607015 * 10^-34 J*s, Plank constant)
//! 9)  [ ]  light           (equal to 299 792 458 m/s, speed of light)
//! 10) [ ]  elecprem        (equal to 0.0000000000088541878188 = 8.8541878188 * 10^-12, vacuum electric permittivity)
//! 11) [ ]  magnprem        (equal to 0.00000125663706127 = 1.25663706127 * 10^-6, vacuum magnetic permeability)
//! 12) [ ]  elecmass        (equal to 0.00000000000000000000000000000091093837139 = 9.1093837139 * 10^-31 kg, mass of the electron)
//!
//!
//! Constants can be written on any combination of uppercase and lowercase letters.
//! Physical constants have [IS units](https://en.wikipedia.org/wiki/International_System_of_Units).
//!

use clap::ArgAction;
use clap::{command, Arg};
use core::panic;
use std::env;

//#[allow(unused_parens)]

/// All information regarding the datastructures.
pub mod datastructures;

/// Functions to process the string until evaluation.
pub mod processing;

/// Hardcoded data that other parts of the code use.
pub mod setup;

/// ## Cotains the functions and constants that can be used.
pub mod functions;

/// All the testing.
#[cfg(test)]
mod tests;

use crate::datastructures::*;
use crate::processing::*;
use functions::Constants;

const DESCRIPTION: &'static str = "
 - Every parenthesis must be closed. Any of `()`, `{}` and `[]` can be used. They are equivalent but must to match it's counterpart. 

 - Spaces are ignored, you can add all you want or even remove them completly. 

 - Remember that a [logarithm](https://en.wikipedia.org/wiki/Logarithm#Change_of_base) in any base `b` can be expressed as `log_b(x) = (ln(x)/ln(b))` .

\tVaid operations: 
Addition (`+`)
Substraction (`-`)
Multiplication (`*`)
Division (`/`)
Modulus (`%`)
Exponentiation (`^` or `**`)
Factorial (`!`)
Square root (`sqrt(x)`)
Sinus (`sin(x)`)
Cosinus (`cos(x)`)
Tangent  (`tan(x)`)
Arcsinus (`arcsin(x)`)
Arccosinus (`arccos(x)`)
Arctangent (`arctan(x)`)
Exponential (`exp(x)`, recommended over `e^x`)
Natural logarithm (`ln(x)`)
Absolute value (`abs(x)`)
Floor function (`floor(x)`)
Ceil function (`ceil(x)`)
Random  (Uniform [0, 1]) (`rand(x)`)
Gamma (`gamma(x)`)

 - All the functions can be combined and composed in any way as long as they are mathematically correct and fullfill the syntax requirments.

 - Use `der(x)` to calculate the derivative of the inside. 



"; 

/*
    Addition (`+`)
    Substraction (`-`)
    Multiplication (`*`)\
    Division (`/`)
    Modulus (`%`)
    Exponentiation (`^` or `**`)
    Factorial (`!`)
    Square root (`sqrt(x)`)
    Sinus (`sin(x)`)
    Cosinus (`cos(x)`)
    Tangent  (`tan(x)`)
    Arcsinus (`arcsin(x)`)
    Arccosinus (`arccos(x)`)
    Arctangent (`arctan(x)`)
    Exponential (`exp(x)`, recommended over `e^x`)
    Natural logarithm (`ln(x)`)
    Absolute value (`abs(x)`)
    Floor function (`floor(x)`)
    Ceil function (`ceil(x)`)
    Random  (Uniform [0, 1]) (`rand(x)`)
    Gamma (`gamma(x)`)

    //! 15) Exponential                     (using `exp(x)`)
    //! 16) Natural logarithm               (using `ln(x)`)
    //! 17) Absolute value                  (using `abs(x)`)
    //! 18) Floor function                  (using `floor(x)`)
    //! 19) Ceil function                   (using `ceil(x)`)
    //! 20) Random  (Uniform [0, 1])        (using `rand(x)`)
    //! 21) Gamma                           (using `gamma(x)`)
*/

fn main() {
    let matched: clap::ArgMatches = command!()
        .about("\n\n\nThis program allows evaluating numerical expressions, evaluating a given function and deriving functions. ")
        .arg(
            Arg::new("expression")
                .required(true)
                .help("The expression or function to evaluate or derive. ")
        ).arg(
            Arg::new("evaluation_points")
                .short('e')
                .long("evaluate")
                .action(ArgAction::Append)
                .help("If set, will evaluate the expression at the given point for the given variable. Must have the form -e <var>=<num>   ")
        ).arg(
            Arg::new("derives")
                .short('d')
                .long("derive")
                .help("Derivates the expression the desired amount of times. Must be a positive number. ")
        ).arg(
            Arg::new("derive_var")
                .short('D')
                .long("dervar")
                .help("The variable wich all the derivartves will be taken in respect to. Deafult is 'x' unless overiden. ")
        ).arg(
            Arg::new("raw")
                .short('r')
                .long("raw")
                .help("Do not simplify or rewrite any expression. ")
                .action(ArgAction::SetFalse)
        ).arg(
            Arg::new("numerical")
                .short('n')
                .long("numerical")
                .help("Displays the most exact results possible, without any sort of simplification. ")
                .action(ArgAction::SetTrue)
        ).arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Displays extra information regarding to the steps to get the awnser, such as each derivation step. ")
                .action(ArgAction::SetTrue)
        ).after_help(DESCRIPTION)
        
        .get_matches();


    
    



    let evaluation_vec: Vec<(char, Number)> = matched
        .get_many::<String>("evaluation_points")
        .unwrap_or_default()
        .map(|v| {
            let eval_result: Result<(char, Number), String> = parse_evaluation_args(v.as_str()); 
            match eval_result {
                Ok(x) => x,
                Err(e) => panic!("{}", e),
            }
        
        })
        .collect::<Vec<(char, Number)>>() ;

    // Just takes the 1st char of the string
    let diferentiation_variable: char = matched.get_one::<String>("derive_var").map(|s|s.to_owned().chars().next().unwrap()).unwrap_or('x'); 


    let number_of_derives: u16 =
        matched.get_one::<String>("derives")
            .map(|n| match n.parse::<u16>() {
                Ok(v) => v,
                Err(e) => panic!("{}", e),
            }).unwrap_or(0);


    let verbose_flag: bool = matched.get_flag("verbose");

    let simplify_output_flag: bool = matched.get_flag("raw");

    if matched.get_flag("numerical") {
        unsafe {
            datastructures::NUMERICAL_OUTPUTS = true;
        }
    }

    let input: String = matched
        .get_one::<String>("expression")
        .expect("Main argument is needed. ")
        .to_owned();

    // ****************************************************************************

    let mut calc: Calculator = Calculator::new(setup::setup_dfas(), SRA::new(setup::get_rules()));

    let original_ast: AST = match generate_ast(input, &mut calc, verbose_flag) {
        Ok(ret) => ret,
        Err(msg) => panic!("\n{}", msg),
    };

    if verbose_flag {
        println!(
            "\n\tThe generated AST: \n\n{:#?}\n",
            original_ast.to_string()
        );
        println!(
            "Does the AST contain: \n\tVariables: {}\n\tDerivatives: {}\n",
            original_ast.contains_variable(),
            original_ast.contains_derives()
        );
    }

    let expanded_ders: AST = {
        if original_ast.contains_derives() {
            
            if verbose_flag {
                println!("Expanding the derivatives in the expression: \n"); 
            }

            match original_ast.execute_derives(diferentiation_variable, false, verbose_flag) {
                Ok((v, f)) => {
                    assert!(f == false);
                    v
                }
                Err(msg) => panic!("\n{}", msg),
            }
        } else {
            original_ast
        }
    };

    if verbose_flag {
        println!("AST stringified: \n\n{}\n\n", expanded_ders.to_string());
    }

    let mut ast: AST = if simplify_output_flag {
        match expanded_ders.simplify_expression() {
            Ok(x) => match x.partial_evaluation() {
                Ok(y) => y,
                Err(msg) => panic!("\n{}", msg),
            },
            Err(msg) => panic!("\n{}", msg),
        }
    } else {
        expanded_ders
    };

    if verbose_flag && number_of_derives != 0 {
        println!("\tDerivating expression {} times: \n", number_of_derives); 
    }

    for _i in 0..number_of_derives {
        ast = match ast.full_derive(diferentiation_variable, simplify_output_flag, verbose_flag) {
            Ok(new_ast) => new_ast,
            Err(msg) => panic!("{}", msg),
        }
    }

    if verbose_flag {
        println!("Final AST: \n\n\t{}\n\n", ast.to_string());
    } else {
        // we need to print the final result anyway
        println!("\n\n{}", ast.to_string());
    }

    if evaluation_vec.len() != 0 {
        let result: Number = match ast.evaluate(evaluation_vec) {
            Ok(v) => v,
            Err(msg) => panic!("{}", msg),
        };
        if verbose_flag {
            println!("The function was evaluated to: {}\n", result.as_str());
        } else {
            // We need to print what we were asked for. 
            println!("{}\n", result.as_str());
        }
    }

    print!("\n\n");
}

/*

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut input_vec: Vec<String> = args.clone();
    input_vec.remove(0); // 0 is name of program

    println!("{:#?}", input_vec);
    println!("|{}|", input_vec[0]);

    let mut input: String = if input_vec[0].contains('"') {
        input_vec[0][1..(input_vec[0].len() - 1)].to_string()
        //remove quotes and return string
    } else {
        //panic!("Pass the function argument wrapped in \". For example: \"x^2-ln(4*x + sqrt(PI))\"\n\n");
        input_vec[0].clone()
    };

    if input.len() == 0 {
        panic!("\nUse this command with an input. \ncargo run -- \"<your expression>\"");
    }

    let evaluation_point: Option<Number> = match input_vec.get(2) {
        Some(point) => match parse_evaluation_point(point) {
            Ok(v) => Some(v),
            Err(msg) => panic!("{}", msg),
        },
        None => None,
    };

    input = input.trim().to_string();
    println!("\nProcessed input: {:#?}\n", input);
}
*/
