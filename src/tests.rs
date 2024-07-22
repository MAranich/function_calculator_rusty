use crate::*;
use core::panic;


#[test]
fn stringify_ast_test() {

    let tests: Vec<(String, String)> = vec![(String::from("2 +  3"), String::from("2+3"))]; 

    let mut calc: Calculator = Calculator::new(setup::setup_dfas(), SRA::new(setup::get_rules()));

    for (input, expected_output) in tests {
        println!("\nTesting input: {:#?}\n", input);

        match generate_ast(input, &mut calc, true) {
            Ok(ret) => if ret.to_string() != expected_output {
                panic!("\nExpression evaluated to: \n\t{}\nInstead of teh expected: \n\t{}", ret.to_string(), expected_output); 
            }, 
            Err(msg) => panic!("\n{}", msg),
        }; 
    }



}
