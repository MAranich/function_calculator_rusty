use std::cell::RefCell;
use std::rc::Rc;

use crate::datastructures::*;
use crate::setup;
use crate::Constants;

/// All the valid characters that will separate tokens.
///
/// The token lexemme is separated both righ before and right after the separator itself.
/// It is already pre-sorted so binary search can be used on them.
const SEPARATORS: [char; 14] = [
    ' ', '!', '%', '(', ')', '*', '+', '-', '/', '[', ']', '^', '{', '}',
];

const DERIVE_KEYWORD_1: &'static str = "der";
const DERIVE_KEYWORD_2: &'static str = "derive";

pub fn get_ptr<T>(x: T) -> Rc<RefCell<T>>{
    Rc::new(RefCell::new(x))
}

/// Converts the given string into a vector of [Token] with the given [InstanceDFA].
///
/// No further processing is done.
pub fn tokenize_input(idfas: &mut Vec<InstanceDFA>, input: &String) -> Vec<Token> {
    let mut chars = input.chars();

    let mut is_separator: bool = SEPARATORS.binary_search(&chars.nth(0).unwrap()).is_ok();
    let final_c: char = chars.nth(input.len() - 2).unwrap();

    let mut token_list: Vec<Token> = vec![];
    let mut current_lexeme: Vec<char> = vec![];

    for processing_chars in input.chars().collect::<Vec<_>>().windows(2) {
        let c: char = processing_chars[0];
        let lookahead: char = processing_chars[1];

        let next_is_separator: bool = SEPARATORS.binary_search(&lookahead).is_ok();
        //find(|&&x| x == lookahead).is_some();

        current_lexeme.push(c);
        for dfa in &mut *idfas {
            if !dfa.alive {
                continue;
            }

            let _ = dfa.advance(c);
        }

        if is_separator || next_is_separator {
            let new_token: Token;
            let mut category: TokenClass = TokenClass::None;
            let mut accepted: bool;
            for dfa in &mut *idfas {
                accepted = dfa.finalize();
                if accepted {
                    category = dfa.asociated_class.clone().unwrap_or(TokenClass::None);
                    break;
                }
            }
            if current_lexeme[0] != ' ' {
                new_token = Token::new(Some(current_lexeme.iter().collect()), category);
                token_list.push(new_token);
            }

            current_lexeme = vec![];

            idfas.iter_mut().for_each(|x| x.reset());
        }

        is_separator = next_is_separator;
    }

    current_lexeme.push(final_c);
    for dfa in &mut *idfas {
        if !dfa.alive {
            continue;
        }

        let _ = dfa.advance(final_c);
    }

    let new_token: Token;
    let mut category: TokenClass = TokenClass::None;
    let mut accepted: bool;
    for dfa in &mut *idfas {
        accepted = dfa.finalize();
        if accepted {
            category = dfa.asociated_class.clone().unwrap_or(TokenClass::None);

            break;
        }
    }
    new_token = Token::new(Some(current_lexeme.iter().collect()), category);
    token_list.push(new_token);

    idfas.iter_mut().for_each(|x| x.reset());

    return token_list;
}

///If it finds a string that matches a known constant it will replace the [Token] identifier
/// with a [Token] Number that contains the number.
pub fn constant_matcher(mut input: Vec<Token>) -> Vec<Token> {
    //Swaps the constants (such as "pi" or "e" to their numerical counterparts)

    for element in &mut input {
        if element.class == TokenClass::Identifier && element.lexeme.is_some() {
            match Constants::get_constant(element.lexeme.clone().unwrap().as_ref()) {
                Some(number) => {
                    element.class = TokenClass::Number;
                    element.lexeme = Some(number.as_str());
                }
                None => {}
            }
        }
    }

    return input;
}

/// Uses the shunting yard algorithm to put the tokens from infix
/// notation in postfix notation.
///
/// Uses the shunting yard algorithm to put the vector of tokens from infix
/// notation (the normal one, such `"9+16*(5+2)"`) in postfix notation (also known as
/// reverse polish notation, `"9 16 5 2 + * +"`). Note how the reverse polish notation
/// does not need parenthesis. This makes the following parsing much more managable.  
/// Will return an error with a string in the case that the process fails.
pub fn shunting_yard_algorithm(input: Vec<Token>) -> Result<Vec<Token>, String> {
    // Es: https://es.wikipedia.org/wiki/Algoritmo_shunting_yard
    // En: https://en.wikipedia.org/wiki/Shunting_yard_algorithm

    let mut ret: Vec<Token> = Vec::with_capacity(input.len());
    let mut operators: Vec<Token> = Vec::new();

    let open_parenthesis: TokenModel = TokenModel::from_token(
        Token::new(Some("(".to_string()), TokenClass::SpecialChar),
        true,
    );
    let close_parenthesis: TokenModel = TokenModel::from_token(
        Token::new(Some(")".to_string()), TokenClass::SpecialChar),
        true,
    );

    for tok in input {
        if let TokenClass::None = tok.class {
            return Err(format!(
                "Token class is undefined (None). Token: {:#?}\n",
                tok
            ));
        }

        match tok.class.clone() {
            TokenClass::Number => ret.push(tok),
            TokenClass::Identifier => operators.push(tok),
            TokenClass::Variable => ret.push(tok),
            TokenClass::Operator => {
                let (op_priority, is_left_asociative): (i32, bool) =
                    match tok.get_precedence_operator() {
                        Some(tuple) => tuple,
                        None => {
                            return Err(format!(
                                "Token: {:?} marked as operator_opt is invalid/unsuported. ",
                                tok
                            ))
                        }
                    }; // 4, false

                loop {
                    let op_2: &Token = match operators.last() {
                        None => break,
                        Some(v) => v,
                    };

                    if open_parenthesis.cmp(op_2) {
                        break;
                    }
                    let op_2_info_opt: Option<(i32, bool)> = op_2.get_precedence_operator();

                    match op_2_info_opt {
                        //3, false
                        None => return Err(format!("Unidentified operator_opt: {:?}", op_2)),
                        Some((op_2_priority, _)) => {
                            /*if is_left_asociative {
                                if op_priority < op_2_priority {
                                    break;
                                }
                            } else {
                                if op_priority <= op_2_priority {
                                    break;
                                }
                            }*/
                            if op_priority < op_2_priority {
                                break;
                            } else if !is_left_asociative && op_priority == op_2_priority {
                                break;
                            }
                        }
                    }

                    ret.push(operators.pop().unwrap());
                }

                operators.push(tok);
            }
            TokenClass::SpecialChar => {
                if open_parenthesis.cmp(&tok) {
                    operators.push(tok);
                } else if close_parenthesis.cmp(&tok) {
                    /*
                    loop {
                        /*
                        let poped: Option<Token> = operators.pop();
                        let v: Token = if poped.is_none() {
                            return Err(String::from("Parenthesis has not been closed. \n"));
                        } else {
                            poped.unwrap()
                        };

                        if open_parenthesis.cmp(&v) {
                            let mut possible_func: bool = false;
                            operators.last().inspect(|&x| {
                                x.class.as_ref().inspect(|&tok_class| match tok_class {
                                    TokenClass::Identifier => possible_func = true,
                                    _ => possible_func = false,
                                });
                            });

                            if possible_func {
                                ret.push(operators.pop().unwrap());
                            }

                            break;
                        }
                        ret.push(v); */

                        match operators.pop() {
                            Some(v) => {
                                if open_parenthesis.cmp(&v) {
                                    let mut possible_func: bool = false;
                                    operators.last().inspect(|&x| {
                                        x.class.as_ref().inspect(|&v| match v {
                                            TokenClass::Identifier => possible_func = true,
                                            _ => possible_func = false,
                                        });
                                    });

                                    if possible_func {
                                        ret.push(operators.pop().unwrap());
                                    }

                                    break;
                                }
                                ret.push(v)
                            }
                            None => {
                                //impossible case, because it should have been checked before
                                return Err(String::from("Parenthesis has not been closed. \n"));
                            }
                        }
                    }
                    */

                    let mut operator_opt: Option<Token>;

                    loop {
                        operator_opt = operators.pop();
                        if operator_opt.is_none() {
                            return Err(String::from("Parenthesis has not been closed. \n"));
                        }

                        let operator: Token = operator_opt.unwrap();

                        if open_parenthesis.cmp(&operator) {
                            //do not push parenthesis
                            break;
                        }

                        ret.push(operator);
                    }

                    let prev_tok_opt: Option<Token> = operators.pop();
                    if let Some(prev_tok) = prev_tok_opt {
                        //let prev_tok: Token = prev_tok_opt.unwrap();

                        if prev_tok.class == TokenClass::Identifier {
                            ret.push(prev_tok);
                        } else {
                            //undo the previous pop
                            operators.push(prev_tok);
                        }
                    }
                } else {
                    //comma (',') is not supported
                    return Err(String::from("There has been an error. Unexpected Special char. Only '(' and ')' are supported. "));
                }
            }

            _ => {
                return Err(format!("Invalid token class. Token: {:#?}", tok));
            }
        }
    }

    //Everything read

    let len: usize = operators.len();
    for _i in 0..len {
        ret.push(operators.pop().unwrap());
    }

    return Ok(ret);
}

/// Substitutes the corresponding substraction [Token] `-` for negarion [Token] `--`
/// when needed.
///
/// The substraction (as in "4-6") and the negation (as in "-(2+6)") share the
/// same symbol, but they operate in a very diferent way. The usual substraction
/// takes 2 inputs but negation only takes 1. To fix this, this function substitutes
/// the [Token] containing a "-" lexemme (string literal) for "--" (wich denotes negation).
pub fn negation_substituter(input: Vec<Token>) -> Vec<Token> {
    /*There is a bug that causes functions in the form 1+f(-x) to give an error
    due to the double function of the - sing. To solve this, we will be substituting
    "-" (substraction, a - b) for "--" (negation, -b = -1 * (b))
    */
    let mut ret: Vec<Token> = Vec::with_capacity(input.len()); //input.len() * 1.125

    let mut prev_token: Token = input[0].clone();

    {
        //if the first token is already a - , change it for a --
        let prev_token_clone: Token = prev_token.clone();
        if prev_token_clone.lexeme.is_some_and(|s| s == "-")
            && prev_token_clone.class == TokenClass::Operator
        {
            prev_token = Token {
                lexeme: Some(String::from("--")),
                class: TokenClass::Operator,
            }
        }
    }

    let open_parenthesis_model: TokenModel = TokenModel {
        token: Token::new(Some(String::from("(")), TokenClass::SpecialChar),
        compare_lexemme: true,
    };
    /*let close_parenthesis_model: TokenModel = TokenModel {
        token: Token::new(Some(String::from(")")), TokenClass::SpecialChar),
        compare_lexemme: true
    };*/
    let minus_model: TokenModel = TokenModel {
        token: Token::new(Some(String::from("-")), TokenClass::Operator),
        compare_lexemme: true,
    };
    let operator_model: TokenModel = TokenModel {
        token: Token::new(None, TokenClass::Operator),
        compare_lexemme: false,
    };
    let negation_model: TokenModel = TokenModel {
        token: Token {
            lexeme: Some(String::from("--")),
            class: TokenClass::Operator,
        },
        compare_lexemme: true,
    };

    for token in input.iter().skip(1) {
        if minus_model.cmp(token) {
            if open_parenthesis_model.cmp(&prev_token) {
                // (-   situation

                ret.push(prev_token); // add (
                prev_token = negation_model.get_token(); // next will be added the --
                continue;
            } else if operator_model.cmp(&prev_token) {
                // <op>-  situation (this ptobably indicates negation of next result (unary operation))

                ret.push(prev_token); // add operator_opt
                prev_token = negation_model.get_token(); // next will be added the --
                continue;
            }
            // ^group with or ???
        }

        ret.push(prev_token);
        prev_token = token.clone();
    }

    ret.push(prev_token); // add final token

    return ret;
}

/// Substitute the literal "x" by a [Token] with the variant [TokenClass::Variable].
pub fn variable_substituter(input: Vec<Token>) -> Vec<Token> {
    let mut ret: Vec<Token> = Vec::with_capacity(input.len());
    let variable_raw_model: TokenModel = TokenModel {
        token: Token {
            lexeme: Some(String::from("x")),
            class: TokenClass::Identifier,
        },
        compare_lexemme: true,
    };

    let variable_model: TokenModel = TokenModel {
        token: Token {
            lexeme: Some(String::from("x")),
            class: TokenClass::Variable,
        },
        compare_lexemme: true,
    };

    for tok in input {
        if variable_raw_model.cmp(&tok) {
            ret.push(variable_model.get_token());
            continue;
        }
        ret.push(tok);
    }

    return ret;
}

/// Detects the identifiers with ´"der"´ and changes the element variant
/// to [Element::Derive].
///
pub fn derive_substituter(input: &mut AST) {
    if let Element::Function(str) = &input.value {
        match str.to_lowercase().as_str() {
            DERIVE_KEYWORD_1 | DERIVE_KEYWORD_2 => {
                input.value = Element::Derive;
            }
            _ => {}
        }
    }

    // Apply recursively to the children.
    input
        .children
        .iter_mut()
        .for_each(|ch| derive_substituter(&mut ch.borrow_mut()));
}

/// Processes the given ´input´ string with the calculator and returns the [AST]
/// with the exprssion it represents or a corresponding error.
/// Set print_messages to true in order to print extra information.
pub fn generate_ast(
    input: String,
    calc: &mut Calculator,
    print_messages: bool,
) -> Result<AST, String> {
    if print_messages {
        println!("");
    }
    if input.len() == 0 {
        return Err("Use this command with an input. ".to_string());
    }

    calc.parser.reset();

    //tokenitzation
    let mut token_list: Vec<Token> = tokenize_input(&mut calc.idfas, &input);

    if print_messages {
        println!("Parsed string. {} tokens detected. ", token_list.len());
    }

    //constant substitution
    {
        token_list = constant_matcher(token_list);
    }

    // parenthesis correcness
    {
        if !setup::correct_parenthesis(&token_list) {
            return Err("The parenthesis  ( ) [ ] { }  are used incorrecly. Some causes may be: \nThey are not closed ( a \'{\' without a \'}\' ) \nThe type of the open one does not match the type of the closing one ( a \'{\' with a \']\'). \nOne closing is found without opening (\'}\' without anything prior). \n".to_string());
        }

        if print_messages {
            println!("Checked use of parenthesis. ");
        }
    }

    // remove redundancy:
    {
        let num_tokens_with_redundancy: usize = token_list.len();
        let mut redundancy_remover: SRA = SRA::new(setup::get_pre_rules());

        for tok in token_list {
            redundancy_remover.advance(tok)?;
        }

        token_list = redundancy_remover.stack;

        if print_messages {
            if num_tokens_with_redundancy != token_list.len() {
                println!(
                    "Removed redundancy. Reduced to {} tokens. ",
                    token_list.len()
                );
            }
        }
    }

    // negation tokens:
    token_list = negation_substituter(token_list);

    // variable detection
    token_list = variable_substituter(token_list);

    // transform to RPN:
    {
        token_list = shunting_yard_algorithm(token_list)?;

        if print_messages {
            println!("Transformed to reverse polish notation. ");
        }

        if print_messages {
            println!("Token_list in RPN: ");
            for token in &token_list {
                println!("\t{:?}", token);
            }
        }
    }

    // Parse:
    {
        for tok in token_list {
            calc.parser.advance(tok)?;
            /*
            let remainings: String = calc.parser.stack.iter().fold(
                String::from("\n\nStack remainings: \n"),
                |acc, x| format!("{acc}{:?}\n", *x),
            );
            println!("{}", remainings);
            */
        }

        if !calc.parser.just_starting_token() {
            let remainings: String = calc.parser.stack.iter().fold(
                String::from("INVALID parsing! Stack remainings: \n"),
                |acc, x| format!("{acc}{:?}\n", *x),
            );
            return Err(remainings);
        }

        if print_messages {
            println!("Valid Parsing! ");
        }

        if calc.parser.ast.len() != 1 {
            return Err("Invalid AST. Possible Parsing Error. \n".to_string());
        }
    }

    if calc.parser.ast.len() != 1 {
        return Err(String::from("AST has non-unit length. "));
    }

    let mut ret: AST = calc.parser.ast.pop().unwrap();

    derive_substituter(&mut ret);

    return Ok(ret);
}

/// evaluate_expression() is the function in charge to process all the information given the input.
///
/// Set print_messages to true in order to print extra information.
/// Will return an error with a string if the process fails.
pub fn evaluate_expression(
    input: String,
    calc: &mut Calculator,
    print_messages: bool,
) -> Result<Number, String> {
    let ast: AST = generate_ast(input, calc, print_messages)?;

    let final_result: Number = ast.evaluate(None)?;
    if print_messages {
        //display result
        println!("\nEvaluated to: \t{:?}\n\n", final_result);
    }

    return Ok(final_result);
}
