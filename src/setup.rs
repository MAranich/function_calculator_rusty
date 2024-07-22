use std::rc::Rc;

use crate::datastructures::*;

/// Prepares the [DFA]
pub fn setup_dfas() -> Vec<Rc<DFA>> {
    let mut number_dfa: DFA;
    {
        let alphabet: Vec<char> = vec!['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
        let mapping_vector: Vec<u16> = vec![2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        number_dfa = DFA::new(5, alphabet, 3 as u16, vec![3, 5]);

        number_dfa
            .full_alphabet_map(mapping_vector)
            .expect("Problems with mapping vector. ");
        number_dfa.sort_alphabet_and_map();

        let tt: Vec<Vec<u16>> = vec![
            vec![3, 0, 2],
            vec![3, 0, 0],
            vec![3, 4, 0],
            vec![5, 0, 5],
            vec![5, 0, 0],
        ];
        let res: Result<(), &str> = number_dfa.set_full_transition_table(tt, true);
        match res {
            Err(e) => {
                panic!("{}", e);
            }
            Ok(_) => {}
        }
    }

    let mut operator_dfa: DFA;
    {
        //addition, substraction, multiplication, division, exponentiation, factoria, modulo
        let alphabet: Vec<char> = vec!['+', '-', '*', '/', '^', '!', '%'];
        let len: usize = alphabet.len();
        let mapping_vector: Vec<u16> = vec![0; len];

        operator_dfa = DFA::new(2, alphabet, 1 as u16, vec![2]);

        operator_dfa
            .full_alphabet_map(mapping_vector)
            .expect("Problems with mapping vector. ");
        operator_dfa.sort_alphabet_and_map();

        let tt: Vec<Vec<u16>> = vec![vec![2], vec![0]];
        let res: Result<(), &str> = operator_dfa.set_full_transition_table(tt, true);
        match res {
            Err(e) => {
                panic!("{}", e);
            }
            Ok(_) => {}
        }
    }

    let mut identifier_dfa: DFA;
    {
        let mut alphabet: Vec<char> = vec![
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        ];
        let alph_mayus: Vec<char> = alphabet
            .iter()
            .map(|&x| x.to_uppercase().next().unwrap())
            .collect();
        alphabet = [alphabet, alph_mayus].concat();
        alphabet = [
            vec!['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_'],
            alphabet,
        ]
        .concat();
        let len: usize = alphabet.len();

        let map_vector: Vec<u16> = [vec![0; 10], vec![1; len - 10]].concat();

        identifier_dfa = DFA::new(2, alphabet, 2 as u16, vec![2]);

        identifier_dfa
            .full_alphabet_map(map_vector)
            .expect("Problems with mapping vector. ");

        identifier_dfa.sort_alphabet_and_map();
        let tt: Vec<Vec<u16>> = vec![vec![0, 2], vec![2, 2]];
        let res: Result<(), &str> = identifier_dfa.set_full_transition_table(tt, true);
        match res {
            Err(e) => {
                panic!("{}", e);
            }
            Ok(_) => {}
        }
    }

    let mut special_chars_dfa: DFA;
    {
        let alphabet: Vec<char> = vec!['(', ')', '{', '}', '[', ']'];
        let len: usize = alphabet.len();

        let mapping_vector: Vec<u16> = vec![0; len];

        special_chars_dfa = DFA::new(2, alphabet, 1 as u16, vec![2]);

        special_chars_dfa
            .full_alphabet_map(mapping_vector)
            .expect("Problems with mapping vector. ");

        special_chars_dfa.sort_alphabet_and_map();

        let tt: Vec<Vec<u16>> = vec![vec![2], vec![0]];
        let res: Result<(), &str> = special_chars_dfa.set_full_transition_table(tt, true);
        match res {
            Err(e) => {
                panic!("{}", e);
            }
            Ok(_) => {}
        }
    }

    let mut dfa_vec: Vec<Rc<DFA>> =
        vec![Rc::new(DFA::new(0, vec![], 0, vec![])); NUM_DFA_CATEGORY_T];

    dfa_vec[TokenClass::Number as usize] = Rc::new(number_dfa);
    dfa_vec[TokenClass::Operator as usize] = Rc::new(operator_dfa);
    dfa_vec[TokenClass::SpecialChar as usize] = Rc::new(special_chars_dfa);
    dfa_vec[TokenClass::Identifier as usize] = Rc::new(identifier_dfa);

    return dfa_vec;
}

/// Prepares the [InstanceDFA]. 
pub fn setup_idfas(
    dfas: &Vec<Rc<DFA>>,
    mut associated_class: Vec<Option<TokenClass>>,
) -> Vec<InstanceDFA> {
    let mut ret: Vec<InstanceDFA> = vec![];
    for i in (0..dfas.len()).rev() {
        ret.push(InstanceDFA::new(
            dfas[i].clone(),
            associated_class.pop().unwrap(),
        ));
    }

    ret.reverse();

    return ret;
}

/// Returns true if parenthesis are used correcly. 
/// 
/// The types of parenthesis are `()`, `{}` and `[]`. Each parenthesis must 
/// be matched with one of the same corresponding type. 
pub fn correct_parenthesis(token_list: &Vec<Token>) -> bool {
    let open_parenthesis: TokenModel;
    let close_parenthesis: TokenModel;
    let open_bracket: TokenModel;
    let close_bracket: TokenModel;
    let open_curly_bracket: TokenModel;
    let close_curly_bracket: TokenModel;

    {
        open_parenthesis = TokenModel::from_token(
            Token::new(Some("(".to_string()), TokenClass::SpecialChar),
            true,
        );
        close_parenthesis = TokenModel::from_token(
            Token::new(Some(")".to_string()), TokenClass::SpecialChar),
            true,
        );
        open_bracket = TokenModel::from_token(
            Token::new(Some("[".to_string()), TokenClass::SpecialChar),
            true,
        );
        close_bracket = TokenModel::from_token(
            Token::new(Some("]".to_string()), TokenClass::SpecialChar),
            true,
        );
        open_curly_bracket = TokenModel::from_token(
            Token::new(Some("{".to_string()), TokenClass::SpecialChar),
            true,
        );
        close_curly_bracket = TokenModel::from_token(
            Token::new(Some("}".to_string()), TokenClass::SpecialChar),
            true,
        );
    }

    let mut stack: Vec<&Token> = Vec::new();

    for tok in token_list {
        if open_parenthesis.cmp(tok) {
            stack.push(tok);
        } else if open_bracket.cmp(tok) {
            stack.push(tok);
        } else if open_curly_bracket.cmp(tok) {
            stack.push(tok);
        } else if close_parenthesis.cmp(tok) {
            let prior = stack.pop();
            match prior {
                None => return false,
                Some(pr) => {
                    if !open_parenthesis.cmp(pr) {
                        return false;
                    }
                }
            }
        } else if close_bracket.cmp(tok) {
            let prior = stack.pop();
            match prior {
                None => return false,
                Some(pr) => {
                    if !open_bracket.cmp(pr) {
                        return false;
                    }
                }
            }
        } else if close_curly_bracket.cmp(tok) {
            let prior = stack.pop();
            match prior {
                None => return false,
                Some(pr) => {
                    if !open_curly_bracket.cmp(pr) {
                        return false;
                    }
                }
            }
        }
    }

    return stack.len() == 0;
}


/// Returns preprocessing parser rules. 
/// 
/// The aim if this automata is not to parse anithing, but simplify the
/// redundant expressions if needed. After parsing the input the reduced
/// result will be on the stack.
/// 
/// Rules for a sra to simplify things:
/// 
/// > 1:&emsp;&emsp;`+`&emsp;-> `++`
/// 
/// > 2:&emsp;&emsp;`-`&emsp;-> `+-`
/// 
/// > 3:&emsp;&emsp;`-`&emsp;-> `-+`
/// 
/// > 4:&emsp;&emsp;`+`&emsp;-> `--`
/// 
/// > 5:&emsp;&emsp;`**`&nbsp;&nbsp;&nbsp;-> `^`
/// 
/// > 6:&emsp;&emsp;`[`&emsp;-> `(`
/// 
/// > 7:&emsp;&emsp;`]`&emsp;-> `)`
/// 
/// > 8:&emsp;&emsp;`{`&emsp;-> `(`
/// 
/// > 9:&emsp;&emsp;`}`&emsp;-> `)`
/// 
pub fn get_pre_rules() -> Vec<Rule> {
    /*
    Rules for a sra to simplify things:
        1:      +   -> ++
        2:      -   -> +-
        3:      -   -> -+
        4:      +   -> --
        5:      **  -> ^

        6:      [   -> (
        7:      ]   -> )
        8:      {   -> (
        9:      }   -> )

    The aim if this automata is not to parse anithing, but simplify the
    redundant expressions if needed. After parsing the input the reduced
    result will be on the stack.

    */

    let rule_1: Rule = {
        let t: Token = Token::new(Some("+".to_string()), TokenClass::Operator);

        let t1: Token = Token::new(Some("+".to_string()), TokenClass::Operator);
        let t2: Token = Token::new(Some("+".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, true);

        let a1: TokenModel = TokenModel::from_token(t1, true);
        let a2: TokenModel = TokenModel::from_token(t2, true);

        Rule {
            antecedent: vec![a1, a2],
            consequent: _cons,
        }
    };

    let rule_2: Rule = {
        let t: Token = Token::new(Some("-".to_string()), TokenClass::Operator);

        let t1: Token = Token::new(Some("-".to_string()), TokenClass::Operator);
        let t2: Token = Token::new(Some("+".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, true);

        let a1: TokenModel = TokenModel::from_token(t1, true);
        let a2: TokenModel = TokenModel::from_token(t2, true);

        Rule {
            antecedent: vec![a1, a2],
            consequent: _cons,
        }
    };

    let rule_3: Rule = {
        let t: Token = Token::new(Some("-".to_string()), TokenClass::Operator);

        let t1: Token = Token::new(Some("+".to_string()), TokenClass::Operator);
        let t2: Token = Token::new(Some("-".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, true);

        let a1: TokenModel = TokenModel::from_token(t1, true);
        let a2: TokenModel = TokenModel::from_token(t2, true);

        Rule {
            antecedent: vec![a1, a2],
            consequent: _cons,
        }
    };

    let rule_4: Rule = {
        let t: Token = Token::new(Some("+".to_string()), TokenClass::Operator);

        let t1: Token = Token::new(Some("-".to_string()), TokenClass::Operator);
        let t2: Token = Token::new(Some("-".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, true);

        let a1: TokenModel = TokenModel::from_token(t1, true);
        let a2: TokenModel = TokenModel::from_token(t2, true);

        Rule {
            antecedent: vec![a1, a2],
            consequent: _cons,
        }
    };

    let rule_5: Rule = {
        let t: Token = Token::new(Some("^".to_string()), TokenClass::Operator);

        let t1: Token = Token::new(Some("*".to_string()), TokenClass::Operator);
        let t2: Token = Token::new(Some("*".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, true);

        let a1: TokenModel = TokenModel::from_token(t1, true);
        let a2: TokenModel = TokenModel::from_token(t2, true);

        Rule {
            antecedent: vec![a1, a2],
            consequent: _cons,
        }
    };

    let rule_6: Rule = {
        let t: Token = Token::new(Some("(".to_string()), TokenClass::SpecialChar);

        let t1: Token = Token::new(Some("[".to_string()), TokenClass::SpecialChar);

        let _cons: TokenModel = TokenModel::from_token(t, true);

        let a1: TokenModel = TokenModel::from_token(t1, true);

        Rule {
            antecedent: vec![a1],
            consequent: _cons,
        }
    };

    let rule_7: Rule = {
        let t: Token = Token::new(Some(")".to_string()), TokenClass::SpecialChar);

        let t1: Token = Token::new(Some("]".to_string()), TokenClass::SpecialChar);

        let _cons: TokenModel = TokenModel::from_token(t, true);

        let a1: TokenModel = TokenModel::from_token(t1, true);

        Rule {
            antecedent: vec![a1],
            consequent: _cons,
        }
    };

    let rule_8: Rule = {
        let t: Token = Token::new(Some("(".to_string()), TokenClass::SpecialChar);

        let t1: Token = Token::new(Some("{".to_string()), TokenClass::SpecialChar);

        let _cons: TokenModel = TokenModel::from_token(t, true);

        let a1: TokenModel = TokenModel::from_token(t1, true);

        Rule {
            antecedent: vec![a1],
            consequent: _cons,
        }
    };

    let rule_9: Rule = {
        let t: Token = Token::new(Some(")".to_string()), TokenClass::SpecialChar);

        let t1: Token = Token::new(Some("}".to_string()), TokenClass::SpecialChar);

        let _cons: TokenModel = TokenModel::from_token(t, true);

        let a1: TokenModel = TokenModel::from_token(t1, true);

        Rule {
            antecedent: vec![a1],
            consequent: _cons,
        }
    };

    return vec![
        rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8, rule_9,
    ];
}

/// Get the rules of the language. 
///
/// > 1:&emsp;`S` -> `S S +`
/// 
/// > 2:&emsp;`S` -> `S S -`
/// 
/// > 3:&emsp;`S` -> `S S *`
/// 
/// > 4:&emsp;`S` -> `S S /`
/// 
/// > 5:&emsp;`S` -> `S S ^`
/// 
/// > 6:&emsp;`S` -> `S !`
/// 
/// > 7:&emsp;`S` -> `S S %`
/// 
/// > 8:&emsp;`S` -> `S iden`
/// 
/// > 9:&emsp;`S` -> `num`
/// 
/// REMOVED: 10: `S` -> `S -` 
/// 
/// > 11: `S` -> `S -- `
/// 
/// > 12: `S` -> `var`
///
/// Order: 9, 8, 1, 2, 11, 3, 4, 5, 6, 7, 12
/// 
/// Use revers polish notation to avoid ambiguity:
/// Source: [link](https://en.wikipedia.org/wiki/Reverse_Polish_notation)
pub fn get_rules() -> Vec<Rule> {
    /*
        Use revers polish notation to avoid ambiguity:
        Source: https://en.wikipedia.org/wiki/Reverse_Polish_notation

        1:  S -> S S +
        2:  S -> S S -
        3:  S -> S S *
        4:  S -> S S /
        5:  S -> S S ^
        6:  S -> S !
        7:  S -> S S %
        8:  S -> S iden
        9:  S -> num
        REMOVED: 10: S -> S - 
        11: S -> S -- 
        12: S -> var
        
        Order: 9, 8, 1, 2, 11, 3, 4, 5, 6, 7,
        Old Order: 9, 8, 1, 2, 10, 11, 3, 4, 5, 6, 7,

    */

    let rule_1: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::NTStart);
        let t2: Token = Token::new(None, TokenClass::NTStart);
        let t3: Token = Token::new(Some("+".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        let a2: TokenModel = TokenModel::from_token(t2, false);
        let a3: TokenModel = TokenModel::from_token(t3, true);
        Rule {
            antecedent: vec![a1, a2, a3],
            consequent: _cons,
        }
    };

    let rule_2: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::NTStart);
        let t2: Token = Token::new(None, TokenClass::NTStart);
        let t3: Token = Token::new(Some("-".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        let a2: TokenModel = TokenModel::from_token(t2, false);
        let a3: TokenModel = TokenModel::from_token(t3, true);
        Rule {
            antecedent: vec![a1, a2, a3],
            consequent: _cons,
        }
    };

    let rule_3: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::NTStart);
        let t2: Token = Token::new(None, TokenClass::NTStart);
        let t3: Token = Token::new(Some("*".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        let a2: TokenModel = TokenModel::from_token(t2, false);
        let a3: TokenModel = TokenModel::from_token(t3, true);
        Rule {
            antecedent: vec![a1, a2, a3],
            consequent: _cons,
        }
    };

    let rule_4: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::NTStart);
        let t2: Token = Token::new(None, TokenClass::NTStart);
        let t3: Token = Token::new(Some("/".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        let a2: TokenModel = TokenModel::from_token(t2, false);
        let a3: TokenModel = TokenModel::from_token(t3, true);
        Rule {
            antecedent: vec![a1, a2, a3],
            consequent: _cons,
        }
    };

    let rule_5: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::NTStart);
        let t2: Token = Token::new(None, TokenClass::NTStart);
        let t3: Token = Token::new(Some("^".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        let a2: TokenModel = TokenModel::from_token(t2, false);
        let a3: TokenModel = TokenModel::from_token(t3, true);
        Rule {
            antecedent: vec![a1, a2, a3],
            consequent: _cons,
        }
    };

    let rule_6: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::NTStart);
        let t2: Token = Token::new(Some("!".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        let a2: TokenModel = TokenModel::from_token(t2, true);
        Rule {
            antecedent: vec![a1, a2],
            consequent: _cons,
        }
    };

    let rule_7: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::NTStart);
        let t2: Token = Token::new(None, TokenClass::NTStart);
        let t3: Token = Token::new(Some("%".to_string()), TokenClass::Operator);

        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        let a2: TokenModel = TokenModel::from_token(t2, false);
        let a3: TokenModel = TokenModel::from_token(t3, true);
        Rule {
            antecedent: vec![a1, a2, a3],
            consequent: _cons,
        }
    };

    let rule_8: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::NTStart);
        let t2: Token = Token::new(None, TokenClass::Identifier);

        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        let a2: TokenModel = TokenModel::from_token(t2, false);
        Rule {
            antecedent: vec![a1, a2],
            consequent: _cons,
        }
    };

    let rule_9: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::Number);

        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        Rule {
            antecedent: vec![a1],
            consequent: _cons,
        }
    };

    /* 
    let rule_10: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::NTStart);
        let t2: Token = Token::new(Some("-".to_string()), TokenClass::Operator);


        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        let a2: TokenModel = TokenModel::from_token(t2, true);
        Rule {
            antecedent: vec![a1, a2],
            consequent: _cons,
        }
    };
    */

    let rule_11: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::NTStart);
        let t2: Token = Token::new(Some("--".to_string()), TokenClass::Operator);


        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        let a2: TokenModel = TokenModel::from_token(t2, true);
        Rule {
            antecedent: vec![a1, a2],
            consequent: _cons,
        }
    };

    let rule_12: Rule = {
        let t: Token = Token::new(None, TokenClass::NTStart);

        let t1: Token = Token::new(None, TokenClass::Variable);

        let _cons: TokenModel = TokenModel::from_token(t, false);

        let a1: TokenModel = TokenModel::from_token(t1, false);
        Rule {
            antecedent: vec![a1],
            consequent: _cons,
        }
    };

    return vec![
        rule_9, rule_8, rule_1, rule_2, rule_11, rule_3, rule_4, rule_5, rule_6, rule_7, rule_12
    ];
}
