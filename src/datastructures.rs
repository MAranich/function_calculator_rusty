use core::fmt;
use integer_sqrt::IntegerSquareRoot;
use rand::Rng;
use std::{cell::RefCell, iter::zip, ops, rc::Rc, vec};

use crate::functions::Functions;

/// Represents the state that all [DFA] will start.
const INITIAL_STATE: u16 = 1;

/// Represents the rejecting state in all [DFA].
const REJECTING_STATE: u16 = 0;

/// When printing a [Number] of the [Number::Rational] variant,
/// if both numbers numbers have an absolute value less than [PRINT_FRACTION_PRECISION_THRESHOLD],
/// it will be printed as a fraction ( numerator/denominator ). Otherwise, a numerical
/// representation will be used.
pub const PRINT_FRACTION_PRECISION_THRESHOLD: i32 = 1024;

/// If printing a number with a fractionary expansion and all the
/// digits are not needed, the result will be truncated to [PRINT_NUMBER_DIGITS] digits.
///
/// Regardless of how they are displayed, all the number are always processed
/// with maximum precision.
pub const PRINT_NUMBER_DIGITS: u32 = 4;

pub const ADD_STR: &'static str = "+";
pub const SUB_STR: &'static str = "-";
pub const MULT_STR: &'static str = "*";
pub const DIV_STR: &'static str = "/";
pub const EXP_STR: &'static str = "^";
pub const FACT_STR: &'static str = "!";
pub const MOD_STR: &'static str = "%";
pub const NEG_STR: &'static str = "--";

/// The tokens are the minimal expression with semantig meaning and the [TokenClass]
#[derive(Debug, PartialEq, Clone)]
pub enum TokenClass {
    ///The lexemme is just a number ("1.9232")
    Number,
    /// The lexemme is an operator ("\*")
    Operator,
    /// The lexemme is a special character ("(")
    SpecialChar,
    /// The lexemme is a function call or a constant ("sin", "PI")  
    Identifier,
    /// The variable to analyze
    Variable,
    /// Used for [Rule] aplications. Non Terminal / Start
    NTStart,
    /// None of the previous
    None,
}
pub const NUM_DFA_CATEGORY_T: usize = 4;

/// A representation of a [Number].
///
/// It can be a rational number (can be expresed as a/b,
/// where b!=0 and a and b are whole numbers). This duality allows to perform some basic
/// operations between real numbers in a fast and exact way while retaining the versatility
/// of the eral numbers when the rationals are not enough.
#[derive(PartialEq, Clone)]
pub enum Number {
    /// A number that cannot be expressed as a/b
    Real(f64),
    /// A rational number.
    Rational(i64, u64),
}

/// The [AST] contains [Element] wich can be any of the following.
#[derive(Debug, PartialEq, Clone)]
pub enum Element {
    Derive,
    Function(String),
    Add,
    Sub,
    Mult,
    Div,
    Exp,
    Fact,
    Mod,
    Number(Number),
    Var,
    Neg,
    None,
}

/// Deterministic Finite Automata [DFA]
///
/// A [DFA] is the basic structure to identify the given strings. We use multiple [DFA] to identify
/// different tokens in out input string. In out case we have a [DFA] for numbers, operators,
/// identifiers (function mnames or constant names) and special characters (`(` and `)`, basically).
///
/// However this structure by itself does not parse anything, it only tells how it should be parsed.
/// The concrete instance in charge of actually identifying strings is the [InstanceDFA].
///
/// Go [here](https://en.wikipedia.org/wiki/Deterministic_finite_automaton) for more inforamtion.
#[derive(Debug, PartialEq)]
pub struct DFA {
    pub num_states: u16,
    pub alphabet: Vec<char>,
    /// In this case we also use an alphabet map that before provessing each character, it is maped to
    /// the corresponding value. This is uscefull because it avoids replication when multiple
    /// characters behave in the exact same way.
    pub alphabet_map: Vec<u16>,
    pub unique_symbols: u16,
    pub ending_states: Vec<u16>,
    pub transition_table: Vec<Vec<u16>>,
    /* Each entry in the tt represents all the transitions for a state. */
}

/// A concrete instance of a [DFA].
///
/// Can parse a string and identrify it using the information given by it's [DFA].
///
/// Contains information about the current state and if it's alive or not, as well as
/// an optional associated [TokenClass].
#[derive(Debug, PartialEq, Clone)]
pub struct InstanceDFA {
    pub reference: Rc<DFA>,
    pub state: u16,
    pub alive: bool,
    /// For processing pruposes we also aded an optional [TokenClass] wich the [InstanceDFA]
    /// can relate to. Therefore, if the string is accedpted it will be of the given [TokenClass].
    pub asociated_class: Option<TokenClass>,
}

/// A [Token] is the minimal lexical unit with sintactical meaning.
///
/// It is the basic processing unit for most of the program.
#[derive(Debug, PartialEq, Clone)]
pub struct Token {
    /// An optional string of the literal text that the token is referring to.
    pub lexeme: Option<String>,
    /// The [TokenClass] that this token is asociated to.
    pub class: TokenClass,
}

/// A variant of [Token] explicitly designed to be compared or cloned.
///
/// It provides utility functions to simplify work while comparing and makes
/// explicit the intention with that [Token].
#[derive(Debug, PartialEq, Clone)]
pub struct TokenModel {
    pub token: Token,
    pub compare_lexemme: bool,
}

/// A parsing [Rule].
///
/// Multiple of them generate languages. In this case, they are used to parse the language
/// with a [SRA]. If the same [Token] in the antecedent are found while parsing, they will
/// be substituted by the consequent.
#[derive(Debug, PartialEq, Clone)]
pub struct Rule {
    /// If these [Token] are found in that order,
    pub antecedent: Vec<TokenModel>,
    /// They can be converted to the following.
    pub consequent: TokenModel,
}

/// Shift Reduce Automata [SRA].
///
/// Reads [Token] and stores them in the stack. If a [Rule] can be used with the last n
/// elements, it is aplied. While apliying a rule, it also creates an [AST] that can be
/// used later.
#[derive(Debug, PartialEq)]
pub struct SRA {
    pub stack: Vec<Token>,
    pub rules: Vec<Rule>,
    pub ast: Vec<AST>,
}

/// Abstract Syntax Tree ([AST]) is a datastructure that encapsulates the
/// meaning of an expression using a tree structure.
///
/// More details [here](https://en.wikipedia.org/wiki/Abstract_syntax_tree).
/// Each leaf contains a number and it's parent contains how to operate
/// between the children. This allows an unambiguous way to evaluate an expression.
#[derive(Debug, PartialEq, Clone)]
pub struct AST {
    pub value: Element,
    pub children: Vec<Rc<RefCell<AST>>>,
}

/// Wrapper that bundles together all the needed datastructures for a simpler
/// execution of the higher level functions.
pub struct Calculator {
    pub dfas: Vec<Rc<DFA>>,
    pub idfas: Vec<InstanceDFA>,
    pub parser: SRA,
}

/// A structure that can be evaluated.
///
/// It returns a Number or a String containing an explanation of the error.
pub trait Evaluable {
    fn evaluate(&mut self, x: Option<Number>) -> Result<Number, String>;
}

/// An [AST] that contains the number 0.
const AST_ZERO: AST = AST {
    value: Element::Number(Number::Rational(0, 1)),
    children: Vec::new(),
};
/// An [AST] that contains the number 1.
const AST_ONE: AST = AST {
    value: Element::Number(Number::Rational(1, 1)),
    children: Vec::new(),
};

impl DFA {
    /// Create a new [DFA].
    ///
    /// The transition table is set to everything going to the rejecting state (0).
    /// The alphabet matp is set to the identity.
    pub fn new(
        _num_states: u16,
        _alphabet: Vec<char>,
        _unique_symbols: u16,
        _ending_states: Vec<u16>,
    ) -> DFA {
        let len: usize = _alphabet.len();
        let mut base_map: Vec<u16> = Vec::with_capacity(len);
        for i in 0..len {
            base_map.push(i as u16 % _unique_symbols);
        }

        let _transition_table: Vec<Vec<u16>> =
            vec![vec![REJECTING_STATE; _unique_symbols as usize]; (_num_states + 1) as usize];

        let new_dfa: DFA = DFA {
            num_states: _num_states,
            alphabet: _alphabet,
            alphabet_map: base_map,
            unique_symbols: _unique_symbols,
            ending_states: _ending_states,
            transition_table: _transition_table,
        };

        return new_dfa;
    }

    /// Assigns a full transition table
    pub fn set_full_transition_table(
        &mut self,
        mut new_transition_table: Vec<Vec<u16>>,
        add_rejecting_row: bool,
    ) -> Result<(), &str> {
        let uniq: u16 = self.unique_symbols;

        if add_rejecting_row {
            new_transition_table.splice(0..0, vec![vec![0 as u16; uniq as usize]]);
        }

        let mut proper_matrix_dimensions: bool =
            new_transition_table.len() as u16 == self.num_states + 1;
        if !proper_matrix_dimensions {
            return Err("Not enough/too many rows for states. ");
        }

        proper_matrix_dimensions = new_transition_table.iter().all(|x| x.len() as u16 == uniq);
        if !proper_matrix_dimensions {
            return Err("Incorrect number of transitions for at least 1 state vector. ");
        }

        let n_states = self.num_states + 1;
        let all_values_valid: bool = new_transition_table.iter().flatten().all(|&x| x < n_states);
        if !all_values_valid {
            return Err("Exists invalid transition. ");
        }

        self.transition_table = new_transition_table;

        return Ok(());
    }

    /// Sets a single element of the transition table.
    pub fn set_transition_table(
        &mut self,
        state: u16,
        uniq_symbol: u16,
        new_state: u16,
    ) -> Result<(), &str> {
        //sets a single element
        match self.transition_table.get_mut(state as usize) {
            Some(row) => match row.get_mut(uniq_symbol as usize) {
                Some(v) => {
                    *v = new_state;
                }
                None => return Err("Invalid uniq_symbol. "),
            },
            None => return Err("State out of bounds. "),
        }

        return Ok(());
    }

    /// Set all of the alphabet map.
    pub fn full_alphabet_map(&mut self, new_map: Vec<u16>) -> Result<(), &str> {
        let len: u16 = self.alphabet.len() as u16;
        if len != new_map.len() as u16 {
            return Err("Invalid_length. ");
        }

        let inside_bounds: bool = new_map.iter().all(|&x| x < len);
        if !inside_bounds {
            return Err("Map points to invalid element. ");
        }

        self.alphabet_map = new_map;

        return Ok(());
    }

    ///Change the whole alphabet map
    pub fn change_map(&mut self, indexes: Vec<u16>, maps: Vec<u16>) -> Result<(), &str> {
        let len: u16 = self.alphabet.len() as u16;
        let all_indexes_valid: bool = indexes.iter().all(|&x| x < len);
        if !all_indexes_valid {
            return Err("Some index is outside of bounds. ");
        }

        let all_maps_are_valid = maps.iter().all(|&x| x < len);
        if !all_maps_are_valid {
            return Err("Some map is outside of bounds. ");
        }

        indexes
            .iter()
            .zip(maps.iter())
            .for_each(|(&i, &m)| self.alphabet_map[i as usize] = m);

        return Ok(());
    }

    /// Sort the alphabet and keep the map in relative order.
    ///
    /// Really just a wrapper of [fn@DFA::heap_sort].
    pub fn sort_alphabet_and_map(&mut self) {
        if self.alphabet.len() != self.alphabet_map.len() {
            panic!("Alphabet and map does not have the same length. ");
        }

        Self::heap_sort(&mut self.alphabet, &mut self.alphabet_map);
    }

    /// Sorts the char_vec while also retaining the correspondencies with map_vec.
    ///
    /// For performance reasons, both the alphabet and the map must be sorted in
    /// ascending order. This is needed to be able to use binary search.
    ///
    /// If in a given index i map_vec\[i\] is translated to char_vec\[i\],
    /// after sorting, this relation is conserved even if i changes.
    /// For more information about heapsort see [here](https://en.wikipedia.org/wiki/Heapsort).
    fn heap_sort(char_vec: &mut Vec<char>, map_vec: &mut Vec<u16>) {
        // implementation source: https://www.geeksforgeeks.org/heap-sort/

        let len: u16 = char_vec.len() as u16;
        for i in (0..len / 2).rev() {
            Self::heapify(char_vec, map_vec, len, i);
        }

        for i in (1..len).rev() {
            char_vec.swap(0, i as usize);
            map_vec.swap(0, i as usize);

            Self::heapify(char_vec, map_vec, i, 0);
        }
    }

    /// Function used in [fn@DFA::heap_sort].
    fn heapify(char_vec: &mut Vec<char>, map_vec: &mut Vec<u16>, len: u16, idx: u16) {
        let length: usize = len as usize;
        let mut i: usize = idx as usize;
        loop {
            let left: usize = 2 * i + 1;
            let right: usize = left + 1;

            let mut largest: usize = i;

            if left < length && char_vec[largest] < char_vec[left] {
                largest = left;
            }

            if right < length && char_vec[largest] < char_vec[right] {
                largest = right;
            }

            if largest != i {
                char_vec.swap(i, largest);
                map_vec.swap(i, largest);
                i = largest;
            } else {
                break;
            }
        }

        /*
                loop {
            let left: u16 = 2 * idx + 1;
            let right: u16 = left + 1;

            let mut largest: u16 = idx;

            if left < len && char_vec[largest as usize] < char_vec[left as usize] {
                largest = left;
            }

            if right < len && char_vec[largest as usize] < char_vec[right as usize] {
                largest = right;
            }

            if largest != idx {
                char_vec.swap(idx as usize, largest as usize);
                map_vec.swap(idx as usize, largest as usize);
                idx = largest;
            } else {
                break;
            }
        }
         */
    }

    /// Using binary search, get the index of the given symbol.
    fn get_index_bs(&self, symbol: &char) -> Option<usize> {
        //source: https://spin.atomicobject.com/learning-rust-binary-search/
        let len: isize = self.alphabet.len() as isize;
        let mut high: isize = len - 1;
        let mut low: isize = 0;
        let mut mid: isize = len >> 1; //len / 2
        let mut current: char;

        while low <= high {
            current = self.alphabet[mid as usize];
            match current.cmp(symbol) {
                std::cmp::Ordering::Equal => return Some(mid as usize),
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid - 1,
            }
            mid = (high + low) >> 1;
        }

        return None;
    }
}

impl InstanceDFA {
    /// Create a new instance of a given [DFA].
    pub fn new(_reference: Rc<DFA>, _asociated_class: Option<TokenClass>) -> Self {
        InstanceDFA {
            reference: _reference,
            state: INITIAL_STATE,
            alive: true,
            asociated_class: _asociated_class,
        }
    }

    /// Advance the [InstanceDFA] by using the given symbol.
    ///
    /// Returns a boolean indicating if the [InstanceDFA] is alive or an error.
    pub fn advance(&mut self, symbol: char) -> Result<bool, String> {
        //self.reference.alphabet.iter().position(|&x| x == symbol);
        //println!("Symbol: {:#?}", symbol);
        let index: usize;
        match self.reference.get_index_bs(&symbol) {
            Some(v) => index = v,
            None => {
                self.alive = false;
                return Err(format!("Symbol |{}| not in alphabet. ", symbol));
            }
        }

        let column: u16 = self.reference.alphabet_map[index]; //assume valid because assume DFA is properly set up

        self.state = self.reference.transition_table[self.state as usize][column as usize];

        self.alive = self.state != REJECTING_STATE;

        return Ok(self.alive);
    }

    /// Reset the [InstanceDFA] so it can be used again as new.
    pub fn reset(&mut self) {
        self.state = INITIAL_STATE;
        self.alive = true;
    }

    /// Determinates if the [DFA] accepts or rejects the given String.
    pub fn finalize(&self) -> bool {
        if !self.alive {
            return false;
        }

        return self
            .reference
            .ending_states
            .iter()
            .any(|&f_state| f_state == self.state);
    }

    /// Analyze a whole String and return if the [InstanceDFA] accepts or not, or an error.
    pub fn analyze(&mut self, input: String) -> Result<bool, String> {
        for c in input.chars() {
            if !self.advance(c)? {
                return Ok(false);
            }
        }
        return Ok(self.finalize());
    }
}

impl Token {
    /// Create a new [Token].
    pub fn new(_lexeme: Option<String>, _class: TokenClass) -> Self {
        Token {
            lexeme: _lexeme,
            class: _class,
        }
    }

    /// Checks if the classes of 2 tokens are the same.
    pub fn class_eq(&self, t: &Token) -> bool {
        return &self.class == &t.class;
    }

    /// Joins the 2 [Token] into one.
    ///
    /// The lexemmes are concatenated and the
    /// class must be the same or one of them must be TokenClass::None, and
    /// the other one will be set.
    pub fn concatenate(&self, t: &Token, new_class: Option<TokenClass>) -> Token {
        let new_lexemme: Option<String>;

        new_lexemme = match &t.lexeme {
            Some(t_lex) => match &self.lexeme {
                Some(s_lex) => Some(s_lex.to_owned() + t_lex),
                None => Some(t_lex.to_owned()),
            },
            None => match &self.lexeme {
                Some(s_lex) => Some(s_lex.to_owned()),
                None => None,
            },
        };

        /*
            if new class is given, use it. Otherwise, if the classes of the
            tokens coincide, use them or use TokenClass::None otherwise.
        */
        let new_class: TokenClass = match new_class {
            Some(class) => class,
            None => {
                if t.class == self.class {
                    self.class.clone()
                } else {
                    TokenClass::None
                }
            }
        };

        let ret: Token = Token::new(new_lexemme, new_class);

        return ret;
    }

    /// Get the precedence and if it is left-asociative.
    ///
    /// If the operator is recognized, returns a tuple containing the precedence of the
    /// operator and a bool determinating if its left-asociative or not. A lower value of
    /// precedence means that it should be evaluated first.
    pub fn get_precedence_operator(&self) -> Option<(i32, bool)> {
        /*
            If valid, returns a tuple with the precedence and if it is left-asociative (Left-to-right).
            If invalid (Lexeme = None or Class is None or distinct from class opertor),
            returns None.

            small values of priority means that the priority is high and
            must be evaluated first. Inspired on the C/C++ standard:
            https://en.wikipedia.org/wiki/Operators_in_C_and_C%2B%2B#Operator_precedence
        */

        if self.lexeme.is_none() {
            return None;
        }

        match self.class {
            TokenClass::Operator => {
                let lex: &str = &self.lexeme.as_ref().unwrap()[..];

                match lex {
                    ADD_STR => return Some((7, true)),
                    SUB_STR => return Some((7, true)),
                    MULT_STR => return Some((6, true)),
                    DIV_STR => return Some((5, true)),
                    EXP_STR => return Some((4, false)),
                    FACT_STR => return Some((2, false)),
                    MOD_STR => return Some((8, true)),
                    NEG_STR => return Some((3, false)),
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
}

impl TokenModel {
    /// Creates a [TokenModel] of the given [Token].
    pub fn from_token(t: Token, _compare_lexemme: bool) -> Self {
        Self {
            token: t,
            compare_lexemme: _compare_lexemme,
        }
    }

    /// Compares a [Token] to itself.
    pub fn cmp(&self, t: &Token) -> bool {
        if !self.token.class_eq(t) {
            return false;
        }

        if self.compare_lexemme {
            return self.token.lexeme == t.lexeme;
        }
        return true;
    }

    /// Gets a clone of the [Token] it contains.
    pub fn get_token(&self) -> Token {
        return self.token.clone();
    }
}

impl Rule {
    /// Returns true if the given vector of [Token] follows the [Rule].
    pub fn follows_rule(&self, challenger: Vec<Token>) -> bool {
        if self.antecedent.len() != challenger.len() {
            return false;
        }

        let is_correct: bool =
            zip(self.antecedent.iter(), challenger.iter()).all(|(model, t)| model.cmp(t));

        return is_correct;
    }
}

impl SRA {
    /// Cretaes a new [SRA].
    pub fn new(_rules: Vec<Rule>) -> Self {
        Self {
            stack: vec![],
            rules: _rules,
            ast: vec![],
        }
    }
    /// Shifts the [SRA]. (Adds a new token)
    fn shift(&mut self, t: Token) -> Result<(), String> {
        self.add_ast(&t)?;
        self.stack.push(t);
        return Ok(());
    }

    /// Reduces the stack with the [Rule] until it can no longer be reduced further.
    fn reduce(&mut self) -> bool {
        for rule in &self.rules {
            let slice_start: isize = self.stack.len() as isize - rule.antecedent.len() as isize;
            if slice_start < 0 {
                continue;
            }

            let scan: bool = rule.follows_rule(self.stack[(slice_start as usize)..].to_vec());
            if scan {
                let keep_items_num: usize = self.stack.len() - rule.antecedent.len();
                let dr = self.stack.drain(keep_items_num..);
                let mut cons_tok: Token = rule.consequent.get_token();
                if cons_tok.lexeme.is_none() {
                    let new_lex = dr.fold(String::from(""), |mut acc: String, t: Token| {
                        match t.lexeme {
                            Some(lex) => {
                                acc.push_str(&lex);
                                acc.push_str(" ");
                            }
                            None => {}
                        }
                        acc
                    });
                    cons_tok.lexeme = Some(new_lex);
                } else {
                    drop(dr);
                }
                self.stack.push(cons_tok);
                self.update_AST(keep_items_num);
                return true;
            }
        }

        return false;
    }

    /// Uses the given token to [fn@SRA::shift] and [fn@SRA::reduce].
    pub fn advance(&mut self, t: Token) -> Result<(), String> {
        self.shift(t)?;

        let mut has_reduced: bool = true;
        while has_reduced {
            has_reduced = self.reduce();
        }

        return Ok(());
    }

    /// Returns true if the stack only contains the staritng [Token].
    ///
    /// Used to determinate if the parsing is successfull.
    pub fn just_starting_token(&self) -> bool {
        if self.stack.len() != 1 {
            return false;
        }

        return self.stack[0].class == TokenClass::NTStart;
    }

    /// Resets the [SRA] so it can be used again.
    pub fn reset(&mut self) {
        self.stack.clear();
        self.ast.clear();
    }

    /// Internal function to add a token to the [AST].
    fn add_ast(&mut self, new_tok: &Token) -> Result<(), String> {
        //let new_tok = self.stack.get(self.stack.len() - 1).unwrap();
        if let TokenClass::None = &new_tok.class {
            self.ast.push(AST::new_empty());
            return Ok(());
        }
        let class: TokenClass = new_tok.class.clone();

        let ret: Element;
        match class {
            TokenClass::Number => {
                let n: f64 = new_tok.lexeme.clone().unwrap().parse::<f64>().unwrap();
                let result: Result<Number, Number> = Number::rationalize(n);
                match result {
                    Ok(v) => ret = Element::Number(v),
                    Err(_) => ret = Element::Number(Number::Real(n)),
                }
            }
            TokenClass::Operator => {
                let lexeme_opt: &Option<String> = &new_tok.lexeme;
                let aux: &String = lexeme_opt.as_ref().unwrap();
                let lexeme_str: &str = aux.as_str();
                match lexeme_str {
                    ADD_STR => {
                        ret = Element::Add;
                    }
                    SUB_STR => {
                        ret = Element::Sub;
                    }
                    MULT_STR => {
                        ret = Element::Mult;
                    }
                    DIV_STR => {
                        ret = Element::Div;
                    }
                    EXP_STR => {
                        ret = Element::Exp;
                    }
                    FACT_STR => {
                        ret = Element::Fact;
                    }
                    MOD_STR => {
                        ret = Element::Mod;
                    }
                    NEG_STR => {
                        ret = Element::Neg;
                    }
                    _ => {
                        return Err(format!(
                            "Invalid operator / operator not supported: {:?}",
                            lexeme_str
                        ));
                    }
                }
            }
            TokenClass::Identifier => {
                ret = Element::Function(new_tok.lexeme.as_ref().unwrap().clone());
            }
            TokenClass::SpecialChar => ret = Element::None,
            TokenClass::NTStart => ret = Element::None,
            TokenClass::Variable => ret = Element::Var,
            TokenClass::None => {
                return Err(format!("Token has Tokenclass::None: {:?}", new_tok));
            }
        }

        self.ast.push(AST::new(ret));
        return Ok(());
    }

    /// Updates the [AST] so it is now correct.
    #[allow(non_snake_case)]
    fn update_AST(&mut self, start_idx: usize) {
        /*Basic implementation: always assumes last element is the relevant operator and uses
        everything else as childs. [start_idx, end_idx). */
        let end_idx: usize = self.ast.len();

        //let values_slice = &self.ast[start_idx..(end_idx - 1)];
        let mut oper_token: AST = self.ast.pop().unwrap();

        let new_childs = self.ast.drain(start_idx..(end_idx - 1));

        new_childs.for_each(|x| oper_token.add_children(x));
        //new_childs.for_each(|x| oper_token.children.push(Rc::new(RefCell::new(x))));

        self.ast.push(oper_token);
    }
}

impl AST {
    /// Creates an empty [AST] with 1 element that contains nothing.
    pub fn new_empty() -> Self {
        Self {
            value: Element::None,
            children: vec![],
        }
    }

    /// Creates an [AST] using the given element.
    pub fn new(expr: Element) -> Self {
        Self {
            value: expr,
            children: vec![],
        }
    }

    /// Creates an [AST] containing the given value with no children.
    ///
    /// Just serves to reduce boilerplate and increase redability.
    pub fn from_number(num: Number) -> Self {
        AST {
            value: Element::Number(num),
            children: Vec::new(),
        }
    }

    /// Adds the given [AST] as a children of Self.
    pub fn add_children(&mut self, child: AST) {
        let ast_ptr: Rc<RefCell<AST>> = Rc::new(RefCell::new(child));
        self.children.push(ast_ptr);
    }

    /// Clears the [AST].
    pub fn set_empty(&mut self) {
        self.children.clear();
        self.value = Element::None;
    }

    /// Returns true if both [AST] contain the same values.
    ///
    /// The order of the child matters even if the operation is commutative.
    /// To fix this, [AST::sort] can be called before.
    pub fn equal(&self, other: &Self) -> bool {
        // check that they have the same value + same number of childs
        if self.value != other.value || self.children.len() != other.children.len() {
            return false;
        }

        // add childs to the stack
        let mut stack: Vec<(Rc<RefCell<AST>>, Rc<RefCell<AST>>)> = Vec::new();
        for (child_self, child_other) in self.children.iter().zip(other.children.iter()) {
            stack.push((Rc::clone(child_self), Rc::clone(&child_other)));
        }

        // repeat the process until no more childs
        while let Some((current_self, current_other)) = stack.pop() {
            if current_self.borrow().value != current_other.borrow().value
                || current_self.borrow().children.len() != current_other.borrow().children.len()
            {
                return false;
            }

            for (child_self, child_other) in current_self
                .borrow()
                .children
                .iter()
                .zip(current_other.borrow().children.iter())
            {
                stack.push((Rc::clone(child_self), Rc::clone(&child_other)));
            }
        }

        true
    }

    /// Sorts the given [AST] in a consistent way
    ///
    /// Rules
    /// 1. If a part of the tree that has the same operation and is commutative has brances with and without
    /// the variable. The ones that contain the variable are going to the index 0 in the list of childs.
    pub fn sort(&mut self) {
        todo!("Implement AST sort");
    }

    /// Determinates if the [AST] contains a variable from this node down.
    ///
    /// If it does not, it can be safely evaluated.
    ///
    /// Only the leafs are checked. If there is a variable outside the leafs,
    /// the [AST] is invalid.  
    pub fn contains_variable(&self) -> bool {
        if self.value == Element::Var {
            // In case this is a leaf, do quick check
            return true;
        }

        let mut nodes: Vec<Rc<RefCell<AST>>> = Vec::new();
        // Put all children in nodes
        for child in &self.children {
            nodes.push(Rc::clone(child));
        }

        while let Some(current_node) = nodes.pop() {
            // current_node = nodes.pop().unwrap();

            let borrow_node: std::cell::Ref<AST> = current_node.borrow();

            if borrow_node.children.len() == 0 {
                //is a leaf
                if current_node.borrow().value == Element::Var {
                    return true; // var found, early return
                }
            } else {
                //is not a leaf, put it's children in nodes
                for child in &borrow_node.children {
                    nodes.push(Rc::clone(child));
                }
            }
        }

        //var not found
        return false;
    }

    // Simplifies the parts of the tree that can be substitutes by the correspondent numerical value.
    /* pub fn simplify_expression(&mut self) -> Result<(), String> {

        //Idea: get the largest tree that does not contain any variable, evaluete it and substitute it.
        // Repeat this until the largest one is a leaf.

        // Idea 2: Get ALL subtrees without variables. Filter out leafs. From the bottom up,
        // evaluate them and substitute them.

        // Idea 3: get a node, call contains_variable(). If true, evaluate and simplify,
        // Otherwise call recursively on the children

        if self.children.is_empty() {
            //node is a leaf. It is already simplified, do early return.
            return Ok(());
        }

        if !Self::contains_variable(self) {
            // Simplify expression
            let result: Number = self.evaluate()?;
            self.value = Element::Number(result);
            self.children.clear();

        } else {

            for child in &mut self.children {
                child.borrow_mut().simplify_expression()?;
            }
        }

        return Ok(());
    }*/

    /// Simplifies the parts of the tree that can be substitutes by the correspondent numerical value.
    ///
    /// If expression contains no variables, call direcly `evaluate()` since it's more efficient.
    /// Will return an error if the expression is not valid (dividing by 0 or by
    /// evaluating a function outside it's domains).
    pub fn simplify_expression(self) -> Result<Self, String> {
        // todo!("Add basic arithmetic simplifications: ")

        let original_node: Rc<RefCell<AST>> = Rc::new(RefCell::new(self));

        let mut stack: Vec<Rc<RefCell<AST>>> = vec![original_node.clone()];

        while let Some(node_rc) = stack.pop() {
            let mut node: std::cell::RefMut<AST> = node_rc.borrow_mut();

            if node.children.is_empty() {
                continue; // Already simplified
            }

            if !Self::contains_variable(&node) {
                // Simplify expression
                let result: Number = node.evaluate(None)?;
                node.value = Element::Number(result);
                node.children.clear();
            } else {
                // Add children to the stack for further processing
                for child in &node.children {
                    stack.push(Rc::clone(child));
                }
            }
        }

        let arithmetical = Rc::try_unwrap(original_node)
            .expect("Failed to unwrap Rc. ")
            .into_inner();

        let ret: AST = arithmetical.simplify_arithmetical()?;

        return Ok(ret);
    }

    /// Simplifies the ast using some basic mathematical identities
    ///
    /// 1)   x +- 0 = x
    /// 2)   x * 0 = 0
    /// 3)   x * 1 = x
    /// 4)   0/x = 0
    /// 5)   x/1 = x
    /// 6)   x/x = 1
    /// 7)   x ^ 1 = x
    /// 8)   x ^ 0 = 1
    /// 10)  1 ^ x = 1
    /// 12)  sqrt(x^(2*a)) = |x|^a    // a is whole
    /// 13)  x + a + x = 2*x + a
    /// 14)  -(-x) = x
    /// Unimplemented: 15) (a/b) / (c/d) = a*d / (b*c)
    ///
    /// Discarded:  9)   0 ^ x = 0       (if x is neg or 0, it does not work)
    /// 11)  x^a * x^b = x^(a+b)         (done in join_terms)
    ///
    fn simplify_arithmetical(self) -> Result<Self, String> {
        // Assumes numerical subtrees has been evaluated. Otherwise call [AST::simplify_expression]

        // 1)   x +- 0 = x
        // 2)   x * 0 = 0
        // 3)   x * 1 = x
        // 4)   0/x = 0
        // 5)   x/1 = x
        // 6)   x/x = 1
        // 7)   x ^ 1 = x
        // 8)   x ^ 0 = 1
        // 10)  1 ^ x = 1
        // 12)  sqrt(x^(2*a)) = |x|^a    // a is whole
        // 13)  x + a + x = 2*x + a
        // 14)  -(-x) = x
        //
        // Discarded:  9)   0 ^ x = 0       (if x is neg or 0, it does not work)
        // 11)  x^a * x^b = x^(a+b)         (done in join_terms)

        let mut rnd: rand::prelude::ThreadRng = rand::thread_rng();
        let call_id: f64 = rnd.gen::<f64>();

        println!("In [{:.4}]: {:?}", call_id, self.to_string());

        let mut ret: AST = match self.value {
            Element::Function(_) => self,
            Element::Add => {
                // 1) x + 0 = x

                let substitute_0: bool =
                    if let Element::Number(is_zero) = &self.children[0].borrow().value {
                        match is_zero {
                            Number::Real(r) => r == &0.0,
                            Number::Rational(n, _) => n == &0,
                        }
                    } else {
                        false
                    };

                let substitute_1: bool =
                    if let Element::Number(is_zero) = &self.children[1].borrow().value {
                        match is_zero {
                            Number::Real(r) => r == &0.0,
                            Number::Rational(n, _) => n == &0,
                        }
                    } else {
                        false
                    };

                match (substitute_0, substitute_1) {
                    (true, true) => AST::from_number(Number::Rational(0, 1)),
                    (true, false) => self.children[1].borrow().deep_copy(),
                    (false, true) => self.children[0].borrow().deep_copy(),
                    (false, false) => self,
                }
            }
            Element::Sub => {
                // 1) x - 0 = x

                let substitute_1: bool =
                    if let Element::Number(is_zero) = &self.children[1].borrow().value {
                        match is_zero {
                            Number::Real(r) => r == &0.0,
                            Number::Rational(n, _) => n == &0,
                        }
                    } else {
                        false
                    };

                if substitute_1 {
                    self.children[0].borrow().deep_copy()
                } else {
                    self
                }
            }
            Element::Mult => 'mult: {
                // 2)   x * 0 = 0
                let set_zero: bool = if self.children[0].borrow().equal(&AST_ZERO) {
                    true
                } else if self.children[1].borrow().equal(&AST_ZERO) {
                    true
                } else {
                    false
                };

                if set_zero {
                    break 'mult AST_ZERO.clone(); // no deep_clone because it has no children
                }

                // 3)   x * 1 = x
                let set_to: Option<usize> = if self.children[0].borrow().equal(&AST_ONE) {
                    // 1 * x
                    Some(1)
                } else if self.children[1].borrow().equal(&AST_ONE) {
                    // x * 1
                    Some(0)
                } else {
                    None
                };

                if let Some(i) = set_to {
                    break 'mult self.children[i].borrow().deep_copy();
                }

                // done in join terms
                /*
                let is_exp_with_same_base: bool = 'exp_check: {
                    if let Element::Exp = self.children[0].borrow().value {
                    } else {
                        break 'exp_check false;
                    }

                    if let Element::Exp = self.children[1].borrow().value {
                    } else {
                        break 'exp_check false;
                    }

                    let base_0_aux: std::cell::Ref<AST> = self.children[0].borrow();
                    let base_1_aux: std::cell::Ref<AST> = self.children[1].borrow();

                    match (base_0_aux.children.get(0), base_1_aux.children.get(0)) {
                        (None, None) => false,
                        (None, Some(_)) => false,
                        (Some(_), None) => false,
                        (Some(b1), Some(b2)) => b1.borrow().equal(&b2.borrow()),
                    }
                };

                if is_exp_with_same_base {
                    // both children are exponents and have the same base
                    // x^a + x^b => x^(a+b)

                    // a+b
                    let base_0_aux: std::cell::Ref<AST> = self.children[0].borrow();
                    let base_1_aux: std::cell::Ref<AST> = self.children[1].borrow();
                    let sum_exp: AST = AST {
                        value: Element::Add,
                        children: vec![
                            Rc::new(RefCell::new(base_0_aux.children[1].borrow().deep_copy())),
                            Rc::new(RefCell::new(base_1_aux.children[1].borrow().deep_copy())),
                        ],
                    };

                    // x^(a+b)
                    let power = AST {
                        value: Element::Exp,
                        children: vec![
                            Rc::new(RefCell::new(base_0_aux.children[0].borrow().deep_copy())),
                            Rc::new(RefCell::new(sum_exp)),
                        ],
                    };

                    break 'mult power;
                }*/

                self
            }
            Element::Div => 'div: {
                // 4)   0/x = 0
                if self.children[0].borrow().equal(&AST_ZERO) {
                    break 'div AST_ZERO.clone(); // no deep_clone because it has no children
                }

                // 5)   x/1 = x
                if self.children[1].borrow().equal(&AST_ONE) {
                    break 'div self.children[0].borrow().deep_copy(); // no deep_clone because it has no children
                }

                // 6)   x/x = 1
                if self.children[0].borrow().equal(&self.children[1].borrow()) {
                    break 'div AST_ONE.clone(); // no deep_clone because it has no children
                }

                self
            }
            Element::Exp => 'exp: {
                // 7)   x ^ 1 = x
                if self.children[1].borrow().equal(&AST_ONE) {
                    break 'exp self.children[0].borrow().deep_copy(); // no deep_clone because it has no children
                }

                // 8)   x ^ 0 = 1
                if self.children[1].borrow().equal(&AST_ZERO) {
                    break 'exp AST_ONE.clone(); // no deep_clone because it has no children
                }
                // 10)  1 ^ x = 1
                if self.children[0].borrow().equal(&AST_ONE) {
                    break 'exp AST_ONE.clone(); // no deep_clone because it has no children
                }

                self
            }
            Element::Neg => {
                // 14)  -(-x) = x
                if self.children[0].borrow().value == Element::Neg {
                    self.children[0].borrow().children[0].borrow().deep_copy()
                } else {
                    self
                }
            }
            _ => {
                // No simplification for:
                //      derive, None, Number, mod, fact, var
                self
            }
        };

        // Do the same recursively for the children

        let updated: Result<Vec<AST>, String> = ret
            .children
            .into_iter()
            .map(|child: Rc<RefCell<AST>>| 'clos: {
                // simplify aritmetically recursively
                let simplified: Result<AST, String> =
                    child.borrow().deep_copy().simplify_arithmetical();
                break 'clos simplified;

                match simplified {
                    Ok(ast) => ast.join_terms(),
                    Err(e) => Err(e),
                }
            })
            .collect();

        ret.children = updated?
            .into_iter()
            .map(|updated| Rc::new(RefCell::new(updated)))
            .collect();

        println!("Out [{:.4}]: {:?}\n", call_id, ret.to_string());

        return Ok(ret);
    }

    /// Joins terms into a simplified version of the [AST]
    ///
    /// This function works when [self] has a value of [Element::Add], [Element::Mult]
    /// and it's children that contain the same element. Then, it performs simplifications:
    /// 1) x + x + ... + x = n * x      //n is whole
    /// 3) x * x * ... * x = n ^ x
    /// To be implemented:
    /// 2) a * x + b * (-x) = (a-b) * x
    /// 4) x^a * x^b * ... * x^n= x^(a+b+ ... +n)
    ///
    /// x can be any expresion, not just the variable. It may also reorder terms
    pub fn join_terms(self) -> Result<Self, String> {
        let (operation, operation_joiner): (Element, Element) = match self.value {
            Element::Add => (Element::Add, Element::Mult),
            Element::Mult => (Element::Mult, Element::Exp),
            _ => return Ok(self),
        };
        // operation is the operation we are working with, operation_joiner is the operator that
        // allows us to join multiple of the same elements into one.

        // childs vec will contain the childs of self with the same element as self
        let mut childs: Vec<Rc<RefCell<AST>>> = Vec::new();

        {
            let mut stack: Vec<Rc<RefCell<AST>>> = vec![Rc::new(RefCell::new(self))];

            while let Some(ast) = stack.pop() {
                if ast.borrow().value == operation {
                    ast.borrow()
                        .children
                        .iter()
                        .for_each(|ch| stack.push(Rc::clone(ch)));
                } else {
                    childs.push(Rc::clone(&ast));
                }
            }
            // assert!(stack.len() == 0);
        }

        // groups will contain each of the AST children and the ammount of times it has been seen.
        let mut groups: Vec<Rc<RefCell<AST>>> = Vec::new();
        while let Some(ast) = childs.pop() {
            let mut counter: u32 = 1; // 1 is ast

            while let Some(other_ast_index) = childs
                .iter()
                .position(|ch| ast.borrow().equal(&ch.borrow()))
            {
                // some other child is exacly the same as the current one
                counter += 1;
                childs.swap_remove(other_ast_index);
            }
            // create an AST that joins the elements accordingly
            groups.push(Rc::new(RefCell::new(AST {
                value: operation_joiner.clone(),
                children: vec![
                    ast,
                    Rc::new(RefCell::new(AST::from_number(Number::Rational(
                        counter as i64,
                        1,
                    )))),
                ],
            })));
        }

        // Now we need to join all the elements into a the AST structure

        let mut base_layer: Vec<Rc<RefCell<AST>>> = groups;
        let mut upper_layer: Vec<Rc<RefCell<AST>>> =
            Vec::with_capacity((base_layer.len() >> 1) + 1);
        let mut missing: Option<Rc<RefCell<AST>>> = None;
        // ^ missing is meeded if the number if elements in base_layer is not even.
        // It acts as a "carry" for the next iteration

        while 1 < base_layer.len() {
            for i in 0..(base_layer.len() >> 1) {
                let left: Rc<RefCell<AST>> = Rc::clone(&base_layer[i * 2]);
                let right: Rc<RefCell<AST>> = Rc::clone(&base_layer[i * 2 + 1]);
                upper_layer.push(Rc::new(RefCell::new(AST {
                    value: operation.clone(),
                    children: vec![left, right],
                })));
            }

            if base_layer.len() & 1 == 1 {
                // = is odd
                missing = match missing {
                    None => Some(base_layer.pop().unwrap()),
                    Some(miss) => {
                        upper_layer.push(Rc::new(RefCell::new(AST {
                            value: operation.clone(),
                            children: vec![miss, base_layer.pop().unwrap()],
                        })));
                        None
                    }
                }
            }

            base_layer.clear();
            base_layer = upper_layer.drain(0..).collect();
            upper_layer.clear();
        }

        assert!(base_layer.len() == 1);

        // if we have a missing carry, ajust
        match missing {
            Some(miss) => {
                let final_elem: Rc<RefCell<AST>> = base_layer.pop().unwrap();

                base_layer.push(Rc::new(RefCell::new(AST {
                    value: operation.clone(),
                    children: vec![miss, final_elem],
                })))
            }
            None => {}
        }

        // Return properly as AST

        let ret: AST = Rc::try_unwrap(base_layer.pop().unwrap())
            .expect("Failed to unwrap Rc pointer in join_terms. ")
            .into_inner();
        Ok(ret)
    }

    /// If the given node is a substraction it changes it to the addition of a negated value
    ///
    /// otherwise does nothing
    fn sub_to_neg(&mut self) {
        if self.value != Element::Neg {
            return;
        }

        let neg: AST = AST {
            value: Element::Neg,
            children: vec![self.children.pop().unwrap()],
        };

        self.value = Element::Add;
        self.children.push(Rc::new(RefCell::new(neg)));
    }

    /// Deep copies the [AST]
    pub fn deep_copy(&self) -> Self {
        let mut childs: Vec<Rc<RefCell<AST>>> = Vec::with_capacity(self.children.len());

        for child in self.children.iter() {
            childs.push(Rc::new(RefCell::new(child.borrow().deep_copy())));
        }

        AST {
            value: self.value.clone(),
            children: childs,
        }
    }
    /*
    pub fn deep_copy(&self) -> Self {
        /* //Recursive:
        let mut childs: Vec<Rc<RefCell<AST>>> = Vec::with_capacity(self.children.len());

        for child in self.children.iter() {
            childs.push(Rc::new(RefCell::new(child.borrow().deep_copy())));
        }

        AST {
            value: self.value.clone(),
            children: childs,
        }

        ////////////

        let ret: AST = AST::new(self.value.clone());
        for child in self.children.iter() {
            ret.add_children(child.borrow().deep_copy())
        }

        return Rc::new(RefCell::new(ret));
        */

        let ret: Rc<RefCell<AST>> = Rc::new(RefCell::new(AST::new(self.value.clone())));

        // Use a queue to manage nodes to be copied
        let mut queue: VecDeque<(Rc<RefCell<AST>>, Rc<RefCell<AST>>)> = VecDeque::new();
        queue.push_back((Rc::clone(&ret), Rc::new(RefCell::new(self.clone()))));

        while let Some((new_node, old_node)) = queue.pop_front() {
            let old_children: Vec<Rc<RefCell<AST>>> = old_node.borrow().children.clone();

            for child in old_children {
                // Create a new child node
                let new_child: Rc<RefCell<AST>> =
                    Rc::new(RefCell::new(AST::new(child.borrow().value.clone())));

                // Add the new child to the current new node's children
                new_node.borrow_mut().children.push(Rc::clone(&new_child));

                // Add the new child and corresponding old child to the queue
                queue.push_back((new_child, child));
            }
        }

        return Rc::try_unwrap(ret)
            .expect("Failed to unwrap Rc. ")
            .into_inner();
    }*/

    pub fn to_string(&self) -> String {
        return match &self.value {
            Element::Derive => format!("der({})", self.children[0].borrow().to_string()),
            Element::Function(iden) => {
                format!("{}({})", iden, self.children[0].borrow().to_string())
            }
            Element::Add => {
                format!(
                    "{}+{}",
                    self.children[0].borrow().to_string(),
                    self.children[1].borrow().to_string()
                )
            }
            Element::Sub => {
                format!(
                    "{}-{}",
                    self.children[0].borrow().to_string(),
                    self.children[1].borrow().to_string()
                )
            }
            Element::Mult => {
                let child_left: std::cell::Ref<AST> = self.children[0].borrow();
                let left_side: String = match child_left.value.clone() {
                    Element::Add => format!("({})", child_left.to_string()),
                    Element::Sub => format!("({})", child_left.to_string()),
                    Element::Number(number) => number.get_numerical().to_string(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => child_left.to_string(), //der, fn, mult, div, exp, fact, mod, neg
                };

                let child_right: std::cell::Ref<AST> = self.children[1].borrow();
                let right_side: String = match child_right.value.clone() {
                    Element::Add => format!("({})", child_right.to_string()),
                    Element::Sub => format!("({})", child_right.to_string()),
                    Element::Number(number) => number.get_numerical().to_string(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => child_right.to_string(), //der, fn, mult, div, exp, fact, mod, neg
                };

                format!("{}*{}", left_side, right_side)
            }
            Element::Div => {
                let child_left: std::cell::Ref<AST> = self.children[0].borrow();
                let numerator: String = match child_left.value.clone() {
                    Element::Add => format!("({})", child_left.to_string()),
                    Element::Sub => format!("({})", child_left.to_string()),
                    Element::Number(number) => number.get_numerical().to_string(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => child_left.to_string(), // der, fn, mult, div, exp, fact, mod, neg
                };

                let child_right: std::cell::Ref<AST> = self.children[1].borrow();
                let denominator: String = match child_right.value.clone() {
                    Element::Derive => child_right.to_string(),
                    Element::Function(_) => child_right.to_string(),
                    Element::Exp => child_right.to_string(),
                    Element::Fact => child_right.to_string(),
                    Element::Mod => child_right.to_string(),
                    Element::Number(number) => number.get_numerical().to_string(),
                    Element::Var => String::from("x"),
                    Element::Neg => child_right.to_string(),
                    Element::None => String::from("None"),
                    _ => format!("({})", child_right.to_string()), // +, -, *, /
                };

                format!("{}/{}", numerator, denominator)
            }
            Element::Exp => {
                let child_left: std::cell::Ref<AST> = self.children[0].borrow();
                let left_side: String = match child_left.value.clone() {
                    Element::Derive => child_left.to_string(),
                    Element::Function(_) => child_left.to_string(),
                    Element::Fact => child_left.to_string(),
                    Element::Number(number) => number.get_numerical().to_string(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => format!("({})", child_left.to_string()),
                };

                let child_right: std::cell::Ref<AST> = self.children[1].borrow();
                let right_side: String = match child_right.value.clone() {
                    Element::Derive => child_right.to_string(),
                    Element::Function(_) => child_left.to_string(),
                    Element::Exp => child_right.to_string(),
                    Element::Fact => child_right.to_string(),
                    Element::Number(number) => number.get_numerical().to_string(),
                    Element::Var => String::from("x"),
                    Element::Neg => child_right.to_string(),
                    Element::None => String::from("None"),
                    _ => format!("({})", child_right.to_string()),
                };

                format!("{}^{}", left_side, right_side)
            }
            Element::Fact => {
                let child: std::cell::Ref<AST> = self.children[0].borrow();
                let left_side: String = match child.value.clone() {
                    Element::Derive => child.to_string(),
                    Element::Function(ident) => format!("{}({})", ident, child.to_string()),
                    Element::Fact => child.to_string(),
                    Element::Number(number) => number.get_numerical().to_string(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => format!("({})", child.to_string()), // +, -, *, /, ^
                };

                format!("{}!", left_side)
            }
            Element::Mod => {
                let child_left: std::cell::Ref<AST> = self.children[0].borrow();
                let left_side: String = match child_left.value.clone() {
                    Element::Number(number) => number.get_numerical().to_string(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => child_left.to_string(),
                };

                let child_right: std::cell::Ref<AST> = self.children[1].borrow();
                let right_side: String = match child_right.value.clone() {
                    Element::Number(number) => number.get_numerical().to_string(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => child_right.to_string(),
                };

                format!("{}%{}", left_side, right_side)
            }
            Element::Number(number) => number.as_str(),
            Element::Var => String::from("x"),
            Element::Neg => {
                let child_left: std::cell::Ref<AST> = self.children[0].borrow();
                let left_side: String = match child_left.value.clone() {
                    Element::Number(number) => number.get_numerical().to_string(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    Element::Add => format!("({})", child_left.to_string()),
                    Element::Sub => format!("({})", child_left.to_string()),
                    Element::Mod => format!("({})", child_left.to_string()),
                    _ => child_left.to_string(),
                };

                format!("-{}", left_side)
            }
            Element::None => String::from("None"),
        };
    }

    /// Derives the contents of the given [AST].
    pub fn derive(&self) -> Result<Self, String> {
        // Derivative rules: https://en.wikipedia.org/wiki/Differentiation_rules

        let ret: AST = match self.value {
            Element::Derive => {
                todo!("No support for 2nd derivatives right now. To be implemented. ")
            }
            //Element::Derive => self.children[0].borrow().derive(),
            Element::Function(_) => {
                //todo!("Use derivative rule for each function. ")}
                Functions::func_derive(self)?
            }
            Element::Add => AST {
                value: Element::Add,
                children: vec![
                    Rc::new(RefCell::new(self.children[0].borrow().derive()?)),
                    Rc::new(RefCell::new(self.children[1].borrow().derive()?)),
                ],
            },
            Element::Sub => AST {
                value: Element::Sub,
                children: vec![
                    Rc::new(RefCell::new(self.children[0].borrow().derive()?)),
                    Rc::new(RefCell::new(self.children[1].borrow().derive()?)),
                ],
            },
            Element::Mult => {
                // (f*g)' = f'*g + g'*f
                // assume only 2 multiplied elements, otherwise invalid AST

                // f'
                let der_0: AST = self.children[0].borrow().derive()?;
                // g'
                let der_1: AST = self.children[1].borrow().derive()?;

                // f'*g
                let prod_0: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(der_0)),
                        self.children.get(0).unwrap().clone(),
                    ],
                };

                // g'*f
                let prod_1: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(der_1)),
                        self.children.get(1).unwrap().clone(),
                    ],
                };

                AST {
                    value: Element::Add,
                    children: vec![Rc::new(RefCell::new(prod_0)), Rc::new(RefCell::new(prod_1))],
                }
            }
            Element::Div => {
                // assume only 2 divided elements, otherwise invalid AST
                // (f/g)' = (f'*g - g'*f)/g^2

                // f'
                let der_0: AST = self.children[0].borrow().derive()?;

                // g'
                let der_1: AST = self.children[1].borrow().derive()?;

                // f'*g
                let prod_0: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(der_0)),
                        self.children.get(0).unwrap().clone(),
                    ],
                };

                // g'*f
                let prod_1: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(der_1)),
                        self.children.get(1).unwrap().clone(),
                    ],
                };

                // f'*g + g'*f
                let numerator: AST = AST {
                    value: Element::Sub,
                    children: vec![Rc::new(RefCell::new(prod_0)), Rc::new(RefCell::new(prod_1))],
                };

                // g^2
                let denominator: AST = AST {
                    value: Element::Exp,
                    children: vec![
                        Rc::new(RefCell::new(self.children[1].borrow().deep_copy())),
                        Rc::new(RefCell::new(AST {
                            value: Element::Number(Number::Rational(2, 1)),
                            children: Vec::new(),
                        })),
                    ],
                };

                AST {
                    value: Element::Div,
                    children: vec![
                        Rc::new(RefCell::new(numerator)),
                        Rc::new(RefCell::new(denominator)),
                    ],
                }
            }
            Element::Exp => {
                // If form f^a => a*f^(a-1) * f'     (a is constant)
                // If form a^f => a^f * ln(a)*f'
                // If form f^g => f^g * (f' * g/f + g' * ln(f))

                let contains_var_0: bool = self.children[0].borrow().contains_variable();
                let contains_var_1: bool = self.children[1].borrow().contains_variable();
                match (contains_var_0, contains_var_1) {
                    (true, true) => {
                        // f^g => f^g * (f' * g/f + g' * ln(f))
                        // oh, boy...

                        // f'
                        let der_0: AST = self.children[0].borrow().derive()?;
                        // g'
                        let der_1: AST = self.children[1].borrow().derive()?;

                        // ln(f)
                        let ln_f: AST = AST {
                            value: Element::Function(String::from("ln")),
                            children: vec![self.children[0].clone()],
                        };

                        // g/f
                        let g_over_f: AST = AST {
                            value: Element::Div,
                            children: vec![self.children[1].clone(), self.children[0].clone()],
                        };

                        // f' * g/f
                        let prod_1: AST = AST {
                            value: Element::Mult,
                            children: vec![
                                Rc::new(RefCell::new(der_0)),
                                Rc::new(RefCell::new(g_over_f)),
                            ],
                        };

                        // g' * ln(f)
                        let prod_2: AST = AST {
                            value: Element::Mult,
                            children: vec![
                                Rc::new(RefCell::new(der_1)),
                                Rc::new(RefCell::new(ln_f)),
                            ],
                        };

                        // f' * g/f + g' * ln(f)
                        let chain_rule_coef: AST = AST {
                            value: Element::Add,
                            children: vec![
                                Rc::new(RefCell::new(prod_1)),
                                Rc::new(RefCell::new(prod_2)),
                            ],
                        };

                        // f^g * (f' * g/f + g' * ln(f))
                        AST {
                            value: Element::Mult,
                            children: vec![
                                Rc::new(RefCell::new(self.clone())),
                                Rc::new(RefCell::new(chain_rule_coef)),
                            ],
                        }
                    }
                    (true, false) => {
                        // f^a => a*f^(a-1) * f'

                        //f'
                        let der: AST = self.children[0].borrow().derive()?;

                        // a
                        let exp: Number = self.children[1].borrow_mut().evaluate(None)?;

                        // a-1
                        let exp_minus_1: Number = exp.clone() - Number::Rational(1, 1);

                        // f^(a-1)
                        let power: AST = AST {
                            value: Element::Exp,
                            children: vec![
                                self.children[0].clone(),
                                Rc::new(RefCell::new(AST {
                                    value: Element::Number(exp_minus_1),
                                    children: Vec::new(),
                                })),
                            ],
                        };

                        // a * f^(a-1)
                        let power_const: AST = AST {
                            value: Element::Mult,
                            children: vec![
                                Rc::new(RefCell::new(AST {
                                    value: Element::Number(exp),
                                    children: Vec::new(),
                                })),
                                Rc::new(RefCell::new(power)),
                            ],
                        };

                        // a*f^(a-1) * f'     (chain rule)
                        AST {
                            value: Element::Mult,
                            children: vec![
                                Rc::new(RefCell::new(power_const)),
                                Rc::new(RefCell::new(der)),
                            ],
                        }
                    }
                    (false, true) => {
                        //a^f => a^f * ln(a)*f'

                        //f'
                        let der: AST = self.children[1].borrow().derive()?;

                        let mut ln_a_numerical: Number =
                            self.children[0].borrow_mut().evaluate(None)?;
                        ln_a_numerical = Functions::find_and_evaluate("ln", ln_a_numerical)?;

                        let ln_a: AST = AST {
                            value: Element::Number(ln_a_numerical),
                            children: Vec::new(),
                        };

                        // ln(a)*f'
                        let der_ln_a: AST = AST {
                            value: Element::Mult,
                            children: vec![Rc::new(RefCell::new(ln_a)), Rc::new(RefCell::new(der))],
                        };

                        // a^f * ln(a)*f'
                        AST {
                            value: Element::Mult,
                            children: vec![
                                Rc::new(RefCell::new(self.clone())),
                                Rc::new(RefCell::new(der_ln_a)),
                            ],
                        }
                    }
                    (false, false) => {
                        //just a constant. The derivative of a constant is 0.
                        AST {
                            value: Element::Number(Number::Rational(0, 1)),
                            children: Vec::new(),
                        }
                    }
                }
            }
            Element::Fact => {
                return Err(String::from(
                    "Derivative of the factorial function is not supported. ",
                ))
            }
            Element::Mod => self.clone(), //just the identity
            Element::Number(_) => AST {
                //derivative of constant is 0
                value: Element::Number(Number::Rational(0, 1)),
                children: Vec::new(),
            },
            Element::Var => AST {
                // derivative of x is 1
                value: Element::Number(Number::Rational(1, 1)),
                children: Vec::new(),
            },
            Element::Neg => {
                let der: AST = self.children[0].borrow().derive()?;

                AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(AST {
                            value: Element::Number(Number::Rational(-1, 1)),
                            children: Vec::new(),
                        })),
                        Rc::new(RefCell::new(der)),
                    ],
                }
            }
            Element::None => return Err(String::from("No derivative of None. ")),
        };

        return Ok(ret);
    }
}

impl Evaluable for AST {
    /// Evaluates the [AST] recursively.
    fn evaluate(&mut self, var_value: Option<Number>) -> Result<Number, String> {
        match &self.value {
            Element::Derive => {
                return Err(String::from(
                    "Cannor evaluate derivative. Derive first and then evaluate. ",
                ))
            }
            Element::Function(name) => {
                return crate::functions::Functions::find_and_evaluate(
                    name.as_str(),
                    (*self.children[0].borrow_mut()).evaluate(var_value)?,
                );
            }
            Element::Add => {
                let mut acc: Number = Number::new_rational(0, 1)?;

                for x in &self.children {
                    acc = (*x.borrow_mut()).evaluate(var_value.clone())? + acc
                }

                Ok(acc)
            }
            Element::Sub => {
                match self.children.len() {
                    1 => {
                        //deprecated case
                        match (*self.children[0].borrow_mut()).evaluate(var_value)? {
                            Number::Real(x) => Ok(Number::new_real(-x)),
                            Number::Rational(n, d) => Ok(Number::new_rational(-n, d)?),
                        }
                    }
                    2 => Ok(
                        (*self.children[0].borrow_mut()).evaluate(var_value.clone())?
                            - (*self.children[1].borrow_mut()).evaluate(var_value)?,
                    ),
                    _ => Err(format!(
                        "Substraction needs exacly 2 arguments, {:?} were provided. \n",
                        self.children.len()
                    )),
                }
            }
            Element::Mult => {
                if self.children.len() < 2 {
                    return Err(format!(
                        "Multiplication needs at least 2 arguments, {:?} were provided. \n",
                        self.children.len()
                    ));
                }

                let mut acc: Number = Number::new_rational(1, 1)?;

                for x in &self.children {
                    acc = (*x.borrow_mut()).evaluate(var_value.clone())? * acc;
                }

                Ok(acc)
            }
            Element::Div => {
                if self.children.len() != 2 {
                    return Err(format!(
                        "Division needs exacly 2 arguments, {:?} were provided. \n",
                        self.children.len()
                    ));
                }

                let inverse: Number = crate::functions::Functions::find_and_evaluate(
                    "inv",
                    (*self.children[1].borrow_mut()).evaluate(var_value.clone())?,
                )?;

                Ok((*self.children[0].borrow_mut()).evaluate(var_value)? * inverse)
            }
            Element::Exp => {
                if self.children.len() != 2 {
                    return Err(format!(
                        "Exponentiation needs exacly 2 arguments, {:?} were provided. \n",
                        self.children.len()
                    ));
                }

                Ok((*self.children[0].borrow_mut())
                    .evaluate(var_value.clone())?
                    .raise_exponent((*self.children[1].borrow_mut()).evaluate(var_value)?)?)
            }
            Element::Fact => {
                if self.children.len() != 1 {
                    return Err(format!(
                        "The factorial takes exacly 1 arguments, {:?} were provided. \n",
                        self.children.len()
                    ));
                }

                let x: Number = (*self.children[0].borrow_mut()).evaluate(var_value.clone())?;
                if !x.is_integer() {
                    return Err(format!(
                        "The factorial only takes integer inputs. Number found: {:?}\n",
                        x
                    ));
                }
                if !x.is_positive() {
                    return Err(format!(
                        "The factorial only takes positive inputs. Number found: {:?}\n",
                        x
                    ));
                }

                match x {
                    Number::Rational(num, _) => {
                        if num == 0 {
                            return Ok(Number::new_rational(1, 1)?);
                        }
                        let mut acc: i64 = 1;
                        for i in 1..=num {
                            acc = acc * i;
                        }

                        Ok(Number::new_rational(acc, 1)?)
                    }
                    _ => Err(format!("Impossible case. Real number factorial. \n")),
                }
            }
            Element::Mod => {
                if self.children.len() != 2 {
                    return Err(format!(
                        "Modulus needs exacly 2 arguments, {:?} were provided. \n",
                        self.children.len()
                    ));
                }

                let x: Number = (*self.children[0].borrow_mut()).evaluate(var_value.clone())?;
                let y: Number = (*self.children[1].borrow_mut()).evaluate(var_value)?;

                if x.is_integer() && y.is_integer() {
                    let x_int: i64 = match x {
                        Number::Rational(num, _) => num,
                        _ => return Err(String::from("Unreachable statement. ")),
                    };

                    let y_int: i64 = match y {
                        Number::Rational(num, _) => num,
                        _ => return Err(String::from("Unreachable statement. ")),
                    };

                    return Ok(Number::new_rational(x_int % y_int, 1)?);
                }

                let x_numerical: f64 = x.get_numerical();
                let y_numerical: f64 = y.get_numerical();

                Ok(Number::new_real(x_numerical % y_numerical))
            }
            Element::Number(x) => Ok(x.clone()),
            Element::Neg => {
                /*negation ( -x ) */
                if self.children.len() != 1 {
                    return Err(format!(
                        "Negation (-x) needs exacly 1 arguments, {:?} were provided. \nParsing went wrong. \n",
                        self.children.len()
                    ));
                }
                let mut ret: Number = (*self.children[0].borrow_mut()).evaluate(var_value)?;

                match &mut ret {
                    Number::Real(r) => *r = -*r,
                    Number::Rational(n, _) => *n = -*n,
                }

                Ok(ret)
            }
            Element::Var => match var_value {
                Some(varibale_value) => Ok(varibale_value),
                None => Err(String::from(
                    "Attempt to evaluate a variable without given value. ",
                )),
            },
            Element::None => Err(String::from("None reached. ")),
        }
    }
}

impl Number {
    /// Creates a new real [Number].
    pub fn new_real(x: f64) -> Self {
        Number::Real(x)
    }

    /// Creates a new rational [Number]. Will fail if den = 0.
    pub fn new_rational(num: i64, den: u64) -> Result<Self, String> {
        if den == 0 {
            return Err(format!("Division by 0 is not possible. \n"));
        }
        return Ok(Number::Rational(num, den));
    }

    /// Takes x as input and tries to see if it can be expressed as a/b without too much error.
    ///
    /// Toleracne: 0.000000001
    /// If it can be converted, it will return a [Number::Rational], otherwise will
    /// return [Number::Rational] with the closest aproximation it can find.
    ///
    /// The [Number] contained in the result will be the same. The only difference is if
    /// it is wrapped arround Ok() (The aproximation is good enough) or
    /// Err() (Aproximation is NOT good enough).
    pub fn rationalize(mut x: f64) -> Result<Number, Number> {
        let tolerance: f64 = 0.000000001;
        let terms: u32 = 22;

        if x == 0.0 {
            return Ok(Number::Rational(0, 1));
        }

        let is_neg: bool = if x < 0.0 {
            x = -x;
            true
        } else {
            false
        };

        let mut n_vals: (f64, f64) = (x, 1.0);
        let mut sequence: Vec<i64> = Vec::new();
        let mut found_0: bool = false;

        for _i in 2..terms {
            let new_term: i64 = (n_vals.0 / n_vals.1) as i64;
            sequence.push(new_term);

            let new_n: f64 = n_vals.0 % n_vals.1;
            if new_n <= tolerance {
                found_0 = true;
                break;
            }
            n_vals = (n_vals.1, new_n);
            //println!("N_val[{}] = {}", i, n_vals.1);
        }

        let mut rational: (i64, i64);
        rational = (sequence.pop().unwrap(), 1);

        sequence.reverse();

        for term in sequence {
            rational = (rational.1, rational.0); //inverse
            rational.0 = rational.0 + rational.1 * term; //add term units to the fraction
                                                         //println!("Rational: {:?}", rational);
        }

        if is_neg {
            rational.0 = -rational.0;
        }
        let aprox: f64 = rational.0 as f64 / rational.1 as f64;
        //println!("x = {} ~= {} / {} = {} \nAbs Err = {}\t\t Rel Err = {}\n\n", x, rational.0, rational.1, aprox, (aprox - x).abs(), (aprox - x).abs() / x);

        if found_0 && (aprox - x).abs() < tolerance {
            return Ok(Number::Rational(rational.0, rational.1 as u64));
        } else {
            return Err(Number::Rational(rational.0, rational.1 as u64));
        }
    }

    /// Use the euclidean algorithm to determinate the larger common divider (lcd)
    /// between x and y.  
    pub fn euclidean_algorithm(mut x: u128, mut y: u128) -> u128 {
        loop {
            if x == 0 {
                return y;
            }

            (x, y) = (y % x, x);
        }
    }

    /// Determinates if the given integer is a perfect squer or not.
    pub fn scan_perfect_square(x: i64) -> bool {
        //Perfec number info: https://en.wikipedia.org/wiki/Square_number

        /*

        // fast method:
        let sqrt: i64 = (x as f64).sqrt().floor() as i64;
        let _is_perf_sq: bool = sqrt * sqrt == i;
        // can fail for 2^52 < x
        */

        if x <= 0 {
            // no negative numbers are perf squares
            // and discard annoying case x = 0, wich is true
            return x == 0;
        }

        // Fact: in binary, after removing an even number of 0 at the end
        // of a perfect square, it's final digits are "001"
        // However some non-perfets square numbers do pass the test
        // accuracy up to 2**36 = 83.33371480111964%
        let mut n: i64 = x;
        while (n & (0b11 as i64)) == 0 {
            //ends with 00
            n = n >> 2;
            // loop must terminate because input contains at least 1 bit set to 1
        }

        if (n & (0b111 as i64)) != 1 {
            return false;
        }

        // Use binary search to find correct result

        // use bounds to aproximate sqrt(x) to shorten binary search iterations
        let log: u32 = n.ilog2();
        let aprox: u64 = 1u64 << (log >> 1);
        let mut left: u64 = aprox - 1;
        let mut right: u64 = aprox * 2 + 1;

        while left <= right {
            let mid: u64 = left + (right - left) / 2;
            let square: u64 = mid * mid;

            match square.cmp(&(n as u64)) {
                std::cmp::Ordering::Equal => return true,
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid - 1,
            }
        }

        return false;
    }

    /// Determinates if the given integer is a perfect squer or not.
    ///
    /// If it is, returns
    /// Some() with the square root as an integer. Otherwise returns None.  
    /// See the implementation of [Number::scan_perfect_square] for the
    /// details on how it works. This is a readapted version of that code.
    pub fn is_perfect_square(x: i64) -> Option<i64> {
        //Perfec number info: https://en.wikipedia.org/wiki/Square_number

        match x.cmp(&0) {
            std::cmp::Ordering::Less => return None,
            std::cmp::Ordering::Equal => return Some(0),
            std::cmp::Ordering::Greater => {}
        }

        let mut n: i64 = x;
        while (n & (0b11 as i64)) == 0 {
            n = n >> 2;
        }

        if (n & (0b111 as i64)) != 1 {
            return None;
        }

        let log: u32 = n.ilog2();
        let aprox: u64 = 1u64 << (log >> 1);
        let mut left: u64 = aprox - 1;
        let mut right: u64 = aprox * 2 + 1;

        while left <= right {
            let mid: u64 = left + (right - left) / 2;
            let square: u64 = mid * mid;

            match square.cmp(&(n as u64)) {
                std::cmp::Ordering::Equal => return Some(mid as i64),
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid - 1,
            }
        }

        return None;

        /*

        //No square ends with the digit 2, 3, 7, or 8.
        let remainder: i64 = x % 10;

        match remainder {
            0 => {}
            1 => {}
            4 => {}
            5 => {}
            6 => {}
            9 => {}
            _ => {
                return None;
            }
        }

        let sqrt: i64 = (x as f64).sqrt().floor() as i64;
        if sqrt * sqrt == x {
            return Some(sqrt);
        }

        return None;

        */
    }

    /// Get the numerical value of Self.
    pub fn get_numerical(&self) -> f64 {
        match self {
            Number::Real(r) => r.clone(),
            Number::Rational(n, d) => (n.clone() as f64) / (d.clone() as f64),
        }
    }

    /// If self is [Number::Rational], tries to reduce num/den to it's irreductible fraction.
    ///
    /// It is usefull to keep the values low, while keeping precision.
    pub fn minimize(&mut self) {
        let mut rationalize: Option<(i64, u64)> = None;

        match self {
            Number::Rational(num, den) => {
                let sign: i64 = num.signum();
                *num *= sign;
                let gcd = Self::euclidean_algorithm(*num as u128, *den as u128) as u64;
                if gcd != 1 {
                    *num = *num / gcd as i64;
                    *den = *den / gcd;
                }
                *num *= sign;
            }
            Number::Real(real) => match Number::rationalize(*real) {
                Ok(new_rational) => {
                    rationalize = match new_rational {
                        Number::Real(_) => {
                            unreachable!("rationalize always returns a Rational variant. ")
                        }
                        Number::Rational(n, d) => Some((n, d)),
                    };
                }
                Err(_failed_rational) => {}
            },
        }

        if let Some((num, den)) = rationalize {
            *self = Number::new_rational(num, den).expect("Denominator should be non-zero. ");
        }
    }

    /// Returns true if the number is an integer.
    pub fn is_integer(&self) -> bool {
        //assume self is altrady minimized. if not / not sure, call self.minimize()
        match self {
            Number::Rational(_, den) => *den == 1,
            _ => false, //TODO: floor(x) == x
        }
    }

    /// Returns true is the number is positive or 0.
    pub fn is_positive(&self) -> bool {
        // 0 is considered positive
        match self {
            Number::Real(x) => 0.0 <= *x,
            Number::Rational(x, _) => 0 <= *x,
        }
    }

    /// Computes self^exponent.
    ///
    /// It tries to keep the result as [Number::Rational],
    /// otherwise, it automatically converts it to [Number::Real]. Attempting to
    /// raise a negative value to a non-integer exponent will result in an Error.
    pub fn raise_exponent(&mut self, mut exponent: Number) -> Result<Number, String> {
        self.minimize(); //just to make sure
        exponent.minimize();

        //println!("{:?} ^ {:?}", self, exponent);

        if !self.is_positive() {
            if !exponent.is_integer() {
                return Err(format!("Cannot raise negative number to a fractioary exponent. \nComplex numbers are not handled. \n"));
            }
        }

        if !exponent.is_positive() {
            //a ^ -b = (1/a) ^ b

            match &mut exponent {
                Number::Real(x) => *x = -1.0 * *x,
                Number::Rational(x, _) => *x = -*x,
            }
            *self = match self {
                Number::Real(r) => Number::Real(1.0 / *r),
                Number::Rational(num, den) => Number::Rational(*den as i64, *num as u64),
            }
        }

        if exponent.is_integer() {
            //square and multiply

            let mut exponent_: u64 = match exponent {
                Number::Rational(num, _) => num as u64,
                _ => panic!("Impossible case. \n"),
            };
            let mut accumulator: Number = self.clone();
            let mut ret: Number = Number::new_rational(1, 1).expect("Non zero div rational");

            while 0 < exponent_ {
                if (exponent_ & 1) == 1 {
                    // multiply
                    //is odd
                    ret = ret * accumulator.clone();
                }

                /*
                accumulator = match accumulator {
                    // square
                    Number::Real(x) => Number::Real(x * x),
                    Number::Rational(num, den) => Number::Rational(num * num, den * den),
                };*/
                accumulator = accumulator.clone() * accumulator;

                exponent_ = exponent_ >> 1; // divide by 2
            }

            return Ok(ret);
        }

        let numerical_base: f64 = self.get_numerical();
        let numerical_exponent: f64 = exponent.get_numerical();

        let result: f64 = numerical_base.powf(numerical_exponent);

        return Ok(Number::new_real(result));
    }

    /// Returns true, if self is within tolerance units of the given [Number].
    pub fn in_tolerance_range(&self, other: &Number, tolerance: f64) -> bool {
        /*Use neg. number or 0 in tolerance to ignore check */

        if tolerance <= 0.0 {
            match (&self, &other) {
                (Number::Real(x), Number::Real(y)) => return x == y,
                (Number::Real(r), Number::Rational(num, den)) => {
                    return *r == (*num as f64 / *den as f64);
                }
                (Number::Rational(num, den), Number::Real(r)) => {
                    return *r == (*num as f64 / *den as f64);
                }
                (Number::Rational(self_num, self_den), Number::Rational(oth_num, oth_den)) => {
                    if self_num == oth_num && self_den == oth_den {
                        return true;
                    }

                    return (*self_num as f64 / *self_den as f64)
                        == (*oth_num as f64 / *oth_den as f64);
                }
            }
        }

        match (&self, &other) {
            (Number::Real(x), Number::Real(y)) => return (x - y).abs() < tolerance,
            (Number::Real(r), Number::Rational(num, den)) => {
                return (*r - (*num as f64 / *den as f64)).abs() < tolerance;
            }
            (Number::Rational(num, den), Number::Real(r)) => {
                return (*r - (*num as f64 / *den as f64)).abs() < tolerance;
            }
            (Number::Rational(self_num, self_den), Number::Rational(oth_num, oth_den)) => {
                if self_num == oth_num && self_den == oth_den {
                    return true;
                }

                return ((*self_num as f64 / *self_den as f64)
                    - (*oth_num as f64 / *oth_den as f64))
                    .abs()
                    < tolerance;
            }
        }
    }

    /// Returns the number as a string.
    pub fn as_str(&self) -> String {
        if let Number::Rational(num, den) = self {
            if num.abs() <= PRINT_FRACTION_PRECISION_THRESHOLD as i64
                && *den < PRINT_FRACTION_PRECISION_THRESHOLD as u64
            {
                // small enough number, print as just num/den
                return if den == &1 {
                    // is integer
                    format!("{}", num)
                } else {
                    format!("{}/{}", num, den)
                };
            }
        }

        let mut full_string: String = self.get_numerical().to_string();

        let mut counter: u32 = 0;
        for (i, c) in full_string.char_indices() {
            if PRINT_NUMBER_DIGITS < counter {
                full_string.truncate(i + 1);
                return full_string;
            }
            if c == '.' || 0 < counter {
                counter = counter + 1;
            }
        }

        //not enough decimal digits, just return the number.
        return full_string;

        //return format!("{:.PRINT_NUMBER_DIGITS} ", self.get_numerical());
    }
}

impl ops::Add<Number> for Number {
    type Output = Number;

    fn add(self, rhs: Number) -> Self::Output {
        match self {
            Number::Real(left_real) => match rhs {
                Number::Real(right_real) => Number::new_real(left_real + right_real),
                Number::Rational(right_num, right_den) => {
                    Number::new_real(left_real + (right_num as f64 / right_den as f64))
                }
            },
            Number::Rational(left_num, left_den) => match rhs {
                Number::Real(right_real) => {
                    Number::new_real(right_real + (left_num as f64 / left_den as f64))
                }
                Number::Rational(right_num, right_den) => {
                    //let num: i64 = right_num * left_den as i64 + left_num * right_den as i64;
                    //let den: u64 = left_den * right_den;

                    //assuming both numbers are already minimized
                    let lcd: u128 =
                        Number::euclidean_algorithm(left_den as u128, right_den as u128);
                    let lcm_res: Result<u64, std::num::TryFromIntError> =
                        u64::try_from((left_den as u128 * right_den as u128) / lcd);

                    if lcm_res.is_err() {
                        // lcm is too large, transform to float and continue
                        let left_number: f64 = left_num as f64 / left_den as f64;
                        let right_number: f64 = right_num as f64 / right_den as f64;

                        return Number::new_real(left_number + right_number);
                    }

                    let lcm: u64 = lcm_res.unwrap();

                    let mult_factor_left: i64 = lcm as i64 / left_den as i64;
                    let mult_factor_right: i64 = lcm as i64 / right_den as i64;

                    let num: i64 = right_num * mult_factor_right + left_num * mult_factor_left;
                    let den: u64 = lcm;

                    let mut new_rational: Number =
                        Number::new_rational(num, den).expect("Non zero div rational");
                    new_rational.minimize();

                    return new_rational;
                }
            },
        }
    }
}

impl ops::Sub<Number> for Number {
    type Output = Number;

    fn sub(self, rhs: Number) -> Self::Output {
        //println!("{:?} - {:?}", self, rhs);

        match self {
            Number::Real(left_real) => match rhs {
                Number::Real(right_real) => Number::new_real(left_real - right_real),
                Number::Rational(right_num, right_den) => {
                    Number::new_real(left_real - (right_num as f64 / right_den as f64))
                }
            },
            Number::Rational(left_num, left_den) => match rhs {
                Number::Real(right_real) => {
                    Number::new_real((left_num as f64 / left_den as f64) - right_real)
                }
                Number::Rational(right_num, right_den) => {
                    //let num: i64 = right_num * left_den as i64 - left_num * right_den as i64;
                    //let den: u64 = left_den * right_den;

                    //assuming both numbers are already minimized
                    let lcd: u128 =
                        Number::euclidean_algorithm(left_den as u128, right_den as u128);
                    let lcm_res: Result<u64, std::num::TryFromIntError> =
                        u64::try_from((left_den as u128 * right_den as u128) / lcd);

                    if lcm_res.is_err() {
                        // lcm is too large, transform to float and continue
                        let left_number: f64 = left_num as f64 / left_den as f64;
                        let right_number: f64 = right_num as f64 / right_den as f64;

                        return Number::new_real(left_number + right_number);
                    }

                    let lcm: u64 = lcm_res.unwrap();

                    let mult_factor_left: i64 = lcm as i64 / left_den as i64;
                    let mult_factor_right: i64 = lcm as i64 / right_den as i64;

                    let num: i64 = left_num * mult_factor_left - right_num * mult_factor_right;
                    let den: u64 = lcm;

                    let mut new_rational: Number =
                        Number::new_rational(num, den).expect("Non zero div rational");
                    new_rational.minimize();

                    return new_rational;
                }
            },
        }
    }
}

impl ops::Mul<Number> for Number {
    type Output = Number;

    fn mul(self, rhs: Number) -> Self::Output {
        //println!("{:?} * {:?}", self, rhs);

        match self {
            Number::Real(left_real) => match rhs {
                Number::Real(right_real) => Number::new_real(left_real * right_real),
                Number::Rational(right_num, right_den) => {
                    Number::new_real(left_real * (right_num as f64 / right_den as f64))
                }
            },
            Number::Rational(left_num, left_den) => match rhs {
                Number::Real(right_real) => {
                    Number::new_real((left_num as f64 / left_den as f64) * right_real)
                }
                Number::Rational(right_num, right_den) => {
                    //handle possible overflow and coerse to rational if so
                    let num_opt: Option<i64> = left_num.checked_mul(right_num);
                    let den_opt: Option<u64> = left_den.checked_mul(right_den);

                    let mut new_rational: Number = match (num_opt, den_opt) {
                        (None, None) => {
                            let num: f64 = left_num as f64 * right_num as f64;
                            let den: f64 = left_den as f64 * right_den as f64;
                            Number::new_real(num / den)
                        }
                        (None, Some(den)) => {
                            let num: f64 = left_num as f64 * right_num as f64;
                            Number::new_real(num / den as f64)
                        }
                        (Some(num), None) => {
                            let den: f64 = left_den as f64 * right_den as f64;
                            Number::new_real(num as f64 / den)
                        }
                        (Some(num), Some(den)) => {
                            Number::new_rational(num, den).expect("Non zero div rational")
                        }
                    };

                    new_rational.minimize();

                    return new_rational;
                }
            },
        }
    }
}

impl fmt::Debug for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Real(r) => write!(f, "{} ", r),
            Number::Rational(num, den) => {
                write!(f, "{}/{} ~= {} ", num, den, *num as f64 / *den as f64)
            }
        }
        //f.debug_struct("Number").field("value", &self.value).finish()
    }
}

impl Evaluable for Number {
    /// Returns the [Number] itself.
    fn evaluate(&mut self, _var_value: Option<Number>) -> Result<Number, String> {
        return Ok(self.clone());
    }
}

impl Calculator {
    /// Bundles the necessary info thogether.
    pub fn new(_dfas: Vec<Rc<DFA>>, _parser: SRA) -> Self {
        let associated_class: Vec<Option<TokenClass>> = vec![
            Some(TokenClass::Number),
            Some(TokenClass::Operator),
            Some(TokenClass::SpecialChar),
            Some(TokenClass::Identifier),
        ];

        let _idfas: Vec<InstanceDFA> = crate::setup::setup_idfas(&_dfas, associated_class);

        Self {
            dfas: _dfas,
            idfas: _idfas,
            parser: _parser,
        }
    }
}
