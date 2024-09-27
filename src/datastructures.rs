use core::fmt;
use rand::Rng;
use std::{
    cell::{Ref, RefCell},
    iter::zip,
    ops,
    rc::Rc,
    vec,
};

use crate::{
    functions::{self, FnIdentifier, Functions},
    get_ptr,
};

pub static mut NUMERICAL_OUTPUTS: bool = false;

//pub static NUMERICAL_OUTPUTS: LazyLock<bool> = LazyLock::new(|| {false});

// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
/// It can be a rational number [Number::Rational] (can be expresed as a/b,
/// where b!=0 and a and b are whole numbers) or a real number [Number::Real]
/// (to cover all other edge cases). This duality allows to perform some basic
/// operations between real numbers in a fast and exact way while retaining the versatility
/// of the real numbers when the rationals are not enough.
///
/// Note that for any [Number::Rational], there is the invariant that the
/// denominator (the second number, the u64) is never 0. If you want to create
/// a rational number that you do not know if it has a 0 as denominator or not,
/// use [Number::new_rational].
#[derive(Clone)]
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
    Function(FnIdentifier),
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
    fn evaluate(&self, x: Option<Number>) -> Result<Number, String>;
}

/// An [AST] that contains the number 0.
pub const AST_ZERO: AST = AST {
    value: Element::Number(Number::Rational(0, 1)),
    children: Vec::new(),
};
/// An [AST] that contains the number 1.
pub const AST_ONE: AST = AST {
    value: Element::Number(Number::Rational(1, 1)),
    children: Vec::new(),
};

/// An [AST] that contains the variable.
pub const AST_VAR: AST = AST {
    value: Element::Var,
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
                let lexemme = new_tok.lexeme.as_ref().unwrap(); 
                if lexemme == "x" {
                    ret = Element::Var; 
                }
                else {
                    ret = Element::Function(FnIdentifier::from_str(lexemme)?);
                }
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

        let new_childs: vec::Drain<'_, AST> = self.ast.drain(start_idx..(end_idx - 1));

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
    ///
    /// **IMPORTANT**: this does **NOT** check if the 2 [AST] are mathematically equivalent but
    /// written in a different form. It just checks for *exact* equivalence.
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

            let borrow_node: Ref<'_, AST> = current_node.borrow();

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

    /// Simplifies the parts of the tree that can be substitutes by the correspondent [Number] value.
    ///
    /// If expression contains no variables, call direcly [AST::evaluate] since it's more efficient.
    /// Will return an error if the expression is not valid (dividing by 0 or by
    /// evaluating a function outside it's domains).
    pub fn partial_evaluation(self) -> Result<Self, String> {
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

        let ret: AST = Rc::try_unwrap(original_node)
            .expect("Failed to unwrap Rc. ")
            .into_inner();

        return Ok(ret);
    }

    /// Simplifies the ast using some basic mathematical identities:
    ///
    /// 1)      x +- 0 = x
    /// 2)      x * 0 = 0
    /// 3)      x * 1 = x
    /// 4)      0/x = 0
    /// 5)      x/1 = x
    /// 6)      x/x = 1
    /// 7)      x ^ 1 = x
    /// 8)      x ^ 0 = 1
    /// 10)     1 ^ x = 1
    /// 13)     x + a + x = 2*x + a
    /// 14)     -(-x) = x
    /// 15.1)   (a/b) / (c/d) = a*d / (b*c)
    /// 15.2)   (a/b) / c = a / (b*c)
    /// 15.3)   a / (b/c) = a*c / b
    /// 16)     (x * y)^a = x^a * y^a
    /// 17)     (x / y)^a = x^a / y^a
    /// 19)     (a^b)^c = a^(b*c)
    /// 20.1)     a * 1/b = a/b    
    /// 20.2)     a * (c/b) = (a*c)/b       (where c is a constant (number))
    /// 23)     -1*x = -x
    /// 24)     c * -x = (-c)*x         // constant mult. by neg. expression => opposite constant * expression
    /// 25)     -(c) = (-c)             // negated constants become constants with the oposite value.  
    /// Unimplemented:
    /// 12)     sqrt(x^(2*a)) = |x|^a    // a is whole
    /// 18)     x^-a = 1/x^a
    /// 21)     (a-b)*-c = c * (b - a)
    /// 22)     0-x = -x
    ///
    ///
    /// Discarded:  
    /// 9)   0 ^ x = 0                  (if x is neg or 0, it does not work; maybe retrun err ???)
    /// 11)  x^a * x^b = x^(a+b)        (done in join_terms)
    ///
    /// Assumes all numerical subtrees have been evaluated
    /// (parts of the tree that do not depend on the variable (x)).
    /// Otherwise call [AST::partial_evaluation] first.
    pub fn simplify_expression(self) -> Result<Self, String> {
        /*
        Idea: we have 2 instances of the AST, prev (previous) and cons (consequent).
        If prev != cons then some changes have been made, wich means that there
        could be more changes to do (perhaps unlocked by the previous simplifications).
        So we discard prev and set prev = cons. Then we compute cons = prev.simplify_step()
        and repeat.

        If there are no changes that means that the expression could not be simplified more
        with our rules, therefore we can return. We will also include a security term
        that counts the number of iterations so we don't end up iterating infinitely.

        */

        #[allow(non_snake_case)]
        let MAX_NUMBER_OF_ITERS: i32 = 32;

        let mut previous: AST = AST_ZERO.clone(); //filler just to enter loop
        let mut consequent: AST = self;
        let mut i: i32 = 0;

        while !previous.eq(&consequent) {
            previous = consequent;

            consequent = previous.deep_copy().simplify_step()?;

            i = i + 1;
            if MAX_NUMBER_OF_ITERS <= i {
                return Err(String::from(
                    "Possibly infinite simplification recursion. Recheck implementation rules. ",
                ));
                //return Ok(consequent);
            }
        }

        return Ok(consequent);
    }

    /// Inner function in charge of actually simplifying the given expression.
    fn simplify_step(mut self) -> Result<Self, String> {
        /*

        Idea: if the current [AST] has a form that can be simplified according to our rules, do it.
        Then apply single step recursively. We will first match into the [Element] of self
        and then see if some rule matches for it.

         */

        // Debugging tools:
        #[allow(non_snake_case)]
        let PRINT_DGB_STATEMENTS: bool = false;
        let mut rnd: rand::prelude::ThreadRng = rand::thread_rng();
        let call_id: f64 = rnd.gen::<f64>();

        if PRINT_DGB_STATEMENTS {
            println!("In [{:.4}]: {:?}", call_id, self.to_string());
        }

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

                // 20.1)     a * 1/b = a/b
                // 20.2)     a * (c/b) = (a*c)/b       (where c is a constant (number))

                // first determine if either the left or right term have the form c / a,
                // where a is anything valid and c is a constant.

                let is_left_inverse: Option<Number> = {
                    if self.children[0].borrow().value == Element::Div {
                        if let Element::Number(n) =
                            self.children[0].borrow().children[0].borrow().value.clone()
                        {
                            Some(n)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                let is_right_inverse: Option<Number> = {
                    if self.children[1].borrow().value == Element::Div {
                        if let Element::Number(n) =
                            self.children[1].borrow().children[0].borrow().value.clone()
                        {
                            Some(n)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                match (is_left_inverse, is_right_inverse) {
                    (Some(n1), Some(n2)) => {
                        let numertator: Number = n1 * n2;

                        let denominator: AST = AST {
                            value: Element::Mult,
                            children: vec![
                                Rc::clone(&self.children[0].borrow().children[1]),
                                Rc::clone(&self.children[1].borrow().children[1]),
                            ],
                        };

                        self.value = Element::Div;
                        self.children =
                            vec![get_ptr(AST::from_number(numertator)), get_ptr(denominator)];
                        break 'mult self;
                    }
                    (None, Some(num)) => {
                        // a * (c/b), c is constant => (c*a)/b

                        if num.eq(&Number::Rational(1, 1)) {
                            // no need for multiplication by 1. Just return a/b
                            let numerator: Rc<RefCell<AST>> = Rc::clone(&self.children[0]);
                            let denominator: Rc<RefCell<AST>> =
                                Rc::clone(&self.children[1].borrow().children[1]);

                            self.value = Element::Div;
                            self.children = vec![numerator, denominator];

                            break 'mult self;
                        }

                        // otherwise do the general case

                        let numerator: AST = AST {
                            value: Element::Mult,
                            children: vec![
                                get_ptr(AST::from_number(num)),
                                Rc::clone(&self.children[0]),
                            ],
                        };

                        let denominator: Rc<RefCell<AST>> =
                            Rc::clone(&self.children[1].borrow().children[1]);

                        self.value = Element::Div;
                        self.children = vec![get_ptr(numerator), denominator];
                        break 'mult self;
                    }
                    (Some(num), None) => {
                        // c/a * b, c is constant => (c*b)/a

                        if num.eq(&Number::Rational(1, 1)) {
                            // no need for multiplication by 1. Just return a/b
                            let numerator: Rc<RefCell<AST>> = Rc::clone(&self.children[1]);
                            let denominator: Rc<RefCell<AST>> =
                                Rc::clone(&self.children[0].borrow().children[1]);

                            self.value = Element::Div;
                            self.children = vec![numerator, denominator];

                            break 'mult self;
                        }

                        let numerator: AST = AST {
                            value: Element::Mult,
                            children: vec![
                                get_ptr(AST::from_number(num)),
                                Rc::clone(&self.children[1]),
                            ],
                        };

                        let denominator: Rc<RefCell<AST>> =
                            Rc::clone(&self.children[0].borrow().children[1]);

                        self.value = Element::Div;
                        self.children = vec![get_ptr(numerator), denominator];

                        break 'mult self;
                    }
                    _ => {}
                }

                // 23)     -1*x = -x

                if self.mult_neg1_to_neg() {
                    // if a change was performed, exit.
                    break 'mult self;
                }

                // 24)     c * -x = (-c)*x         // constant mult. by neg. expression => opposite constant * expression
                {
                    if self.children[0].borrow().value == Element::Neg {
                        let maybe_constant: Element = self.children[1].borrow().value.clone();
                        if let Element::Number(constant) = maybe_constant {
                            let new_constant: Number = match constant {
                                Number::Real(r) => Number::Real(-r),
                                Number::Rational(n, d) => Number::Rational(-n, d),
                            };

                            // the expression inside the negation.
                            let other_expression: Rc<RefCell<AST>> =
                                Rc::clone(&self.children[0].borrow().children[0]);

                            self.children =
                                vec![get_ptr(AST::from_number(new_constant)), other_expression];
                        }
                    } else {
                        if self.children[1].borrow().value == Element::Neg {
                            let maybe_constant = self.children[0].borrow().value.clone();
                            if let Element::Number(constant) = maybe_constant {
                                let new_constant: Number = match constant {
                                    Number::Real(r) => Number::Real(-r),
                                    Number::Rational(n, d) => Number::Rational(-n, d),
                                };

                                // the expression inside the negation.
                                let other_expression: Rc<RefCell<AST>> =
                                    Rc::clone(&self.children[1].borrow().children[0]);

                                self.children =
                                    vec![get_ptr(AST::from_number(new_constant)), other_expression];
                            }
                        }
                    }
                }

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

                // 17)     x^-a = 1/x^a
                // If `-a` is a negative number
                {
                    //let power: &Element = &self.children[1].borrow().value;
                    let negative_exponent: Option<Number> =
                        if let Element::Number(num) = &self.children[1].borrow().value {
                            if !num.is_positive() {
                                let neg_num: Number = match num {
                                    Number::Real(r) => Number::Real(-*r),
                                    Number::Rational(n, d) => Number::Rational(-*n, *d),
                                };
                                Some(neg_num)
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                    if let Some(num) = negative_exponent {
                        //set exp to the negative of what it was (now positive)
                        assert!(num.is_positive());
                        self.children[1].borrow_mut().value = Element::Number(num);
                        // 1/b^(-e)
                        let invert: AST = AST {
                            value: Element::Div,
                            children: vec![get_ptr(AST_ONE.clone()), get_ptr(self)],
                        };
                        break 'div invert;
                    }
                    /*
                    match negative_exponent {
                        Some(num) => {
                            //set exp to the negative of what it was (now positive)
                            assert!(num.is_positive());
                            self.children[1].borrow_mut().value = Element::Number(num);
                            // 1/b^(-e)
                            let invert: AST = AST {
                                value: Element::Div,
                                children: vec![get_ptr(AST_ONE.clone()), get_ptr(self)],
                            };
                            break 'div invert;
                        }
                        None => {
                            // Nothing to do. This is just for the borrow checker.
                        }
                    }*/
                }

                // 17)     x^-a = 1/x^a     (v2)
                // if `-a` is a negation of something
                {
                    if self.children[1].borrow().value == Element::Neg {
                        // get reference to the exponent (minus the negation)
                        let ptr_exp: Rc<RefCell<AST>> =
                            Rc::clone(&self.children[1].borrow_mut().children[0]);

                        //drop negation and put the positive exponet
                        self.children[1] = ptr_exp;

                        // 1/ self^-x
                        let invert: AST = AST {
                            value: Element::Div,
                            children: vec![get_ptr(AST_ONE.clone()), get_ptr(self)],
                        };
                        break 'div invert;
                    }
                }

                // 15) (a/b) / (c/d) = a*d / (b*c)  (+variants with only 1 nested division)
                let division_in_numerator: bool = self.children[0].borrow().value == Element::Div;
                let division_in_denominator: bool = self.children[1].borrow().value == Element::Div;

                match (division_in_numerator, division_in_denominator) {
                    // 15.1) (a/b) / (c/d) = a*d / (b*c)
                    (true, true) => {
                        let a: Rc<RefCell<AST>> = Rc::clone(&self.children[0].borrow().children[0]);
                        let b: Rc<RefCell<AST>> = Rc::clone(&self.children[0].borrow().children[1]);
                        let c: Rc<RefCell<AST>> = Rc::clone(&self.children[1].borrow().children[0]);
                        let d: Rc<RefCell<AST>> = Rc::clone(&self.children[1].borrow().children[1]);

                        let num: AST = AST {
                            value: Element::Mult,
                            children: vec![a, d],
                        };

                        let den: AST = AST {
                            value: Element::Mult,
                            children: vec![b, c],
                        };

                        AST {
                            value: Element::Div,
                            children: vec![Rc::new(RefCell::new(num)), Rc::new(RefCell::new(den))],
                        }
                    }
                    // 15.2) (a/b) / c = a / (b*c)
                    (true, false) => {
                        let a: Rc<RefCell<AST>> = Rc::clone(&self.children[0].borrow().children[0]);
                        let b: Rc<RefCell<AST>> = Rc::clone(&self.children[0].borrow().children[1]);
                        let c: Rc<RefCell<AST>> = Rc::clone(&self.children[1]);

                        let den: AST = AST {
                            value: Element::Mult,
                            children: vec![b, c],
                        };

                        AST {
                            value: Element::Div,
                            children: vec![a, Rc::new(RefCell::new(den))],
                        }
                    }
                    // 15.3) a / (b/c) = a*c / b
                    (false, true) => {
                        let a: Rc<RefCell<AST>> = Rc::clone(&self.children[0]);
                        let b: Rc<RefCell<AST>> = Rc::clone(&self.children[1].borrow().children[0]);
                        let c: Rc<RefCell<AST>> = Rc::clone(&self.children[1].borrow().children[1]);

                        let num: AST = AST {
                            value: Element::Mult,
                            children: vec![a, c],
                        };

                        AST {
                            value: Element::Div,
                            children: vec![Rc::new(RefCell::new(num)), b],
                        }
                    }
                    (false, false) => self,
                }
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

                // 16)     (x * y)^a = x^a * y^a
                // 17)     (x / y)^a = x^a / y^a

                // If the exponent is simple
                let exponent_val: Element = self.children[1].borrow().value.clone();
                if let Element::Number(exp) = exponent_val {
                    let base_opertation: Element = self.children[0].borrow().value.clone();
                    if base_opertation == Element::Mult || base_opertation == Element::Div {
                        let expression_0: Rc<RefCell<AST>> =
                            Rc::clone(&self.children[0].borrow().children[0]);
                        let expression_1: Rc<RefCell<AST>> =
                            Rc::clone(&self.children[0].borrow().children[1]);

                        let powered_expression_0: AST = AST {
                            value: Element::Exp,
                            children: vec![expression_0, get_ptr(AST::from_number(exp.clone()))],
                        };

                        let powered_expression_1: AST = AST {
                            value: Element::Exp,
                            children: vec![expression_1, get_ptr(AST::from_number(exp))],
                        };

                        self.value = base_opertation;
                        self.children =
                            vec![get_ptr(powered_expression_0), get_ptr(powered_expression_1)];
                        break 'exp self;
                    }
                }

                // 19)     (a^b)^c = a^(b*c)

                if self.children[0].borrow().value == Element::Exp {
                    // we have the requiered form

                    let exp_1: Rc<RefCell<AST>> = Rc::clone(&self.children[1]);
                    let exp_2: Rc<RefCell<AST>> = Rc::clone(&self.children[0].borrow().children[1]);

                    let final_exponent: AST = AST {
                        value: Element::Mult,
                        children: vec![exp_1, exp_2],
                    };

                    let base: Rc<RefCell<AST>> = Rc::clone(&self.children[0].borrow().children[0]);

                    // self.value = Element::Exp;
                    // ^unnecessary, already done

                    self.children = vec![base, get_ptr(final_exponent)];

                    break 'exp self;
                }

                self
            }
            Element::Neg => 'neg: {
                // 14)  -(-x) = x
                if self.children[0].borrow().value == Element::Neg {
                    break 'neg self.children[0].borrow().children[0].borrow().deep_copy();
                }

                let maybe_const: Element = self.children[0].borrow().value.clone();
                if let Element::Number(constant) = maybe_const {
                    let new_constant: Number = match constant {
                        Number::Real(r) => Number::Real(-r),
                        Number::Rational(n, d) => Number::Rational(-n, d),
                    };

                    self.value = Element::Number(new_constant);
                    self.children.clear();
                }

                self
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
                let simplified: Result<AST, String> = child.borrow().deep_copy().simplify_step();
                break 'clos simplified;

                match simplified {
                    Ok(ast) => ast.join_terms(),
                    Err(e) => Err(e),
                }
            })
            .collect();

        ret.children = updated?.into_iter().map(|x| get_ptr(x)).collect();

        if PRINT_DGB_STATEMENTS {
            println!("Out [{:.4}]: {:?}\n", call_id, ret.to_string());
        }

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
    pub fn join_terms(&self) -> Result<Self, String> {
        /*
        Plan: if multiple elements are multiplied/added they can be re-arranged.
        For addition, we need to find the elements in common in the form:
         (a * f) or f = (1 * f); where f is some function.

         Then we can join then. For multiplication, we need to find:

         (f^a) or f = (f^1) or 1/(f^a) = (f^-a) or 1/f = (f^-1)

            FIRST JOIN NUMBERS (easy join)

         */

        fn matches_sum_form(input: Ref<'_, AST>, other: Ref<'_, AST>) -> Option<Number> {
            /*
            local function. Checks is the given AST follows the form of a sum or multiple of it.

            input: f
            other: f || f * a

            if None, no correlation. If some(k), k is the multiplier of that therm

            */

            if input.eq(&other) {
                return Some(Number::Rational(1, 1));
            }

            if other.value != Element::Mult {
                return None;
            }

            if let Element::Number(n) = &other.children[0].borrow().value {
                if other.children[1].borrow().eq(&input) {
                    return Some(n.clone());
                }
            }

            if let Element::Number(n) = &other.children[1].borrow().value {
                if other.children[0].borrow().eq(&input) {
                    return Some(n.clone());
                }
            }

            None
        }

        fn matches_mult_form(input: Ref<'_, AST>, other: Ref<'_, AST>) -> Option<Number> {
            /*
            local function. Checks is the given AST follows the form of a multiple or power of it.

            input: f
            other: f || f ^ a || 1/f || 1/f^a

            if None, no correlation. If some(k), k is the multiplier of that therm
            */

            // case other = f
            if input.eq(&other) {
                return Some(Number::Rational(1, 1));
            }

            match &other.value {
                Element::Exp => {
                    //case other = f^a      for some a: Number
                    // if power is number and base is same as input
                    if let Element::Number(n) = &other.children[0].borrow().value {
                        if other.children[1].borrow().eq(&input) {
                            return Some(n.clone());
                        }
                    }
                }
                Element::Div => 'div: {
                    if let Element::Number(n) = &other.children[0].borrow().value {
                        let is_numerator_one: bool = match n {
                            Number::Real(r) => *r == 1.0,
                            Number::Rational(n, d) => *n == 1 && *d == 1,
                        };

                        if is_numerator_one {
                            break 'div;
                        }

                        // case other = 1/f
                        if other.children[1].borrow().eq(&input) {
                            return Some(Number::Rational(-1, 1));
                        }

                        // Case other = 1/f^a
                        if let Element::Exp = other.children[1].borrow().value {
                            // if denominator is exponential
                            if let Element::Number(n) =
                                &other.children[1].borrow().children[1].borrow().value
                            {
                                // and power is a number
                                if other.children[1].borrow().children[0].borrow().eq(&input) {
                                    // and base is f
                                    let mut exponent: Number = n.clone();
                                    exponent = exponent * Number::Rational(-1, 1);
                                    return Some(exponent);
                                }
                            }
                        }
                    }
                }
                _ => return None,
            }

            None
        }

        let (operation, operation_joiner, compare): (
            Element,
            Element,
            fn(Ref<'_, AST>, Ref<'_, AST>) -> Option<Number>,
        ) = match self.value {
            Element::Add => (Element::Add, Element::Mult, matches_sum_form),
            Element::Mult => (Element::Mult, Element::Exp, matches_mult_form),
            _ => return Ok(self.deep_copy()),
        };
        // operation is the operation we are working with, operation_joiner is the operator that
        // allows us to join multiple of the same elements into one.

        // childs vec will contain the childs of self with the same element as self
        let mut childs: Vec<Rc<RefCell<AST>>> = Vec::new();
        {
            // 1st iteration manual because type system
            let mut stack: Vec<Rc<RefCell<AST>>> = vec![];
            //already know self.value == operation
            self.children
                .iter()
                .for_each(|ch| stack.push(Rc::clone(ch)));

            //let mut stack: Vec<Rc<RefCell<AST>>> = vec![Rc::new(RefCell::new(self))];
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
            assert!(stack.len() == 0);
        }

        // groups will contain each of the AST children and the ammount of times it has been seen.
        let mut groups: Vec<Rc<RefCell<AST>>> = Vec::new();
        while let Some(ast) = childs.pop() {
            let mut counter: Number = Number::Rational(1, 1); // 1 is ast (current one)

            loop {
                // find if there exist other ast with same structure
                let mut index_and_value_opt: Option<(usize, Number)> = None;
                for (other_idx, other) in childs.iter().enumerate() {
                    if let Some(magnitude) = compare(ast.borrow(), other.borrow()) {
                        index_and_value_opt = Some((other_idx, magnitude));
                    }
                }

                let index_and_value: (usize, Number) = match index_and_value_opt {
                    Some(v) => v,
                    None => break,
                };

                counter = counter + index_and_value.1;
                childs.swap_remove(index_and_value.0);
            }

            if counter == Number::Rational(1, 1) {
                groups.push(ast);
            } else {
                // create an AST that joins the elements accordingly
                groups.push(Rc::new(RefCell::new(AST {
                    value: operation_joiner.clone(),
                    children: vec![ast, Rc::new(RefCell::new(AST::from_number(counter)))],
                })));
            }
        }

        // Now we need to join all the elements into a the AST structure

        let mut base_layer: Vec<Rc<RefCell<AST>>> = groups;
        let mut upper_layer: Vec<Rc<RefCell<AST>>> =
            Vec::with_capacity((base_layer.len() >> 1) + 1);
        let mut missing: Option<Rc<RefCell<AST>>> = None;
        // ^ missing is needed if the number if elements in base_layer is not even.
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
    #[allow(dead_code)]
    fn neg_to_sub(&mut self) {
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

    /// If the [AST] has the form `-1*f`, where f is any expression,
    /// the multiplication by -1 will be substituted to a negation.
    ///
    /// If the [AST] does not follow the `-1*f`, nothing will be done.
    /// Returns a bool indication if a change was performed.
    fn mult_neg1_to_neg(&mut self) -> bool {
        if self.value != Element::Mult {
            return false;
        }

        let first_child_value: Element = self.children[0].borrow().value.clone();
        if let Element::Number(num) = first_child_value {
            if num.in_tolerance_range(&Number::Real(-1.0), 0.00001) {
                let other: Rc<RefCell<AST>> = Rc::clone(&self.children[1]);
                self.value = Element::Neg;
                self.children = vec![other];
                return true;
            }
        }

        let second_child_value: Element = self.children[1].borrow().value.clone();
        if let Element::Number(num) = second_child_value {
            if num.in_tolerance_range(&Number::Real(-1.0), 0.00001) {
                let other: Rc<RefCell<AST>> = Rc::clone(&self.children[0]);
                self.value = Element::Neg;
                self.children = vec![other];
                return true;
            }
        }

        return false;
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

    /// Returns a human-readable stringified version of the [AST].
    pub fn to_string(&self) -> String {
        return match &self.value {
            Element::Derive => format!("der({})", self.children[0].borrow().to_string()),
            Element::Function(identifier) => {
                format!(
                    "{}({})",
                    identifier.to_string(),
                    self.children[0].borrow().to_string()
                )
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
                let child_left: Ref<'_, AST> = self.children[0].borrow();
                let left_side: String = match child_left.value.clone() {
                    Element::Add => format!("({})", child_left.to_string()),
                    Element::Sub => format!("({})", child_left.to_string()),
                    Element::Number(number) => number.as_str(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => child_left.to_string(), //der, fn, mult, div, exp, fact, mod, neg
                };

                let child_right: Ref<'_, AST> = self.children[1].borrow();
                let right_side: String = match child_right.value.clone() {
                    Element::Add => format!("({})", child_right.to_string()),
                    Element::Sub => format!("({})", child_right.to_string()),
                    Element::Number(number) => number.as_str(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => child_right.to_string(), //der, fn, mult, div, exp, fact, mod, neg
                };

                format!("{}*{}", left_side, right_side)
            }
            Element::Div => {
                let child_left: Ref<'_, AST> = self.children[0].borrow();
                let numerator: String = match child_left.value.clone() {
                    Element::Add => format!("({})", child_left.to_string()),
                    Element::Sub => format!("({})", child_left.to_string()),
                    Element::Number(number) => number.as_str(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => child_left.to_string(), // der, fn, mult, div, exp, fact, mod, neg
                };

                let child_right: Ref<'_, AST> = self.children[1].borrow();
                let denominator: String = match child_right.value.clone() {
                    Element::Derive => child_right.to_string(),
                    Element::Function(_) => child_right.to_string(),
                    Element::Exp => child_right.to_string(),
                    Element::Fact => child_right.to_string(),
                    Element::Mod => child_right.to_string(),
                    Element::Number(number) => number.as_str(),
                    Element::Var => String::from("x"),
                    Element::Neg => child_right.to_string(),
                    Element::None => String::from("None"),
                    _ => format!("({})", child_right.to_string()), // +, -, *, /
                };

                format!("{}/{}", numerator, denominator)
            }
            Element::Exp => {
                let child_left: Ref<'_, AST> = self.children[0].borrow();
                let left_side: String = match child_left.value.clone() {
                    Element::Derive => child_left.to_string(),
                    Element::Function(_) => child_left.to_string(),
                    Element::Fact => child_left.to_string(),
                    Element::Number(number) => number.as_str(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => format!("({})", child_left.to_string()),
                };

                let child_right: Ref<'_, AST> = self.children[1].borrow();
                let right_side: String = match child_right.value.clone() {
                    Element::Derive => child_right.to_string(),
                    Element::Function(_) => child_right.to_string(),
                    Element::Exp => child_right.to_string(),
                    Element::Fact => child_right.to_string(),
                    Element::Number(number) => number.as_str(),
                    Element::Var => String::from("x"),
                    Element::Neg => child_right.to_string(),
                    Element::None => String::from("None"),
                    _ => format!("({})", child_right.to_string()),
                };

                format!("{}^{}", left_side, right_side)
            }
            Element::Fact => {
                let child: Ref<'_, AST> = self.children[0].borrow();
                let left_side: String = match child.value.clone() {
                    Element::Derive => child.to_string(),
                    Element::Function(ident) => {
                        format!("{}({})", ident.to_string(), child.to_string())
                    }
                    Element::Fact => child.to_string(),
                    Element::Number(number) => number.as_str(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => format!("({})", child.to_string()), // +, -, *, /, ^
                };

                format!("{}!", left_side)
            }
            Element::Mod => {
                let child_left: Ref<'_, AST> = self.children[0].borrow();
                let left_side: String = match child_left.value.clone() {
                    Element::Number(number) => number.as_str(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => child_left.to_string(),
                };

                let child_right: Ref<'_, AST> = self.children[1].borrow();
                let right_side: String = match child_right.value.clone() {
                    Element::Number(number) => number.as_str(),
                    Element::Var => String::from("x"),
                    Element::None => String::from("None"),
                    _ => child_right.to_string(),
                };

                format!("{}%{}", left_side, right_side)
            }
            Element::Number(number) => number.as_str(),
            Element::Var => String::from("x"),
            Element::Neg => {
                let child_left: Ref<'_, AST> = self.children[0].borrow();
                let left_side: String = match child_left.value.clone() {
                    Element::Number(number) => number.as_str(),
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

    /// Inserts a derive block and moves everything else
    ///
    /// (including self to the child of the new [Element::Derive]).
    /// Just created for convenience.
    ///
    /// Parent of self -> Self
    /// Parent of self -> Derive -> Self
    pub fn insert_derive(mut self) -> Self {
        let sub_tree: AST = AST {
            value: self.value.clone(),
            children: self.children,
        };

        self.value = Element::Derive;
        self.children = vec![Rc::new(RefCell::new(sub_tree))];

        self
    }

    /// Inserts a derive block and moves everything else
    ///
    /// (including self to the child of the new [Element::Derive]).
    /// Just created for convenience. Same as [AST::insert_derive]
    /// but with diferent signature.
    ///
    /// Parent of self -> Self
    /// Parent of self -> Derive -> Self
    fn insert_derive_new(&self) -> Self {
        let mut input: AST = self.deep_copy();

        let sub_tree: AST = AST {
            value: input.value,
            children: input.children,
        };

        input.value = Element::Derive;
        input.children = vec![Rc::new(RefCell::new(sub_tree))];

        input
    }

    /// Inserts a derive block and moves everything else
    ///
    /// (including self to the child of the new [Element::Derive]).
    /// Just created for convenience. Same as [AST::insert_derive]
    /// but with diferent signature.
    ///
    /// Parent of self -> Self
    /// Parent of self -> Derive -> Self
    #[allow(dead_code)]
    fn insert_derive_ref(&mut self) {
        let childs: vec::Drain<'_, Rc<RefCell<AST>>> = self.children.drain(0..);
        let elem: Element = self.value.clone();

        let sub_tree: AST = AST {
            value: elem,
            children: childs.collect(),
        };

        self.value = Element::Derive;
        self.children = vec![Rc::new(RefCell::new(sub_tree))];
    }

    /// Returns true if the [AST] contains any [Element::Derive].
    pub fn contains_derives(&self) -> bool {
        if self.value == Element::Derive {
            return true;
        }

        let mut stack: Vec<Rc<RefCell<AST>>> = Vec::new();
        self.children
            .iter()
            .for_each(|ch| stack.push(Rc::clone(ch)));

        while let Some(child) = stack.pop() {
            if child.borrow().value == Element::Derive {
                return true;
            }

            child
                .borrow()
                .children
                .iter()
                .for_each(|ch| stack.push(Rc::clone(ch)));
        }

        return false;
    }

    /// Derives the contents of the given [AST].
    ///
    /// Calling this function on a [AST] means that it's parent has
    /// a value of the variant [Element::Derive].
    pub fn derive(&self) -> Result<Self, String> {
        // Derivative rules: https://en.wikipedia.org/wiki/Differentiation_rules

        if let Element::Derive = self.value {
            //(f')' = f''

            /*
            //derive it's children recursively.
            let derivated_child: AST = self.children[0].borrow().derive()?;

            // derive again for the parent of the given &self.
            return derivated_child.derive();
            */
            return Ok(self.insert_derive_new());
        }

        let ret: AST = match self.value {
            Element::Derive => {
                panic!("Impossible case. ");
                // todo!("No support for 2nd derivatives right now. To be implemented. ")
            }
            //Element::Derive => self.children[0].borrow().derive(),
            Element::Function(_) => {
                //todo!("Use derivative rule for each function. ")}
                Functions::func_derive(self)?
            }
            Element::Add => AST {
                value: Element::Add,
                children: vec![
                    Rc::new(RefCell::new(self.children[0].borrow().insert_derive_new())),
                    Rc::new(RefCell::new(self.children[1].borrow().insert_derive_new())),
                ],
            },
            Element::Sub => AST {
                value: Element::Sub,
                children: vec![
                    Rc::new(RefCell::new(self.children[0].borrow().insert_derive_new())),
                    Rc::new(RefCell::new(self.children[1].borrow().insert_derive_new())),
                ],
            },
            Element::Mult => {
                // (f*g)' = f'*g + g'*f
                // assume only 2 multiplied elements, otherwise invalid AST

                // f'
                let der_0: AST = self.children[0].borrow().insert_derive_new();
                // g'
                let der_1: AST = self.children[1].borrow().insert_derive_new();

                // f'*g
                let prod_0: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(der_0)),
                        Rc::new(RefCell::new(self.children[1].borrow().deep_copy())),
                    ],
                };

                // g'*f
                let prod_1: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(der_1)),
                        Rc::new(RefCell::new(self.children[0].borrow().deep_copy())),
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
                let der_0: AST = self.children[0].borrow().insert_derive_new();

                // g'
                let der_1: AST = self.children[1].borrow().insert_derive_new();

                // f'*g
                let prod_0: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(der_0)),
                        Rc::new(RefCell::new(self.children[1].borrow().deep_copy())),
                    ],
                };

                // g'*f
                let prod_1: AST = AST {
                    value: Element::Mult,
                    children: vec![
                        Rc::new(RefCell::new(der_1)),
                        Rc::new(RefCell::new(self.children[0].borrow().deep_copy())),
                    ],
                };

                // f'*g - g'*f
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

                let base_contains_var: bool = self.children[0].borrow().contains_variable();
                let exponent_contains_var: bool = self.children[1].borrow().contains_variable();
                match (base_contains_var, exponent_contains_var) {
                    (true, true) => {
                        // f^g => f^g * (f' * g/f + g' * ln(f))
                        // oh, boy...

                        // f'
                        let der_0: AST = self.children[0].borrow().insert_derive_new();
                        // g'
                        let der_1: AST = self.children[1].borrow().insert_derive_new();

                        // ln(f)
                        let ln_f: AST = AST {
                            value: Element::Function(FnIdentifier::Ln),
                            children: vec![Rc::new(RefCell::new(
                                self.children[0].borrow().deep_copy(),
                            ))],
                        };

                        // g/f
                        let g_over_f: AST = AST {
                            value: Element::Div,
                            children: vec![
                                Rc::new(RefCell::new(self.children[1].borrow().deep_copy())),
                                Rc::new(RefCell::new(self.children[0].borrow().deep_copy())),
                            ],
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
                        let der: AST = self.children[0].borrow().insert_derive_new();

                        // a
                        let exp: Number = self.children[1].borrow_mut().evaluate(None)?;

                        // a-1
                        let exp_minus_1: Number = exp.clone() - Number::Rational(1, 1);

                        // f^(a-1)
                        let power: AST = AST {
                            value: Element::Exp,
                            children: vec![
                                Rc::new(RefCell::new(self.children[0].borrow().deep_copy())),
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
                        let der: AST = self.children[1].borrow().insert_derive_new();

                        let mut ln_a_numerical: Number =
                            self.children[0].borrow_mut().evaluate(None)?;
                        ln_a_numerical =
                            Functions::find_and_evaluate(FnIdentifier::Ln, ln_a_numerical)?;

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
                                Rc::new(RefCell::new(self.deep_copy())),
                                Rc::new(RefCell::new(der_ln_a)),
                            ],
                        }
                    }
                    (false, false) => {
                        //just a constant. The derivative of a constant is 0.
                        AST_ZERO.clone()
                    }
                }
            }
            Element::Fact => {
                return Err(String::from(
                    "Derivative of the factorial function is not supported. ",
                ))
            }
            Element::Mod => {
                //just the identity

                /*
                   h(x) = f(x) mod g(x) = f(x) - floor(f(x)/g(x)) * g(x)
                               = f(x) - g(x) * floor(f(x)/g(x))

                   if g(x) is constant then g'(x) = 0 and h'(x) = f'(x).
                   Otherwise: (simplifiying derivative of floor(x) to 0)

                   h'(x) = f'(x) - (floor'(f(x)/g(x))*(f(x)/g(x))'*g(x) + g'(x) * floor(f(x)/g(x)))
                   h'(x) = f'(x) - 0*(f(x)/g(x))'*g(x) - g'(x) * floor(f(x)/g(x))
                   h'(x) = f'(x) - g'(x) * floor(f(x)/g(x))

                */

                #[warn(UncompleteCode)]
                self.clone()
            }
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
                let der: AST = self.children[0].borrow().insert_derive_new();

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

    /// Expants the derivated subtrees to it's corresponding derivated representation.
    ///
    /// If there are no derives it just returns a deep clone of self and the bool flag = false.
    ///
    /// If `one_step = false`, then this will be executed until no more derives
    /// are left in the tree. It also implies that the bool flag = false
    ///
    /// If successfull, the flag determines if there is any other [Element::Derive]
    /// left in the tree.
    ///
    /// If verbose = true, will print the stringified [AST] after each round of derives.
    /// Will not print the [AST] before deriving it.
    pub fn execute_derives(&self, one_step: bool, verbose: bool) -> Result<(Self, bool), String> {
        /* If there are multiple derives, we will only execute the ones
        that do not contain any other derive. (`der(der(x^2) + der(4x))` => `der(2*x + 4)`).

        If `one_step` = false. We will repeat this process until no more derives are found.

        We are interested in the nodes of the [AST] that do not contain any other Element::Derive
        among it's descendants.

        */

        let root: Rc<RefCell<AST>> = Rc::new(RefCell::new(self.deep_copy()));

        let derives_left: bool = loop {
            let mut pending: Vec<Rc<RefCell<AST>>> = AST::get_derive_descendants(Rc::clone(&root));
            // elements that do not have any other derive in its descendants.
            let mut done: Vec<Rc<RefCell<AST>>> = Vec::new();
            let mut abandoned_derives: bool = false;

            while let Some(node) = pending.pop() {
                //let mut derive_descendants: Vec<Rc<RefCell<AST>>> = AST::get_derive_descendants(Rc::clone(&node));
                let mut derive_descendants: Vec<Rc<RefCell<AST>>> = node
                    .borrow()
                    .children
                    .iter()
                    .map(|children| AST::get_derive_descendants(Rc::clone(&children)))
                    .reduce(|mut a, mut b| {
                        a.append(&mut b);
                        a
                    })
                    .unwrap_or(Vec::new());

                if derive_descendants.is_empty() {
                    done.push(node);
                } else {
                    pending.append(&mut derive_descendants);
                    abandoned_derives = true;
                }
            }

            let mut added_derives: bool = false;

            for node in done {
                // get the child and derive it.
                let derivative: AST = match node.borrow().children.get(0) {
                    Some(v) => v.borrow().derive()?,
                    None => return Err(String::from("Expected child after Element::Derive. \n")),
                };

                //set node = to the corresponding derivative.
                node.borrow_mut().children.clear();
                node.borrow_mut().children = derivative.children;
                node.borrow_mut().value = derivative.value;

                added_derives = added_derives || node.borrow().contains_derives();
            }

            if verbose {
                //printing time!
                println!("\n{}\n", root.borrow().to_string());
            }

            // if we just asked for 1 step or we don't have any derives left, exit
            if one_step || !(abandoned_derives || added_derives) {
                break abandoned_derives || added_derives;
            }
        };

        let ret: AST = Rc::try_unwrap(root)
            .expect("Failed to unwrap Rc pointer in execute_derives. ")
            .into_inner();

        Ok((ret, derives_left))
    }

    /// reutrns a vec of the descendants (ir input) that are [Element::Derive]
    fn get_derive_descendants(input: Rc<RefCell<Self>>) -> Vec<Rc<RefCell<Self>>> {
        if input.borrow().value == Element::Derive {
            return vec![input];
        }

        input
            .borrow()
            .children
            .iter()
            .map(|child| AST::get_derive_descendants(Rc::clone(child)))
            .reduce(|mut a, mut b| {
                a.append(&mut b);
                a
            })
            .unwrap_or(Vec::new())
    }

    pub fn full_derive(
        &self,
        simplify_final_expression: bool,
        print_procedure: bool,
    ) -> Result<Self, String> {
        // Add a derive and expand

        let ast: AST = self.deep_copy().insert_derive();

        if print_procedure {
            println!("With derivative: {}\n", self.to_string());
        }

        let (mut derivated, flag) = ast.execute_derives(false, print_procedure)?;
        assert!(flag == false); // No more missing derives since one_step = false (all steps done)
        if print_procedure {
            println!("Completely derives: {}\n", self.to_string());
        }

        if simplify_final_expression {
            derivated = derivated
                .partial_evaluation()?
                .simplify_expression()?
                .partial_evaluation()?;
        }

        Ok(derivated)
    }
}

impl Evaluable for AST {
    /// Evaluates the [AST] recursively.
    fn evaluate(&self, var_value: Option<Number>) -> Result<Number, String> {
        match &self.value {
            Element::Derive => {
                return Err(String::from(
                    "Cannot evaluate derivative. Derive first and then evaluate. ",
                ))
            }
            Element::Function(name) => {
                return crate::functions::Functions::find_and_evaluate(
                    name.clone(),
                    (*self.children[0].borrow_mut()).evaluate(var_value)?,
                );
            }
            Element::Add => {
                let mut acc: Number = Number::Rational(0, 1);

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
                            Number::Rational(n, d) => {
                                let a: Number = match Number::new_rational(-n, d) {
                                    Ok(v) => v,
                                    Err(_) => {
                                        return Err(String::from(
                                            "Division by 0 is not possible. \n",
                                        ))
                                    }
                                };
                                Ok(a)
                            }
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

                let mut acc: Number = Number::Rational(1, 1);

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
                    FnIdentifier::Inv,
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
                            return Ok(Number::Rational(1, 1));
                        }
                        let mut acc: i64 = 1;
                        for i in 1..=num {
                            acc = acc * i;
                        }

                        Ok(Number::Rational(acc, 1))
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

                    return Ok(Number::Rational(x_int % y_int, 1));
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
    pub fn new_rational(num: i64, den: u64) -> Result<Self, ()> {
        if den == 0 {
            // return Err(format!("Division by 0 is not possible. \n"));
            return Err(());
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

        // we know its a Natural number so it's ok to cast. Done to work with similar types
        let mut n: u64 = x as u64;
        while (n & (0b11 as u64)) == 0 {
            //ends with 00
            n = n >> 2;
            // loop must terminate because input contains at least 1 bit set to 1
        }

        if (n & (0b111 as u64)) != 1 {
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

            match square.cmp(&n) {
                std::cmp::Ordering::Equal => return true,
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid - 1,
            }
        }

        return false; // No sqrt root found
    }

    /// Determinates if the given integer is a perfect squer or not.
    ///
    /// If it is, returns Some() with the square root as an integer.
    /// Otherwise returns None. See the implementation of
    /// [Number::scan_perfect_square] for the details on how it works.
    /// This is a readapted version of that code.
    pub fn is_perfect_square(x: i64) -> Option<i64> {
        //Perfec number info: https://en.wikipedia.org/wiki/Square_number

        match x.cmp(&0) {
            std::cmp::Ordering::Less => return None,
            std::cmp::Ordering::Equal => return Some(0),
            std::cmp::Ordering::Greater => {}
        }

        let mut n: u64 = x as u64;
        let mut reduction_steps: u8 = 0;
        while (n & (0b11 as u64)) == 0 {
            n = n >> 2;
            reduction_steps += 1;
        }

        if (n & (0b111 as u64)) != 1 {
            return None;
        }

        let log: u32 = n.ilog2();
        let aprox: u64 = 1u64 << (log >> 1);
        let mut left: u64 = aprox - 1;
        let mut right: u64 = aprox * 2 + 1;

        while left <= right {
            let mid: u64 = left + (right - left) / 2;
            let square: u64 = mid * mid;

            match square.cmp(&n) {
                std::cmp::Ordering::Equal => return Some((mid as i64) << reduction_steps),
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid - 1,
            }
        }

        return None;
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
            //exact comparasion
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

        /*
        Idea:
        given 0 < tolerance:
        let tolerande = e

        (self - other).abs() < e

        Implementation will depend on the variants of self and other.

        */

        return match (&self, &other) {
            (Number::Real(x), Number::Real(y)) => {
                let ret: bool = (x - y).abs() < tolerance;
                ret
            }
            (Number::Real(r), Number::Rational(num, den)) => {
                //let ret: bool = (*r - (*num as f64 / *den as f64)).abs() < tolerance;

                /*
                abs(a/b - r) < e
                abs(a/b - r*b/b) < e
                abs((a - r*b)/b) < e
                abs(a - r*b)/abs(b) < e
                abs(a - r*b) < e * abs(b)
                */

                let f_den: f64 = *den as f64; // float denominator
                let lhs: f64 = (*num as f64) - (*r) * f_den;
                let rhs: f64 = tolerance * f_den;

                let ret: bool = lhs.abs() < rhs;

                ret
            }
            (Number::Rational(num, den), Number::Real(r)) => {
                //let ret: bool = (*r - (*num as f64 / *den as f64)).abs() < tolerance;

                /*
                abs(a/b - r) < e
                abs(a/b - r*b/b) < e
                abs((a - r*b)/b) < e
                abs(a - r*b)/abs(b) < e
                abs(a - r*b) < e * abs(b)
                */

                let f_den: f64 = *den as f64; // float denominator
                let lhs: f64 = (*num as f64) - (*r) * f_den;
                let rhs: f64 = tolerance * f_den;

                let ret: bool = lhs.abs() < rhs;

                ret
            }
            (Number::Rational(self_num, self_den), Number::Rational(oth_num, oth_den)) => {
                if self_num == oth_num && self_den == oth_den {
                    return true;
                }

                /*
                let ret: bool = ((*self_num as f64 / *self_den as f64)
                    - (*oth_num as f64 / *oth_den as f64))
                    .abs()
                    < tolerance;
                */

                /*
                abs(a/b - c/d) < e
                diff = a/b - c/d

                = a*d/b*d - c*b/d*b
                = (a*d-c*b)/db

                > abs(x/y) = abs(x) * abs(1/y)
                > abs(x/y) = abs(x) * 1/abs(y)
                > abs(x/y) = abs(x)/abs(y)

                abs((a*d-c*b)/db) < e
                abs(a*d-c*b)/abs(db) < e
                abs(a*d-c*b) < e * abs(db)

                */

                let join_num: f64 =
                    (*self_num as f64) * (*oth_den as f64) - (*oth_num as f64) * (*self_den as f64);
                let join_den: f64 = (*self_den as f64) * (*oth_den as f64);

                let ret: bool = join_num.abs() < tolerance * join_den;

                ret
            }
        };
    }

    /// Returns the number as a string.
    ///
    /// If the number happens to be close enough to a constant, returns the
    /// constant name.
    ///
    /// If the number is [Number::Rational] and the numerator and denominator are not
    /// too big (less than [PRINT_FRACTION_PRECISION_THRESHOLD]), they will be
    /// returned in the form "a/b". If they are integers they will only be returned
    /// the integer part normally ("a").
    ///
    /// Otherwise, (if the number is [Number::Real] or [Number::Rational] but too big),
    /// it will return the stringified numerical representation "a.b".
    /// Only [PRINT_NUMBER_DIGITS] will be returned. Note that the number is truncated,
    /// not aproximated.
    ///
    /// If instead you want the **exact awnser** use [Number::as_numerical_str] instead.
    /// This is designed to be a simple human-readable stringification.
    pub fn as_str(&self) -> String {
        /*
        Idea: If number is a constant, print the related literal.

        If it's a rational, print as a/b if a and b are not too large.

        Otherwise, it's numerical representation will be used and only
        PRINT_NUMBER_DIGITS decimal places will be displayed.
        */

        unsafe {
            // This is safe because [NUMERICAL_OUTPUTS] may only be changed
            // right at the start of the program (before this method is called)
            // once depending on the flags.
            if NUMERICAL_OUTPUTS {
                return self.as_numerical_str();
            }
        }

        if let Some(const_str) = functions::Constants::is_constant(self) {
            return const_str.to_string();
        }

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
            if PRINT_NUMBER_DIGITS <= counter {
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

    /// Returns the number as a string, but with maximum precision
    /// and the result is guaranteed to be a valid number
    pub fn as_numerical_str(&self) -> String {
        match self {
            Number::Real(r) => format!("{}", r),
            Number::Rational(n, d) => {
                if d == &1 {
                    // is integer
                    format!("{}", n)
                } else if d == &0 {
                    panic!("Attempting to print a invalid rational. (denominator = 0)")
                } else {
                    format!("{}/{}", n, d)
                }
            }
        }
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Number) -> bool {
        match (self, other) {
            (Number::Real(r1), Number::Real(r2)) => r1 == r2,
            (Number::Real(r), Number::Rational(n, d)) => r * *d as f64 == *n as f64,
            (Number::Rational(n, d), Number::Real(r)) => r * *d as f64 == *n as f64,
            (Number::Rational(n1, d1), Number::Rational(n2, d2)) => {

                // n1 * *d2 as i64 == n2 * *d1 as i64
                

                let mult_1: Option<i64> = n1.checked_mul(*d2 as i64); 
                let mult_2: Option<i64> = n2.checked_mul(*d1 as i64); 

                match (mult_1, mult_2) {
                    (Some(a), Some(b)) => {
                        a == b
                    }
                    _ => {
                        let a: u128 = *n1 as u128; 
                        let b: u128 = *d2 as u128; 
                        let c: u128 = *n2 as u128; 
                        let d: u128 = *d1 as u128; 

                        a*b == c*d
                    }
                }

            }
        }
    }
}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Number) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Number::Real(r1), Number::Real(r2)) => r1.partial_cmp(r2),
            (Number::Real(r), Number::Rational(n, d)) => (r * *d as f64).partial_cmp(&(*n as f64)),
            (Number::Rational(n, d), Number::Real(r)) => (r * *d as f64).partial_cmp(&(*n as f64)),
            (Number::Rational(n1, d1), Number::Rational(n2, d2)) => {
                //    a/b = c/d     => a * d == c * d

                //Some((n1 * *d2 as i64).cmp(&(n2 * *d1 as i64)))

                let mult_1: Option<i64> = n1.checked_mul(*d2 as i64); 
                let mult_2: Option<i64> = n2.checked_mul(*d1 as i64); 
                match (mult_1, mult_2) {
                    (Some(a), Some(b)) => Some(a.cmp(&b)), 
                    _ => {
                        let a: i128 = *n1 as i128; 
                        let b: i128 = *d2 as i128; 
                        let c: i128 = *n2 as i128; 
                        let d: i128 = *d1 as i128; 

                        Some((a * b).cmp(&(c * d)))

                    }
                }

            }
        }
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

                    // Least Common Multiple
                    let lcm: u64 = lcm_res.unwrap(); 

                    let mult_factor_left: i64 = lcm as i64 / left_den as i64;
                    let mult_factor_right: i64 = lcm as i64 / right_den as i64;

                    let num: i64 = right_num * mult_factor_right + left_num * mult_factor_left;
                    let den: u64 = lcm;

                    let mut new_rational: Number =
                        Number::new_rational(num, den).expect("Attempting to add 2 Rationals and 1 of them has 0 as divisor. ");
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
                        Number::new_rational(num, den).expect("Attempting to substract 2 Rationals and 1 of them has 0 as divisor. ");
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
                            Number::new_rational(num, den).expect("Attempting to multiply 2 Rationals and 1 of them has 0 as divisor. ")
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
        //f.debug_struct("Number").field("value", &self.value).finish()

        // Just reuse [Number::as_str]
        write!(f, "{}", self.as_str())
    }
}

impl Evaluable for Number {
    /// Returns the [Number] itself.
    fn evaluate(&self, _var_value: Option<Number>) -> Result<Number, String> {
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
