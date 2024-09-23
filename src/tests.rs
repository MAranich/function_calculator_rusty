use integer_sqrt::IntegerSquareRoot;
use rand::prelude::*;
use rayon::prelude::*;

use crate::*;
use core::panic;
use std::{cell::RefCell, f64::consts::PI, rc::Rc, time::Instant};

#[test]
fn stringify_ast_test() {
    let tests: Vec<(String, String)> = vec![(String::from("2 +  3"), String::from("2+3"))];

    let mut calc: Calculator = Calculator::new(setup::setup_dfas(), SRA::new(setup::get_rules()));

    for (input, expected_output) in tests {
        println!("\nTesting input: {:#?}\n", input);

        match generate_ast(input, &mut calc, true) {
            Ok(ret) => {
                if ret.to_string() != expected_output {
                    panic!(
                        "\nExpression evaluated to: \n\t{}\nInstead of teh expected: \n\t{}",
                        ret.to_string(),
                        expected_output
                    );
                }
            }
            Err(msg) => panic!("\n{}", msg),
        };
    }
}

#[test]
fn global_derive_test() {
    let tests: Vec<(String, String)> = vec![
        (String::from("2 +  3"), String::from("0")),
        (String::from("1+x+x^2+x^3"), String::from("2x+3x^2")),
        (String::from("ln(x^2)"), String::from("2*x/ln(x^2)")),
    ];

    let mut calc: Calculator = Calculator::new(setup::setup_dfas(), SRA::new(setup::get_rules()));

    for (input, _expected_output) in tests {
        println!("\n\nTesting input: {:#?}\n", input);

        match generate_ast(input, &mut calc, true) {
            Ok(ast) => match ast.derive() {
                Ok(_der) => {}
                Err(msg) => panic!("Derivation failed with the error:\n{}", msg),
            },

            Err(msg) => panic!("AST generation failed with the error:\n{}", msg),
        };
    }
}

#[test]
fn printing_numbers() {
    let b: Number = Number::Real(PI*1.333);
    assert_eq!(b.as_str(), "4.18774");

    let a: Number = Number::Rational(7, 3);
    assert_eq!(a.as_str(), "7/3");
}

/*

## Perfect squares:

Binary, (except 0), they have the form XXX001(00)^n
Wich has the mathematical form:
    (8 * m + 1) * 2^2*n

### Proof:

Let Q be a square number and x be it's square root. Both Q and x are positive
integers.

Q   = x * x

In binary, every number can be divided into an even and odd part. (The trailing
zeroes and the rest)

x   = o * e , for all x
where e = 2^n, assuming maximal n and integer o and e.

Q   = (o * e) * (o * e)
    = o * e * o * e
    = o^2 * e^2
    = o^2 * (2^n)^2
    = o^2 * 2^(2*n)

Since o is odd, it has the form o = 2 * m + 1

o^2 = (2 * m + 1)^2 = 4*m^2 + 4 * m + 1 = 4*(m^2 + m) + 1

Q   = o^2 * 2^(2*n)
    = (4*(m^2 + m) + 1) * 2^(2*n)
    = (4*m*(m + 1) + 1) * 2^(2*n)

Note how m*(m+1) is even regardless of the parity of m. So let
2 * k = m * (m + 1)

Q   = (4*m*(m + 1) + 1) * 2^(2*n)
    = (4*2*k + 1) * 2^(2*n)
    = (8*k + 1) * 2^(2*n)

And with this we conclude the proof that any square number (except 0)
can be expressed in the form:

Q   = x * x
    = (8*k + 1) * 2^(2*n)

QED

Note how there are non-square numbers that also take this form such as
272 = 17*16 = 0b100010000 = (8*(2) + 1) * 2^(2*(2))
And there are infinitly many more. Because once we find a number
that ends in "001" we can always add more zeroes to make more.

This can be uscefull as a quick prior test to check if a number
is a perfect square. Although it will not determine it, it can be
used to discard many cases at a very inexpensive computational cost.

### Extra:

Recall in the proof:
2 * k = m * (m + 1)

k   = m * (m + 1)/2

So, if we can prove that k follows this form for some integer m, we will have
proved that Q (our original number) is a perfect square.

2*k = m * (m+1) = m^2 + m

0   = m^2 + m - 2*k

m   = -b/2 [+-] sqrt(b^2/4 - c)
m   = -1/2 [+-] sqrt(1/4 + 2 * k)
m   = -1/2 [+-] sqrt(1/4 + 2 * k)
m   = -1/2 [+-] sqrt(1/4 + 8 * k/4)
m   = -1/2 [+-] sqrt((1 + 8 * k)/4)
m   = -1/2 [+-] sqrt(1 + 8 * k)/2
m   = 1/2 * (-1 [+-] sqrt(1 + 8 * k))
m   = 1/2 * (-1 [+-] sqrt(8 * k + 1))

We can discard the negative solution for m since we know
it's a positive integer:

m   = 1/2 * (-1 + sqrt(8 * k + 1))

Note how sqrt(8 * k' + 1) is only integer iff (assuming our claim)
8 * k' + 1 is already a perfect square, wich is what we are trying to
determine in the first place. Also, we consider the sqrt() operation
expensive and we want to aviod it.





We are stuck here.



    ChatGPT:

I am working on a programming problemand I may need a bit of help. I am trying to determine if a given number (an i64) is a perfect square or not. I know that I can cast the number to a float and compute the square toot to figure it out, but I don't want to do that because sqrt() is slow and the method can fail for numbers greater to 2^52. This is what I came up so far:

If we represent perfect squares in binary, in their representation they end with the numbers "001" followed by an even number of 0 (can be none). Put in other words all perfect squares have the form `(8*k + 1) * 2^(2*n)` for some non-negative integers k and n. The problem with this method is that there are other numbers that follow this form but are not perfect squares, such as 17 (0b10001). A test just checking if the numbers follow this form only results in a ~83% accuracy overall.

Fortunately, after testing a bit, I found out something interesting. If we have a perfect square (with the form `(8*k + 1) * 2^(2*n)`) the k always has the form of the sum of the first m integers. So k = m*(m+1)/2. So if we can determine if a given number k can be expressed in the form m*(m+1)/2 for some non-negative integer.

To do this I tried to do the following:

2*k = m * (m+1) = m^2 + m

0   = m^2 + m - 2*k

And here we only have to solve this polynomial equation. After using the formula (and discarding the negative solution), we can simplify it to:

m   = 1/2 * (-1 + sqrt(8 * k + 1))

Wich is a problem because m will be an integer (a positive awnser) iff 8 * k + 1 is a perfect square, wich is the thing we were trying to do in the first place.

Currenly I am stuck here. Do you have any ideas on how to solve this in an efficient way?


    ChatGPT v2:



Proof square modulo 16:

/*
    Q   = (8*k + 1) * 2^(2*n)
    Q   = (8*k + 1) * 2^(2*n) mod 16

    Test for n = {0, 1, 2}

    Q   = (8*k + 1) * 2^(2*(0)) mod 16
    Q   = (8*k + 1) mod 16
    Q   = 8*k + 1 mod 16

    Q   = (8*k + 1) * 2^(2*(1)) mod 16
    Q   = (8*k + 1) * 4 mod 16
    Q   = 32*k + 4 mod 16
    Q   = 4 mod 16

    Q   = (8*k + 1) * 2^(2*(2)) mod 16
    Q   = (8*k + 1) * 2^4 mod 16
    Q   = (8*k + 1) * 16 mod 16
    Q   = 0 mod 16

    We can easly see that if 2 <= n then Q mod 16 = 0. For n = 0: test k = {0, 1, 2}

    Q   = 8*(0) + 1 mod 16
    Q   = 1 mod 16

    Q   = 8*(1) + 1 mod 16
    Q   = 9 mod 16

    Q   = 8*(2) + 1 mod 16
    Q   = 1 mod 16

    We can easly observe that the results in this case oscilate between 1 and 9 for
    an even or odd k, respectively.

*/

Proof square modulo 6:
/*
    Q   = (8*k + 1) * 2^(2*n)
    Q   = (8*k + 1) * 2^(2*n) mod 6

    Note: 6 = 2 * 3, wich are both primes, therefore the Euler's totient function
    for 6 is (2 - 1) * (3 - 1) = 1 * 2 = 2

    Q   = (8*k + 1) * 2^([2*n mod 2]) mod 6
    Q   = (8*k + 1) * 2^(0) mod 6
    Q   = 8*k + 1 mod 6

    In general, if k = 6*a + b, for some integer a and b in {0, 1, 2, 3, 4, 5}:

    Q   = 8*(6*a + b) + 1 mod 6
    Q   = 8*6*a + 8*b + 1 mod 6
    Q   = 8*b + 1 mod 6

    This reduces the number of cases to study to only 6. Try for k = {0, 1, 2, 3, 4, 5}:

    Q   = 8*(0) + 1 mod 6
    Q   = 1 mod 6

    Q   = 8*(1) + 1 mod 6
    Q   = 9 mod 6
    Q   = 3 mod 6

    Q   = 8*(2) + 1 mod 6
    Q   = 17 mod 6
    Q   = 5 mod 6

    Q   = 8*(3) + 1 mod 6
    Q   = 25 mod 6
    Q   = 5 mod 6

    Q   = 8*(4) + 1 mod 6
    Q   = 33 mod 6
    Q   = 1 mod 6

    Q   = 8*(5) + 1 mod 6
    Q   = 41 mod 6
    Q   = 3 mod 6

    Therefore Q = x * x mod 6 belongs to {1, 3, 5}. // TODO: fix this, 8^2 mod 6 = 4

*/

*/
#[test]
#[ignore = "Long"]
fn perf_square_speed() {
    let zero: Instant = Instant::now();
    let search: i32 = 32;
    (0..(1 << search)).into_iter().for_each(|i| {
        let _is_perf_sq: bool = Number::scan_perfect_square(i);
        //let sqrt: i64 = (i as f64).sqrt().floor() as i64;
        //let _is_perf_sq: bool = sqrt * sqrt == i;
    });

    let time_length: std::time::Duration = Instant::now() - zero;

    println!("Total time: {:.4}", time_length.as_secs_f32());

    assert!(time_length < std::time::Duration::from_secs(160));

    // This should not take longer than 160 s.

    /*
    Test: 2^32
        Basic   :  37.74 s
        New     :  66.27 s  (1.75x)
    Test: 2^38
        Basic   :  77.65 s
        New     : 133.80 s  (1.72x)
    V2: 2^32 (non-paralel)
        Basic       :  (91.01, 93.22, 93.24)    s (1.0x)
        New         :  (152.60, 152.97, 152.54, 150.30) s (1.73x)
        Un-opt      :  (565.04, 568.99, 561.52) s (6.14x)
        mod 16      :  (190.98, 195.16, 193.93) s (2,1x)
        No Bounds   :  (236.75, 238.49, 231.97) s (2.56x)
        Bin search  :  (1098.06, 1079.74, 1149.81)  s (12.06x)

     */
}

#[test]
#[ignore = "Long"]
fn is_perfect_square_test() {
    // numerically tested ut to 2^32 that if test == true => true_ground == true

    let search: i32 = 20;
    let mut fails: Vec<(i64, bool, bool)> = Vec::new();
    let mut counter: u64 = 0;
    //for i in 0..i64::MAX {
    for i in 0..(1 << search) {
        let decision: bool = Number::scan_perfect_square(i); // predicted is perf_sqrt

        let sqrt: i64 = (i as f64).sqrt().floor() as i64;
        //let sqrt: i64 = i.integer_sqrt();
        let true_ground: bool = sqrt * sqrt == i;

        if decision != true_ground {
            fails.push((i, decision, true_ground));
            counter += 1;
        }

        /*if !decision && true_ground {
            // decision => true_ground
            fails.push((i, decision, true_ground));
            counter += 1;
        }*/
    }
    /*
        assert!(
            decision == true_ground,
            "i: {}\t Predicted: {} \tGround: {}",
            i,
            decision,
            true_ground
        );
    */

    println!("(number, decision, ground truth)\n");
    let summary: Vec<(i64, bool, bool)> = fails
        .into_iter()
        .take(20)
        .collect::<Vec<(i64, bool, bool)>>();

    println!("Counter: {}", counter);
    assert!(counter == 0, "{}    {:?}", counter, summary);
    //assert!(fails.len() == 0, "{:?}", fails);

    /*
    base:
        speed:
    2^27 => 10 s
    2^32 => 5 min

    updated:


     */
}

#[test]
#[ignore = "Only used for researching. "]
fn aproximation_failing_point() {
    let b: i64 = 1 << 51;
    // 2^26 = 15.76 s
    // 2^28 = 1 min
    // 2^34 =~ 1h

    //  40.87s => 8.70s                 4,6977
    // 281.63s => 79.20s                3,555

    // ~1h = 3600 s => 1584.21s         ~2

    //for i in 0..i64::MAX {
    //(0..(1 << 30)).into_iter().for_each(|i|
    //(0..(1 << 38)).into_par_iter().for_each(|i|
    (b..(b + (1 << 30))).into_par_iter().for_each(|i| {
        let obtained_sqrt: i64 = (i as f64).sqrt().floor() as i64;
        let ground_sqrt: i64 = i.integer_sqrt();

        assert!(
            obtained_sqrt == ground_sqrt,
            "Iteration: {}\t {} and {}",
            i,
            obtained_sqrt,
            ground_sqrt
        );
    });

    /*
        ---- tests::aproximation_failing_point stdout ----
    thread '<unnamed>' panicked at src\tests.rs:132:9:
    Iteration: 4503600164241423      67108868 and 67108867
    thread '<unnamed>' panicked at src\tests.rs:132:9:
    Iteration: 4503599895805955      67108866 and 67108865
    thread '<unnamed>' panicked at src\tests.rs:132:9:
    Iteration: 4503599761588224      67108865 and 67108864
    thread '<unnamed>' panicked at src\tests.rs:132:9:
    Iteration: 4503600432676899      67108870 and 67108869
    thread '<unnamed>' panicked at src\tests.rs:132:9:
    Iteration: 4503600566894640      67108871 and 67108870
    thread '<unnamed>' panicked at src\tests.rs:132:9:
    Iteration: 4503600030023688      67108867 and 67108866
    thread '<unnamed>' panicked at src\tests.rs:132:9:
    Iteration: 4503600298459160      67108869 and 67108868

         */
}

#[test]
#[ignore = "Long"]
fn confusion_matrix_pref_sq() {
    let search: u32 = 35;
    #[derive(Clone, Debug, Default)]
    struct ConfusionMatrix {
        // is true, said true
        true_positives: u64,
        // is false, said false
        true_negatives: u64,
        // is false, said true
        false_positives: u64,
        // is true, said false
        false_negarives: u64,
        // ^impossible
    }

    let results: Vec<ConfusionMatrix> = (0..(1 << search))
        .into_par_iter()
        .fold(
            || ConfusionMatrix::default(),
            |mut arg, i| {
                let decision: bool = Number::scan_perfect_square(i); // predicted is perf_sqrt

                let sqrt: i64 = (i as f64).sqrt().floor() as i64;
                //let sqrt: i64 = i.integer_sqrt();
                let true_ground: bool = sqrt * sqrt == i;

                match (decision, true_ground) {
                    (true, true) => arg.true_positives += 1,
                    (true, false) => arg.false_positives += 1,
                    (false, true) => arg.false_negarives += 1,
                    (false, false) => arg.true_negatives += 1,
                }
                arg
            },
        )
        .collect::<Vec<ConfusionMatrix>>();

    let number_tests: i64 = (2 as i64).pow(search);
    println!("\n2^{} = {}", search, number_tests);
    println!("Partial results: {:#?} structs. \n\n", results.len());

    let reduced: ConfusionMatrix = results
        .into_iter()
        .reduce(|acc, mut e| {
            e.true_positives += acc.true_positives;
            e.true_negatives += acc.true_negatives;
            e.false_positives += acc.false_positives;
            e.false_negarives += acc.false_negarives;
            e
        })
        .expect("We should have something here. ");

    println!("Confusion matrix: \n{:#?}", reduced);
    let sum: u64 = reduced.true_positives
        + reduced.true_negatives
        + reduced.false_positives
        + reduced.false_negarives;
    println!(
        "Checksum: {:#?} == {} = {}\nAccuracy: {}%",
        sum,
        number_tests,
        sum as i64 == number_tests,
        100.0 * (reduced.true_positives + reduced.true_negatives) as f64 / sum as f64
    );

    if reduced.true_positives + reduced.true_negatives == sum {
        println!("SUCCESS! ");
    }

    println!("\n");
    panic!("I need to know the results! ");

    /*

    Basic test: (checking number finishes with ...001r00)
        Accuracy: 83.33638496696949%  (2^30)

    Test + Binary search 1:
        Accuracy: 99.99695951119065%  (2^30)
        Accuracy: 99.99946101743262%  (2^35)
    Test + Binary search corrected:
        Accuracy: 100%      (2^22)
        Accuracy: 100%      (2^30)
        Accuracy: 100%      (2^35)
        2^40 ~8.78h

     */
}

#[test]
#[ignore = "Only used for researching. "]
fn pattern_perf_squares() {
    // This test attemps to see if there is any pattern in k':
    // Q = x * x = (8*k' + 1) * 2^(2*n)
    // for some integer k' and n.

    let num_squares: i64 = 1 << 16;

    let mut k_vec: Vec<(usize, i64)> = (1..=num_squares)
        .into_par_iter()
        .map(|x: i64| (x as usize, x * x))
        .map(|mut x: (usize, i64)| {
            while (x.1 & (0b11 as i64)) == 0 {
                x.1 = x.1 >> 2;
            }
            x
        })
        .map(|x: (usize, i64)| (x.0, x.1 >> 3))
        .collect::<Vec<(usize, i64)>>();

    k_vec.sort_unstable_by(|a: &(usize, i64), b: &(usize, i64)| a.0.cmp(&b.0));
    // k_vec contains (the original number, the k' value of it's square)

    let table: Vec<(usize, i64, i32)> = k_vec.into_iter().fold(
        Vec::new(),
        |mut acc: Vec<(usize, i64, i32)>, x: (usize, i64)| {
            // (least_num, number, count)
            let element_opt: Option<&mut (usize, i64, i32)> =
                acc.iter_mut().find(|y: &&mut (usize, i64, i32)| y.1 == x.1);

            if let Some(element) = element_opt {
                element.2 += 1;
            } else {
                acc.push((x.0, x.1, 1));
            }
            acc
        },
    );

    let mut acc: usize = 0;
    println!("Results: \n");
    for (i, (index, v, c)) in table.iter().take(800).enumerate() {
        let a: usize = acc + i;
        println!("{}: \t{}  \tx{}, ", index, v, c);
        acc = a
    }

    println!(
        "\nThe correlation holds: {}\n",
        table
            .iter()
            .enumerate()
            .fold((true, 0), |acc, x| {
                let a = acc.1 + x.0; // counter + i
                let t = acc.0 && (a == (x.1).1 as usize);
                (t, a)
            })
            .0
    );
}

#[test]
fn test_ilogs() {
    /*
        Integer logatrithm test

    Target: for a given x, figure out some upper and lower bound for sqrt(x) given
    ilog2(x) is also known. This must be performant.

    0 not allowed into ilog

    Given some 0 < x,

    */

    let mut v: Vec<i32> = vec![
        1,
        2,
        3,
        4,
        5,
        10,
        15,
        20,
        63,
        64,
        155,
        255,
        256,
        257,
        916,
        33 * 2,
        45 * 4,
        534,
        5436,
        12 * 12,
    ];

    v.sort_unstable();

    let r: Vec<(i32, u32)> = v
        .iter()
        .map(|&x| (x, i32::ilog2(x)))
        .collect::<Vec<(i32, u32)>>();
    let t: Vec<((i32, u32), (i32, u32), (u32, u32))> = v
        .iter()
        .map(|&x| (x, x * x))
        .map(|a| {
            let (x, y) = a;
            //(x, y, i32::ilog2(x), i32::ilog2((y as f64).sqrt() as i32))
            let log_y: u32 = i32::ilog2(y);
            let lower: u32 = 1u32 << (log_y >> 1); //left
            let upper: u32 = lower << 2; // right
            ((x, i32::ilog2(x)), (y, log_y), (lower, upper))
        })
        .collect::<Vec<((i32, u32), (i32, u32), (u32, u32))>>();

    println!("{:?}\n\n", r);

    t.iter().for_each(|x|
        //println!("{:?}", x)
        println!("sqrt({}) = {} in [{}, {}]\t", x.1.0, x.0.0, x.2.0, x.2.1));

    let proof: bool = t
        .iter()
        .all(|x| x.2 .0 <= x.0 .0 as u32 && x.0 .0 as u32 <= x.2 .1);
    println!("Condition: {}", proof);

    assert!(proof);
}

#[test]
fn derive_test() {
    // x + 3
    let tree1: AST = {
        AST {
            value: Element::Add,
            children: vec![
                Rc::new(RefCell::new(AST {
                    value: Element::Var,
                    children: Vec::new(),
                })),
                Rc::new(RefCell::new(AST::from_number(Number::Rational(3, 1)))),
            ],
        }
    }
    .insert_derive();

    //x^2 * 3
    let tree2: AST = {
        let x: AST = AST {
            value: Element::Var,
            children: Vec::new(),
        };

        let x_sq: AST = AST {
            value: Element::Exp,
            children: vec![
                get_ptr(x),
                get_ptr(AST::from_number(Number::Rational(2, 1))),
            ],
        };

        AST {
            value: Element::Mult,
            children: vec![
                get_ptr(x_sq),
                get_ptr(AST::from_number(Number::Rational(3, 1))),
            ],
        }
    }
    .insert_derive();

    //x*arccos(x)^10
    let tree3: AST = {
        let arccos: AST = AST {
            value: Element::Function(String::from("arccos")),
            children: vec![get_ptr(AST_VAR.clone())],
        };

        let arccos_10: AST = AST {
            value: Element::Exp,
            children: vec![
                get_ptr(arccos),
                get_ptr(AST::from_number(Number::Rational(10, 1))),
            ],
        };

        AST {
            value: Element::Mult,
            children: vec![get_ptr(AST_VAR.clone()), get_ptr(arccos_10)],
        }
    }
    .insert_derive();

    // tree, evaluation point, result
    let tree_list: Vec<(AST, Number, Number)> = vec![
        (tree1, Number::Rational(1, 1), Number::Rational(1, 1)),
        (tree2, Number::Rational(1, 1), Number::Rational(6, 1)),
        (
            tree3,
            Number::Rational(1, 2),
            Number::Real(-2.785929065537081),
        ),
    ];

    for (ast, eval_point, solution) in tree_list {
        let (der, _): (AST, bool) = match ast.execute_derives(false, false) {
            Ok(v) => v,
            Err(e) => panic!(
                "{} ||| {} ({:?}) = {:?}",
                e,
                ast.to_string(),
                eval_point,
                solution
            ),
        };

        println!("AST: {}\n", ast.to_string());
        println!("Der AST: {}\n", der.to_string());

        let awnser: Number = match der.evaluate(Some(eval_point.clone())) {
            Ok(v) => v,
            Err(e) => panic!(
                "Error: {} ||| {} ({:?}) = {:?}",
                e,
                der.to_string(),
                eval_point,
                solution
            ),
        };

        if !awnser.in_tolerance_range(&solution, 0.0000001) {
            panic!(
                "Non-equal results. Got {:?} ||| {} ({:?}) = {:?}",
                awnser,
                der.to_string(),
                eval_point,
                solution
            );
        }
    }

    //panic!();
}

#[test]
fn is_constant_expected_test() {
    let mut rng: ThreadRng = thread_rng();
    let f: fn(&Number) -> Option<&str> = functions::Constants::is_constant;
    let desviation: f64 = 0.00001;
    let mut add_rand = || (rng.gen::<f64>() * 2.0 - 1.0) * desviation;
    // addrand gives an uniformly distributed value in [-desviation, desviation]

    let iters: usize = 10000;
    for _i in 0..iters {
        let (num, s): (f64, &str) = (core::f64::consts::PI, "pi");
        if let Some(const_str) = f(&Number::new_real(num + add_rand())) {
            if const_str != s {
                panic!()
            }
        } else {
            panic!()
        }

        let (num, s): (f64, &str) = (core::f64::consts::PI / 180 as f64, "deg2rad");
        if let Some(const_str) = f(&Number::new_real(num + add_rand())) {
            if const_str != s {
                panic!()
            }
        } else {
            panic!()
        }

        let (num, s): (f64, &str) = (180.0 as f64 / core::f64::consts::PI, "rad2deg");
        if let Some(const_str) = f(&Number::new_real(num + add_rand())) {
            if const_str != s {
                panic!()
            }
        } else {
            panic!()
        }

        let (num, s): (f64, &str) = ((1.0 as f64 + (5.0 as f64).sqrt()) / 2.0 as f64, "phi");
        if let Some(const_str) = f(&Number::new_real(num + add_rand())) {
            if const_str != s {
                panic!()
            }
        } else {
            panic!()
        }
    }


}

#[test]
fn is_constant_unexpected_test() {

    let f: fn(&Number) -> Option<&str> = functions::Constants::is_constant;

    let failable_part: bool = false; 

    if failable_part {
        // should pass most times (0.6 < )
        let iters: usize = 1 << 16;
        let _ = (0..iters).into_par_iter().for_each_init(
            || rand::thread_rng(),
            |rand_gen, _i| {
                let num: f64 = rand_gen.gen::<f64>() * 10.0;
                if let Some(s) = f(&Number::Real(num)) {
                    panic!("Number: {} \t classifed as: {}", num, s);
                }
            },
        );
    }
        
    let iters: usize = 1 << 24;
    let _ = (0..iters).into_par_iter().for_each_init(
        || rand::thread_rng(),
        |rand_gen, _i| {
            let num: f64 = rand_gen.gen::<f64>() * 90.0+10.0;
            if num as i32 != 57 {
                if let Some(s) = f(&Number::Real(num)) {
                    panic!("Number: {} \t classifed as: {}", num, s);
                }
            }
        },
    );

    let iters: usize = 1 << 24;
    let _ = (0..iters).into_par_iter().for_each_init(
        || rand::thread_rng(),
        |rand_gen, _i| {
            let num: f64 = rand_gen.gen::<f64>() * -10.0;
            if let Some(s) = f(&Number::Real(num)) {
                panic!("Number: {} \t classifed as: {}", num, s);
            }
        },
    );

}

#[test]
fn rand_quick() {
    let mut rng: ThreadRng = thread_rng();
    let desviation: f64 = 1.0;
    let mut add_rand = || (rng.gen::<f64>() * 2.0 - 1.0) * desviation;

    for _i in 0..20 {
        println!("{}", add_rand());
    }

    panic!()
}

/*

Maybe formalize latter:

der(x*tan(x))

=> 1*tan(x)+(tan(x)^2+1)*1*x
tan(x)+(tan(x)^2+1)*x
tan(x) + x/cos(x)^2

====================

der(x^(arcsin(x)^2))

=> x^arcsin(x)^2*(1*arcsin(x)^2/x+2*arcsin(x)^1*1/sqrt(1-x^2)*ln(x))
x^arcsin(x)^2*(arcsin(x)^2/x+2*arcsin(x)/sqrt(1-x^2)*ln(x))
x^arcsin(x)^2*(arcsin(x)^2/x+2*arcsin(x)*ln(x)/sqrt(1-x^2))

================

der(8^cos(e^x) + sqrt(x^2 - ln(x)) + 11/2 * 8*x * e^x * ln(2*x + 1))
der(8^cos(e^x))     => 8^(cos(e^x)*2.0794)*-1*sin(e^x)*e^x*1*1

-8^cos(e^x)*ln(8)*sin(e^x)*e^x

correct: -ln(8)*e^x*8^cos(e^x) *sin(e^x)

der(sqrt(x^2 - ln(x))) => (2*x-1/x)/(2*sqrt(x^2-ln(x)))


der(11/2 * 8*x * e^x * ln(2*x + 1))
    => ((((0*2-0*11)/2^2*8+0*11/2)*x+1*11/2*8)*e^x+e^x*1*1*11/2*8*x)*ln(2*x+1)+(0*x+1*2+0)/(2*x+1)*11/2*8*x*e^x
    => (((11/2*8)*e^x+e^x*11/2*8*x)*ln(2*x+1)+(1*2)/(2*x+1)*11/2*8*x*e^x
^correct

===================

der(sin(raíz(e^x + pi) / 2))

=> cos(sqrt(e^x+pi)/2)*((e^x*1*1+0)/(2*sqrt(e^x+pi))*2-0*sqrt(e^x+pi))/2^2
cos(sqrt(e^x+pi)/2)*(e^x/sqrt(e^x+pi))/4

der(der(sin(sqrt(e^x + pi) / 2)))

=> -1*sin(sqrt(e^x+pi)/2)*((e^x*1*1+0)/(2*sqrt(e^x+pi))*2-0*sqrt(e^x+pi))/2^2*e^x/sqrt(e^x+pi)/4+((e^x*1*1*sqrt(e^x+pi)-(e^x*1*1+0)/(2*sqrt(e^x+pi))*e^x)/sqrt(e^x+pi)^2*4-0*e^x/sqrt(e^x+pi))/4^2*cos(sqrt(e^x+pi)/2)
-sin(sqrt(e^x+pi)/2)*e^x/sqrt(e^x+pi)/4*e^x/sqrt(e^x+pi)/4+((e^x*sqrt(e^x+pi)-e^x/(2*sqrt(e^x+pi))*e^x)/sqrt(e^x+pi)^2*4)/16*cos(sqrt(e^x+pi)/2)

-1*sin(sqrt(e^x+pi)/2)*e^x/(2*sqrt(e^x+pi))*2/4*e^x/(2*sqrt(e^x+pi))*2/4+(e^x*2*sqrt(e^x+pi)-e^x/(2*sqrt(e^x+pi))*2*e^x)/(2^2*sqrt(e^x+pi)^2)*2*4/16*cos(sqrt(e^x+pi)/2)


====================

// 3^4 = 81

(x*3)^4

=> x^4*3^4 => x^4*81

(x/2)^5

=> x^5/32



*/
