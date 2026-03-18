This program allows evaluating numerical expressions, evaluating a given 
function and deriving functions. You can create expressions using basic 
mathematical operations and evaluate and compute it's derivatives. 

Use the character `x` as the variable to create functions. 

***
## Operations and syntax:

This project supports the following operations:

1) Addition                         (using `+`, as in `"2+5"`)
2) Substraction and negation        (using `-`, as in `"3-5.5"` or in `"-sin(-2)"`)
3) Multiplication                   (using `*`, as in `"4*8"`)
4) Division                         (using `/`, as in `"2/9"`)
5) Modulus                          (using `%`, as in `"19%10"`)
6) Exponentiation                   (using `^` or `**` as in `"2^10"` or `"2**10"`)
7) Factorial                        (using `!`, as in `"6!"`)
8) Square root                      (using `sqrt(x)`)
9) Sinus                            (using `sin(x)`)
10) Cosinus                         (using `cos(x)`)
11) Tangent                         (using `tan(x)`)
12) Arcsinus (inverse sinus)        (using `arcsin(x)`)
13) Arccosinus (inverse cosinus)    (using `arccos(x)`)
14) Arctangent (inverse tangent)    (using `arctan(x)`)
15) Exponential                     (using `exp(x)`)
16) Natural logarithm               (using `ln(x)`)
17) Absolute value                  (using `abs(x)`)
18) Floor function                  (using `floor(x)`)
19) Ceil function                   (using `ceil(x)`)
20) Random  (Uniform [0, 1])        (using `rand(x)`)
21) Gamma                           (using `gamma(x)`)

+ All the functions can be combined and composed in any way as long as they are
mathematically correct and fullfill the syntax requirments.

 - Alternatively, use the flag `-d <n>` to derivate the provided functon `<n>` times. 

 - Use `-e x=<y>` to evaluate the function at the input `<y>`. 

+ Some operations have priority over others, such as multiplication over
addition. That means that `"2+5*3"` will be evaluated as `"2+(5*3)"`. To overwrite
the order, parenthesis can be used `()`.

+ All the trigonometric functions work with radians. In order to use degrees, multiply your
value by `DEG2RAD`, for example: `sin(90*DEG2RAD)`

+ Only real values are supported (no complex values), therefore `"sqrt(-1)"` lies outside
the domains of the function and will return an error indicating the invalid
evaluation.

+ Division by 0 is not allowed.

+ Every parenthesis must be closed. Any of `()`, `{}` and `[]` can be used. 
They are equivalent but must to match it's counterpart.

+ Spaces are ignored, you can add all you want or even remove them completly.

+ Remember that a [logarithm](https://en.wikipedia.org/wiki/Logarithm#Change_of_base)
in any base `b` can be expressed as `log_b(x) = (ln(x)/ln(b))` .

***

## Constants:

The program will automatically translate some constants to it's corresponding
numerical values.

1)  [x]  PI              (equal to 3.141592653589793)
2)  [x]  RAD2DEG         (equal to 57.29577951308232 = 180 / PI)
3)  [x]  DEG2RAD         (equal to 0.0174532925199433 = PI / 180)
4)  [x]  phi             (equal to 1.618033988749895 = (1+sqrt(5))/2 )
5)  [x]  e               (equal to 2.718281828459045)
6)  [x]  tau             (equal to 6.283185307179586)
7)  [ ]  gravitational   (equal to 0.000000000066743 = 6.6743 * 10^-11 m^3/(kg * s^2), the gravitational constant)
8)  [ ]  plank           (equal to 0.000000000000000000000000000000000662607015 = 6.62607015 * 10^-34 J*s, Plank constant)
9)  [ ]  light           (equal to 299 792 458 m/s, speed of light)
10) [ ]  elecprem        (equal to 0.0000000000088541878188 = 8.8541878188 * 10^-12, vacuum electric permittivity)
11) [ ]  magnprem        (equal to 0.00000125663706127 = 1.25663706127 * 10^-6, vacuum magnetic permeability)
12) [ ]  elecmass        (equal to 0.00000000000000000000000000000091093837139 = 9.1093837139 * 10^-31 kg, mass of the electron)


Constants can be written on any combination of uppercase and lowercase letters.
Physical constants have [IS units](https://en.wikipedia.org/wiki/International_System_of_Units).
