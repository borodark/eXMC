%%% Lexer for Stan-compatible model DSL.
%%% Compiles via Mix's :erlang compiler (leex).

Definitions.

DIGIT     = [0-9]
ALPHA     = [a-zA-Z_]
ALPHANUM  = [a-zA-Z0-9_]
WS        = [\s\t\r]
NL        = \n

Rules.

%% Line comments
//[^\n]*                : skip_token.

%% Whitespace
{WS}+                  : skip_token.
{NL}+                  : skip_token.

%% Keywords
data                   : {token, {data, TokenLine}}.
parameters             : {token, {parameters, TokenLine}}.
transformed            : {token, {transformed, TokenLine}}.
model                  : {token, {model, TokenLine}}.
generated              : {token, {generated, TokenLine}}.
quantities             : {token, {quantities, TokenLine}}.
real                   : {token, {real, TokenLine}}.
int                    : {token, {int_kw, TokenLine}}.
vector                 : {token, {vector, TokenLine}}.
matrix                 : {token, {matrix, TokenLine}}.
for                    : {token, {for_kw, TokenLine}}.
in                     : {token, {in_kw, TokenLine}}.
lower                  : {token, {lower, TokenLine}}.
upper                  : {token, {upper, TokenLine}}.
target                 : {token, {target, TokenLine}}.

%% Two-char operators (must come before single-char)
\+\=                   : {token, {plus_eq, TokenLine}}.

%% Single-char operators and delimiters
\{                     : {token, {lbrace, TokenLine}}.
\}                     : {token, {rbrace, TokenLine}}.
\(                     : {token, {lparen, TokenLine}}.
\)                     : {token, {rparen, TokenLine}}.
\[                     : {token, {lbracket, TokenLine}}.
\]                     : {token, {rbracket, TokenLine}}.
<                      : {token, {langle, TokenLine}}.
>                      : {token, {rangle, TokenLine}}.
;                      : {token, {semicolon, TokenLine}}.
,                      : {token, {comma, TokenLine}}.
:                      : {token, {colon, TokenLine}}.
~                      : {token, {tilde, TokenLine}}.
=                      : {token, {eq, TokenLine}}.
\+                     : {token, {plus, TokenLine}}.
\-                     : {token, {minus, TokenLine}}.
\*                     : {token, {star, TokenLine}}.
/                      : {token, {slash, TokenLine}}.
\|                     : {token, {pipe, TokenLine}}.
\^                     : {token, {caret, TokenLine}}.
\.                     : {token, {dot, TokenLine}}.

%% Float literals (must come before integer)
{DIGIT}+\.{DIGIT}+([eE][+-]?{DIGIT}+)?  : {token, {float_lit, TokenLine, list_to_float(TokenChars)}}.
{DIGIT}+[eE][+-]?{DIGIT}+               : {token, {float_lit, TokenLine, sci_to_float(TokenChars)}}.

%% Integer literals
{DIGIT}+               : {token, {int_lit, TokenLine, list_to_integer(TokenChars)}}.

%% Identifiers (must come after keywords)
{ALPHA}{ALPHANUM}*     : {token, {ident, TokenLine, to_bin(TokenChars)}}.

Erlang code.

to_bin(L) -> unicode:characters_to_binary(L).

%% "1e5" -> "1.0e5" so list_to_float can handle it
sci_to_float(Chars) ->
  case lists:member($., Chars) of
    true -> list_to_float(Chars);
    false ->
      %% Insert ".0" before the 'e' or 'E'
      {Mantissa, Exp} = lists:splitwith(fun(C) -> C =/= $e andalso C =/= $E end, Chars),
      list_to_float(Mantissa ++ ".0" ++ Exp)
  end.
