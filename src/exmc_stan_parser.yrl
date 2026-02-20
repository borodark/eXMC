%%% Parser for Stan-compatible model DSL.
%%% Compiles via Mix's :erlang compiler (yecc).

Nonterminals
  program blocks block
  data_block parameters_block model_block
  decls decl
  stmts stmt
  type_spec constraints constraint_list constraint
  expr additive_expr multiplicative_expr unary_expr primary_expr
  expr_list
  sampling_stmt.

Terminals
  data parameters transformed model
  real int_kw vector matrix
  for_kw in_kw lower upper target
  lbrace rbrace lparen rparen lbracket rbracket langle rangle
  semicolon comma colon tilde eq plus_eq
  plus minus star slash caret pipe dot
  ident int_lit float_lit.

Rootsymbol program.

program -> blocks : build_program('$1').

blocks -> block blocks : ['$1' | '$2'].
blocks -> block : ['$1'].

block -> data_block : '$1'.
block -> parameters_block : '$1'.
block -> model_block : '$1'.

data_block -> data lbrace decls rbrace : {data_block, '$3'}.
data_block -> data lbrace rbrace : {data_block, []}.

parameters_block -> parameters lbrace decls rbrace : {params_block, '$3'}.
parameters_block -> parameters lbrace rbrace : {params_block, []}.

model_block -> model lbrace stmts rbrace : {model_block, '$3'}.
model_block -> model lbrace rbrace : {model_block, []}.

%% Declarations
decls -> decl decls : ['$1' | '$2'].
decls -> decl : ['$1'].

decl -> type_spec ident semicolon :
  {var_decl, extract_name('$2'), '$1', nil}.

%% Type specs
type_spec -> real : {real, nil}.
type_spec -> real constraints : {real, '$2'}.
type_spec -> int_kw : {int_type, nil}.
type_spec -> int_kw constraints : {int_type, '$2'}.
type_spec -> vector lbracket expr rbracket : {vector_type, '$3', nil}.
type_spec -> vector lbracket expr rbracket constraints : {vector_type, '$3', '$5'}.
type_spec -> matrix lbracket expr comma expr rbracket : {matrix_type, '$3', '$5', nil}.

%% Constraints
constraints -> langle constraint_list rangle : maps:from_list('$2').

constraint_list -> constraint comma constraint_list : ['$1' | '$3'].
constraint_list -> constraint : ['$1'].

constraint -> lower eq expr : {lower, '$3'}.
constraint -> upper eq expr : {upper, '$3'}.

%% Statements
stmts -> stmt stmts : ['$1' | '$2'].
stmts -> stmt : ['$1'].

stmt -> sampling_stmt : '$1'.
stmt -> target plus_eq expr semicolon : {target_incr, '$3'}.

sampling_stmt -> expr tilde ident lparen expr_list rparen semicolon :
  {sample, '$1', extract_name('$3'), '$5'}.
sampling_stmt -> expr tilde ident lparen rparen semicolon :
  {sample, '$1', extract_name('$3'), []}.

%% Expressions — precedence via grammar layers
%% expr > additive > multiplicative > unary > primary
expr -> additive_expr : '$1'.

additive_expr -> additive_expr plus multiplicative_expr : {binop, '+', '$1', '$3'}.
additive_expr -> additive_expr minus multiplicative_expr : {binop, '-', '$1', '$3'}.
additive_expr -> multiplicative_expr : '$1'.

multiplicative_expr -> multiplicative_expr star unary_expr : {binop, '*', '$1', '$3'}.
multiplicative_expr -> multiplicative_expr slash unary_expr : {binop, '/', '$1', '$3'}.
multiplicative_expr -> unary_expr : '$1'.

unary_expr -> minus unary_expr : {neg, '$2'}.
unary_expr -> primary_expr : '$1'.

primary_expr -> lparen expr rparen : '$2'.
primary_expr -> ident lparen expr_list rparen : {call, extract_name('$1'), '$3'}.
primary_expr -> ident lparen rparen : {call, extract_name('$1'), []}.
primary_expr -> ident : {var, extract_name('$1')}.
primary_expr -> int_lit : {lit, extract_value('$1')}.
primary_expr -> float_lit : {lit, extract_value('$1')}.

expr_list -> expr comma expr_list : ['$1' | '$3'].
expr_list -> expr : ['$1'].

Erlang code.

extract_name({ident, _Line, Name}) -> Name.
extract_value({int_lit, _Line, Val}) -> Val;
extract_value({float_lit, _Line, Val}) -> Val.

build_program(Blocks) ->
  Data = proplists:get_value(data_block, Blocks, []),
  Params = proplists:get_value(params_block, Blocks, []),
  Model = proplists:get_value(model_block, Blocks, []),
  {program, Data, Params, Model}.
