import { Lexer, createToken } from 'chevrotain'

// --------------------- Atomic Type Keywords ---------------------
const types = [
	'bool',
	'f16',
	'f32',
	'i32',
	'u32',
	'vec2f',
	'vec2i',
	'vec2u',
	'vec2h',
	'vec3f',
	'vec3i',
	'vec3u',
	'vec3h',
	'vec4f',
	'vec4i',
	'vec4u',
	'vec4h',
	// TODO: double-check if mat3x3h exist
	'mat2x2f',
	'mat2x3f',
	'mat2x4f',
	'mat3x2f',
	'mat3x3f',
	'mat3x4f',
	'mat4x2f',
	'mat4x3f',
	'mat4x4f',
] as const
export type Type = Record<(typeof types)[number], ReturnType<typeof createToken>>
export const typeTokens = Object.fromEntries(
	types.map((name) => [name, createToken({ name: `TYPE_${name.toUpperCase()}`, pattern: name })])
) as Type

// --------------------- Control Keywords ---------------------
const controlKeywords = [
	'fn',
	'let',
	'const',
	'var',
	'override',
	'return',
	'if',
	'else',
	'switch',
	'case',
	'default',
	'for',
	'while',
	'loop',
	'break',
	'continue',
	'fallthrough',
	'discard',
	'continuing',
	'enable',
	'type',
	'uniform',
	'sampler',
	'sampler_comparison',
	'texture_1d',
	'texture_2d',
	'texture_2d_array',
	'texture_3d',
	'texture_cube',
	'texture_cube_array',
	'texture_multisampled_2d',
	'texture_storage_1d',
	'texture_storage_2d',
	'texture_storage_2d_array',
	'texture_storage_3d',
	'texture_depth_2d',
	'texture_depth_2d_array',
	'texture_depth_cube',
	'texture_depth_cube_array',
	'texture_depth_multisampled_2d',
	'bitcast',
	'true',
	'false',
	'ptr',
	'array',
	'struct',
] as const

export type Keyword = Record<(typeof controlKeywords)[number], ReturnType<typeof createToken>>
export const keywordTokens = Object.fromEntries(
	controlKeywords.map((name) => [
		name,
		createToken({ name: `KW_${name.toUpperCase()}`, pattern: name }),
	])
) as Keyword

// --------------------- Operators & Punctuation ---------------------
const operators = {
	LShift: /<</,
	RShift: />>/,
	LessEqual: /<=/,
	GreaterEqual: />=/,
	EqualEqual: /==/,
	NotEqual: /!=/,
	LParen: /\(/,
	RParen: /\)/,
	LBrace: /\{/,
	RBrace: /\}/,
	LBracket: /\[/,
	RBracket: /\]/,
	Comma: /,/,
	Dot: /\./,
	Semicolon: /;/,
	Colon: /:/,
	Arrow: /->/,
	Plus: /\+/,
	Minus: /-/,
	Multiply: /\*/,
	Divide: /\//,
	Percent: /%/,
	And: /&/,
	Or: /\|/,
	Xor: /\^/,
	Not: /!/,
	LessThan: /</,
	GreaterThan: />/,
	Equal: /=/,
} as const
export type Operator = Record<keyof typeof operators, ReturnType<typeof createToken>>
export const operatorTokens = Object.fromEntries(
	Object.entries(operators).map(([key, value]) => [key, createToken({ name: key, pattern: value })])
) as Operator

// --------------------- Identifiers & Literals ---------------------
const operands = {
	Identifier: /[a-zA-Z_][a-zA-Z0-9_]*/,
	FloatLiteral: /[0-9]*\.[0-9]+/,
	IntLiteral: /0|[1-9][0-9]*/,
} as const
export type Operand = Record<keyof typeof operands, ReturnType<typeof createToken>>
export const operandTokens = Object.fromEntries(
	Object.entries(operands).map(([key, value]) => [key, createToken({ name: key, pattern: value })])
) as Operand

// --------------------- Whitespace & Comments ---------------------
const Whitespace = createToken({ name: 'Whitespace', pattern: /\s+/, group: Lexer.SKIPPED })
const LineComment = createToken({ name: 'LineComment', pattern: /\/\/.*/, group: Lexer.SKIPPED })
const BlockComment = createToken({
	name: 'BlockComment',
	pattern: /\/\*[\s\S]*?\*\//,
	group: Lexer.SKIPPED,
})

// --------------------- Lexer Definition ---------------------
export const tokens = [
	Whitespace,
	LineComment,
	BlockComment,
	...Object.values(typeTokens),
	...Object.values(keywordTokens),
	...Object.values(operandTokens),
	...Object.values(operatorTokens),
]

export const WGSLLexer = new Lexer(tokens)
