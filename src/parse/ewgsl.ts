import { CstParser, type IToken } from 'chevrotain'
import {
	Keyword,
	Operand,
	Operator,
	Type,
	WGSLLexer,
	keywordTokens,
	operandTokens,
	operatorTokens,
	tokens,
	typeTokens,
} from './token'
type ASTNode = any // Replace with actual AST type
class EWGSLParser extends CstParser {
	constructor() {
		super(tokens)
		this.performSelfAnalysis()
	}

	public program = this.RULE('program', () => {
		this.MANY(() => this.SUBRULE(this.statement))
	})

	public statement = this.RULE('statement', (): ASTNode => {
		return this.OR([
			{ ALT: () => this.SUBRULE(this.functionDecl) },
			{ ALT: () => this.SUBRULE(this.variableDecl) },
			{ ALT: () => this.SUBRULE(this.assignment) },
			{ ALT: () => this.SUBRULE(this.returnStmt) },
			{ ALT: () => this.SUBRULE(this.expressionStatement) },
		])
	})

	public type = this.RULE('type', (): ASTNode => {
		// Use OR to choose between all possible type tokens
		const token = this.OR(
			Object.values(typeTokens).map((tokenType) => ({
				ALT: () => this.CONSUME(tokenType),
			}))
		)

		// Return the AST node with the type name
		return {
			type: token.image, // e.g., "f32", "vec2"
		}
	})
	public functionDecl = this.RULE('functionDecl', (): ASTNode => {
		this.CONSUME(keywordTokens.fn)
		const name = this.CONSUME(operandTokens.Identifier)
		this.CONSUME(operatorTokens.LParen)
		const params = this.SUBRULE(this.paramList)
		this.CONSUME(operatorTokens.RParen)

		const returnType = this.OPTION(() => {
			this.CONSUME(operatorTokens.Arrow)
			return this.SUBRULE(this.type)
		})

		this.CONSUME(operatorTokens.LBrace)
		const body = this.SUBRULE(this.statement)
		this.CONSUME(operatorTokens.RBrace)

		return {
			type: 'FunctionDeclaration',
			name: name.image,
			params,
			returnType: returnType,
			body,
		}
	})
	public paramList = this.RULE('paramList', () => {
		this.MANY_SEP({
			SEP: operatorTokens.Comma,
			DEF: () => {
				this.CONSUME(operandTokens.Identifier)
				this.CONSUME(operatorTokens.Colon)
				this.SUBRULE(this.type)
			},
		})
	})

	public variableDecl = this.RULE('variableDecl', (): ASTNode => {
		this.CONSUME(keywordTokens.let)
		const name = this.CONSUME(operandTokens.Identifier)
		this.CONSUME(operatorTokens.Colon)
		const type = this.SUBRULE(this.type)
		this.CONSUME(operatorTokens.Equal)
		const value = this.SUBRULE(this.expression)
		this.CONSUME(operatorTokens.Semicolon)
		return { type: 'VariableDeclaration', name: name.image, varType: type, value }
	})

	public assignment = this.RULE('assignment', (): ASTNode => {
		const name = this.CONSUME(operandTokens.Identifier)
		this.CONSUME(operatorTokens.Equal)
		const value = this.SUBRULE(this.expression)
		this.CONSUME(operatorTokens.Semicolon)
		return { type: 'Assignment', name: name.image, value }
	})

	public returnStmt = this.RULE('returnStmt', (): ASTNode => {
		this.CONSUME(keywordTokens.return)
		const value = this.OPTION(() => this.SUBRULE(this.expression))
		this.CONSUME(operatorTokens.Semicolon)
		return { type: 'ReturnStatement', value }
	})

	public expressionStatement = this.RULE('expressionStatement', (): ASTNode => {
		const expr = this.SUBRULE(this.expression)
		this.CONSUME(operatorTokens.Semicolon)
		return expr
	})

	public expression = this.RULE('expression', (): ASTNode => {
		return this.SUBRULE(this.additiveExpression)
	})

	public additiveExpression = this.RULE('additiveExpression', () => {
		this.SUBRULE1(this.multiplicativeExpression)
		this.MANY(() => {
			this.OR([
				{
					ALT: () => {
						this.CONSUME(operatorTokens.Plus)
						this.SUBRULE2(this.multiplicativeExpression)
					},
				},
				{
					ALT: () => {
						this.CONSUME(operatorTokens.Minus)
						this.SUBRULE3(this.multiplicativeExpression)
					},
				},
			])
		})
	})

	public multiplicativeExpression = this.RULE('multiplicativeExpression', () => {
		const firstOperand = this.SUBRULE(this.primaryExpression)
		const rest: { operator: string; operand: ASTNode }[] = []

		const operatorList = this.MANY(() => {
			const operator = this.OR([
				{ ALT: () => this.CONSUME(operatorTokens.Multiply) },
				{ ALT: () => this.CONSUME(operatorTokens.Divide) },
			])
			const operand = this.SUBRULE(this.multiplicativeExpression)

			rest.push({ operator: operator.image, operand })
		})

		return rest.length
			? {
					type: 'MultiplicativeExpression',
					firstOperand,
					rest,
				}
			: firstOperand
	})

	public primaryExpression = this.RULE('primaryExpression', (): ASTNode => {
		return this.OR([
			{ ALT: () => this.CONSUME(operandTokens.Identifier) },
			{ ALT: () => this.CONSUME(operandTokens.FloatLiteral) },
			{ ALT: () => this.CONSUME(operandTokens.IntLiteral) },
		])
	})
}

export const parser = new EWGSLParser()
