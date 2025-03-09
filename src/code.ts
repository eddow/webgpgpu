import { cached, uncacheProperty } from './hacks'
import { CircularImportError } from './types'

export interface CodeParts {
	imports?: Iterable<PropertyKey>
	declarations?: string
	initializations?: string
	computations?: string
	finalizations?: string
}
export abstract class WgslCodeGenerator {
	protected abstract getImport(name: PropertyKey): CodeParts
	constructor(
		protected readonly definitions: readonly CodeParts[],
		protected readonly importUsage: Iterable<PropertyKey>
	) {}

	private untangleImports(
		imports: Iterable<PropertyKey>,
		done: PropertyKey[],
		importing: PropertyKey[] = []
	) {
		for (const imp of imports) {
			if (done.includes(imp)) continue
			const circularIndex = importing.indexOf(imp)
			if (circularIndex !== -1)
				throw new CircularImportError(importing.slice(circularIndex).concat([imp]))
			const subImports = this.getImport(imp).imports
			if (subImports) this.untangleImports([...subImports], [...importing, imp], done)
			done.push(imp)
		}
	}

	@cached()
	protected get allEntries() {
		const importUsage: PropertyKey[] = []
		this.untangleImports(this.importUsage, importUsage)
		return [...importUsage.map((name) => this.getImport(name)), ...this.definitions]
	}
	protected get declarations() {
		return this.allEntries
			.map(({ declarations: declaration }) => declaration)
			.filter(Boolean) as string[]
	}
	protected get initializations() {
		return this.allEntries
			.map(({ initializations: initialization }) => initialization)
			.filter(Boolean) as string[]
	}
	protected get computations() {
		return this.allEntries.map(({ computations }) => computations).filter(Boolean) as string[]
	}
	protected get finalizations() {
		return this.allEntries
			.map(({ finalizations: finalization }) => finalization)
			.filter(Boolean)
			.reverse() as string[]
	}
}

export function preprocessWgsl(code: string): CodeParts {
	// Normalize line breaks to \n
	code = code.replace(/[\r\n]+/g, '\n')

	// Regular expressions for matching comments and section markers
	const commentPattern = /\/\/[^\n]*|\/\*[\s\S]*?\*\//g // Match single-line and multi-line comments
	const importPattern = /^\s*@import\s+([\w,\s]+)$/g
	const pattern = {
		declare: /^\s*@declare\s*$/g,
		init: /^\s*@init\s*$/g,
		process: /^\s*@process\s*$/g,
		finalize: /^\s*@finalize\s*$/g,
	}

	// Arrays to store the comments and their positions
	const comments: string[] = []
	let codeWithPlaceholders = code

	// Replace comments with placeholders and store the original comments
	codeWithPlaceholders = codeWithPlaceholders.replace(commentPattern, (match) => {
		comments.push(match)
		return `<!--COMMENT${comments.length - 1}-->`
	})

	// List to store imports
	const imports: string[] = []

	// Extract section code
	const sections = {
		declare: '',
		init: '',
		process: '',
		finalize: '',
	}
	type Section = keyof typeof sections
	let currentSection: Section = 'declare'

	// Split code into lines
	const lines = codeWithPlaceholders.split('\n')

	for (const line of lines) {
		// Check if the line is inside a comment
		const analyzeLine = line.replace(/<!--COMMENT\d+-->/g, '')
		const imported = importPattern.exec(analyzeLine)
		if (imported) {
			importPattern.lastIndex = 0
			const importNames = imported[1].split(',').map((name) => name.trim())
			imports.push(...importNames)
			continue
		}
		let switchTo: Section | undefined
		for (const s in sections)
			if (pattern[s as Section].test(analyzeLine)) {
				switchTo = s as Section
				break
			}
		if (switchTo) currentSection = switchTo
		// Collect code for the current section
		else sections[currentSection] += `${line}\n`
	}

	// Reconstruct the code with comments back in their places
	for (const c in comments)
		for (const s in sections)
			sections[s as Section] = sections[s as Section].replace(`<!--COMMENT${c}-->`, comments[c])

	// Return the parsed result
	return {
		imports,
		declarations: sections.declare,
		initializations: sections.init,
		computations: sections.process,
		finalizations: sections.finalize,
	}
}
