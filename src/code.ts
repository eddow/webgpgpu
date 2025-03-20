import { elements } from './hacks'
import { CircularImportError } from './types'

// TODO: separate code mingler from import system?
export interface CodeParts {
	imports?: Iterable<PropertyKey>
	definitions?: Record<string, string>
	declarations?: string
	initializations?: string
	computations?: string
	finalizations?: string
}
export abstract class WgslCodeGenerator {
	static uncommented(code: string, comments: string[]) {
		const commentPattern = /\/\/[^\n]*|\/\*[\s\S]*?\*\//g
		return code.replace(commentPattern, (match) => {
			comments.push(match)
			return `/*${comments.length - 1}*/`
		})
	}
	static commented(code: string, comments: string[]) {
		return code.replace(/\*\/(\d+)\/\*/g, (_, index) => comments[index])
	}
	static define(code: string, definitions: Record<string, string>) {
		const comments: string[] = []
		return WgslCodeGenerator.commented(
			WgslCodeGenerator.uncommented(code, comments).replace(
				/[a-zA-Z_][a-zA-Z0-9_]*/g,
				(match) => definitions[match] ?? match
			),
			comments
		)
	}
	protected abstract getImport(name: PropertyKey): CodeParts
	constructor(
		protected readonly codeParts: readonly CodeParts[],
		protected readonly definitions: Record<string, string>,
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
			if (subImports) this.untangleImports([...subImports], done, [...importing, imp])
			done.push(imp)
		}
	}

	protected weave(computation?: string) {
		const importUsage: PropertyKey[] = []
		this.untangleImports(this.importUsage, importUsage)
		const allEntries = [...importUsage.map((name) => this.getImport(name)), ...this.codeParts]
		//const definitions = parts(({ definitions }) => definitions).reduce((acc, cur) => ({ ...acc, ...cur }), this.definitions)
		const definitions = elements(allEntries, 'definitions').reduce(
			(acc, cur) => ({ ...acc, ...cur }),
			this.definitions
		)
		return {
			declarations: elements(allEntries, 'declarations').join('\n'),
			computation: WgslCodeGenerator.define(
				[
					...elements(allEntries, 'initializations'),
					...elements(allEntries, 'computations'),
					...(computation ? [computation] : []),
					...elements(allEntries, 'finalizations'),
				].join('\n\t\t'),
				definitions
			),
		}
	}
}

export function preprocessWgsl(code: string): CodeParts {
	// Normalize line breaks to \n
	code = code.replace(/[\r\n]+/g, '\n')

	// Regular expressions for matching comments and section markers
	const importPattern = /^\s*@import\s+([\w,\s]+)$/g
	const definePattern = /^\s*@define\s+(\w+)\s+(.+)$/
	const pattern = {
		declare: /^\s*@declare\s*$/g,
		init: /^\s*@init\s*$/g,
		process: /^\s*@process\s*$/g,
		finalize: /^\s*@finalize\s*$/g,
	}

	// Arrays to store the comments and their positions
	const comments: string[] = []
	// Replace comments with placeholders and store the original comments
	const codeWithPlaceholders = WgslCodeGenerator.uncommented(code, comments)

	// List to store imports
	const imports: string[] = []
	// Object to store definitions
	const definitions: Record<string, string> = {}

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
		const analyzeLine = line.replace(/\/\*\d+\*\//g, '')
		const imported = importPattern.exec(analyzeLine)
		if (imported) {
			importPattern.lastIndex = 0
			const importNames = imported[1].split(',').map((name) => name.trim())
			imports.push(...importNames)
			continue
		}
		const defined = definePattern.exec(analyzeLine)
		if (defined) {
			definePattern.lastIndex = 0
			definitions[defined[1]] = defined[2]
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
			sections[s as Section] = sections[s as Section].replace(`/*${c}*/`, comments[c])

	// Return the parsed result
	return {
		imports,
		definitions,
		declarations: WgslCodeGenerator.commented(sections.declare, comments),
		initializations: WgslCodeGenerator.commented(sections.init, comments),
		computations: WgslCodeGenerator.commented(sections.process, comments),
		finalizations: WgslCodeGenerator.commented(sections.finalize, comments),
	}
}
