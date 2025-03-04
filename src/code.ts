import { CircularImportError } from './types'

export interface CodeParts {
	declaration?: string
	initialization?: string
	computation?: string
	imports?: Iterable<PropertyKey>
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
			if (circularIndex !== -1) throw new CircularImportError(importing.slice(circularIndex))
			const subImports = this.getImport(imp).imports
			if (subImports) this.untangleImports([...subImports], [...importing, imp], done)
			done.push(imp)
		}
	}

	protected get allEntries() {
		const importUsage: PropertyKey[] = []
		this.untangleImports(this.importUsage, importUsage)
		return [...importUsage.map((name) => this.getImport(name)), ...this.definitions]
	}
	protected get declarations() {
		return this.allEntries.map(({ declaration }) => declaration).filter(Boolean) as string[]
	}
	protected get computations() {
		return this.allEntries
			.map(({ computation }) => computation)
			.filter(Boolean)
			.reverse() as string[]
	}
	protected get initializations() {
		return this.allEntries.map(({ initialization }) => initialization).filter(Boolean) as string[]
	}
}

/* TODO code: something much more powerful with: input/output definitions, "usage" and a preprocessor with:
#import "..."
#init
#compute
& ?
#input someVar: f32[threads.x, 10]
#output someOtherVar: someOtherType
*/
export function preprocess(code: string): CodeParts {
	return { declaration: code }
}
