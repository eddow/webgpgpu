/* TODO code: something much more powerful with: input/output definitions, "usage" and a preprocessor with:
#import "..."
#init
#compute
#input someVar: f32[threads.x, 10]
#output someOtherVar: someOtherType
//*/
export interface CodeParts {
	declaration?: string
	initialization?: string
	computation?: string
}
export abstract class WgslCodeGenerator {
	protected abstract getImport(name: PropertyKey): CodeParts
	constructor(
		protected readonly definitions: readonly CodeParts[],
		protected readonly importUsage: Iterable<PropertyKey>
	) {}

	protected get allEntries() {
		return [...this.definitions, ...[...this.importUsage].map((name) => this.getImport(name))]
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

export function preprocess(code: string): CodeParts {
	return { declaration: code }
}
