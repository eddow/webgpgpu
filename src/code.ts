export interface CodeParts {
	declaration?: string
	initialization?: string
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
	protected get initializations() {
		return this.allEntries.map(({ initialization }) => initialization).filter(Boolean) as string[]
	}
}
