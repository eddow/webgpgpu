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

	protected get declarations() {
		return [
			...this.definitions.map(({ declaration }) => declaration).filter(Boolean),
			[...this.importUsage].map((name) => this.getImport(name).declaration).filter(Boolean),
		] as string[]
	}
	protected get initializations() {
		return [
			...this.definitions.map(({ initialization }) => initialization).filter(Boolean),
			[...this.importUsage].map((name) => this.getImport(name).initialization).filter(Boolean),
		] as string[]
	}
}
