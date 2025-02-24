export abstract class WgslCodeGenerator {
	protected abstract getImport(name: PropertyKey): string
	constructor(
		protected readonly definitions: readonly string[],
		protected readonly importUsage: Iterable<PropertyKey>
	) {}

	protected generateHeader() {
		return `
${this.definitions.join('\n')}
${[...this.importUsage].map(this.getImport).join('\n')}
`
	}
}
