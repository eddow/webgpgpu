export abstract class Indexable<T> {
	[index: number]: T
	abstract getAtIndex(index: number): T
}
Object.setPrototypeOf(
	Indexable.prototype,
	new Proxy(Indexable.prototype, {
		get(target, propKey, receiver) {
			if (typeof propKey === 'symbol') return (target as any)[propKey]
			const index = Number(propKey)
			if (Number.isNaN(index)) return (target as any)[propKey]
			return receiver.getAtIndex(index)
		},
	})
)
