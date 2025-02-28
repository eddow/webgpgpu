export abstract class Indexable<T> {
	[index: number]: T
	abstract getAtIndex(index: number): T
}
Object.setPrototypeOf(
	Indexable.prototype,
	new Proxy(Indexable.prototype, {
		get(target, propKey, receiver) {
			if (typeof propKey === 'symbol') return undefined
			const index = Number(propKey)
			if (Number.isNaN(index)) return undefined
			return receiver.getAtIndex(index)
		},
	})
)

class MyClass extends Indexable<number> {
	getAtIndex(index: number): number {
		return index + 1
	}
}

const myInstance = new MyClass()
const t = myInstance[3]
console.log(myInstance[5])
