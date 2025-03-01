export abstract class Indexable<T> {
	[index: number]: T
	/**
	 * Let to undefined for the default behavior. (returning undefined if not found)
	 * @param index The index of the item to get.
	 */
	getAtIndex?(index: number): T
	/**
	 * Let to undefined for the default behavior. (creating a new property if not found)
	 * Note: If default behavior is used and no such property exists, it will be created with this value. After, `getIndex` and `setIndex`
	 * won't be called anymore for that index
	 * @param index The index of the item to set.
	 * @param value The new value of the item at `index`.
	 */
	setAtIndex?(index: number, value: T): void
}
Object.setPrototypeOf(
	Indexable.prototype,
	new Proxy(Indexable.prototype, {
		get(target, propKey, receiver) {
			const index =
				!receiver.getAtIndex || typeof propKey === 'symbol' ? Number.NaN : Number(propKey)
			if (Number.isNaN(index)) return undefined
			return receiver.getAtIndex(index)
		},
		set(target, propKey, value, receiver) {
			const index =
				!receiver.setAtIndex || typeof propKey === 'symbol' ? Number.NaN : Number(propKey)
			if (Number.isNaN(index))
				Object.defineProperty(receiver, propKey, {
					value,
					writable: true,
					configurable: true,
					enumerable: true,
				})
			else receiver.setAtIndex(index, value)
			return true
		},
	})
)
