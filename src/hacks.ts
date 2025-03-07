// #region Indexable

function prop2index(propKey: PropertyKey, ignore?: boolean): number {
	return ignore || typeof propKey === 'symbol' ? Number.NaN : Number(propKey)
}
export abstract class Indexable<T> implements ArrayLike<T> {
	abstract get length(): number
	[index: number]: T
	/**
	 * Override here.
	 * Let to undefined for the default behavior. (returning undefined if not found)
	 * @param index The index of the item to get.
	 */
	getAtIndex?(index: number): T
	/**
	 * Override here.
	 * Let to undefined for the default behavior: creating a new property in the instance if not found
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
			const index = prop2index(propKey, !receiver.getAtIndex)
			return Number.isNaN(index) ? undefined : receiver.getAtIndex(index)
		},
		set(target, propKey, value, receiver) {
			const index = prop2index(propKey, !receiver.setAtIndex)
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
		has(target, propKey) {
			const index = prop2index(propKey)
			return !Number.isNaN(index) && index < target.length
		},
	})
)

// #endregion
// #region shortcuts

export function mapEntries<From, To, Keys extends PropertyKey>(
	obj: { [key in Keys]: From },
	fn: (value: From, key: PropertyKey) => To | undefined
): { [key in Keys]: To } {
	return Object.fromEntries(
		Object.entries(obj)
			.map(([key, value]: [PropertyKey, unknown]) => [key, fn(value as From, key)])
			.filter(([_, value]) => value !== undefined)
	) as { [key in Keys]: To }
}

export function defined<T>(v: T | undefined): v is T {
	return v !== undefined
}

export function elements<Key extends PropertyKey, Item extends { [k in Key]?: unknown }>(
	items: Iterable<Item>,
	k: Key
): Exclude<Item[Key], undefined>[] {
	return Array.from(items, (item) => item[k]).filter(defined) as Exclude<Item[Key], undefined>[]
}

// #endregion
// #region decorators

/**
 * Cache a property on get.
 * To cache manually, `propertyCache` can be used.
 * `isPropertyCached` can be used to check if a property has been cached or not.
 * To un-cache, deletion is enough - `delete obj.cachedProperty`
 *
 * @example
 * ```ts
 * class MyClass {
 *   @cached('a', 'b')
 *   get c() {
 *     return this.a + this.b;
 *   }
 * }
 *
 * const myInstance = new MyClass();
 *
 * // First time, the method is executed and the result is cached.
 * console.log(myInstance.c()); // Output: 5
 *
 * // Second time, the cached value is returned without executing the method again.
 *
 * ```
 * @param needed The names of the properties on which depends the result of the getter.
 * @returns
 */
export function cached<T>(...needed: PropertyKey[]) {
	return (original: () => T, context: ClassGetterDecoratorContext<unknown, T>) => {
		return function (this: any) {
			const missing = needed.filter((p) => !isPropertyCached(this, p))
			const stringName = context.name.toString()
			if (missing.length)
				throw new Error(`Missing properties to calculate ${stringName}: ${missing.join(', ')}`)
			const rv = original.call(this)
			propertyCache(this, context.name, rv)
			return rv
		}
	}
}

export function isPropertyCached(object: Object, propertyKey: PropertyKey) {
	return !!Object.getOwnPropertyDescriptor(object, propertyKey)
}

export function propertyCache(object: Object, propertyKey: PropertyKey, value: any) {
	Object.defineProperty(object, propertyKey, { value })
}

// #endregion
