import { Indexable } from '../hacks'
import { type AnyInference, type SizeSpec, assertSize } from '../inference'
import { log } from '../log'
import { InferenceValidationError, type InputXD, ParameterError } from '../types'
export type Writer<Element> = (index: number, value: Element) => void
export type Reader<Element> = (index: number) => Element

type SubSpec<Spec> = Spec extends readonly [number, ...infer Rest extends number[]]
	? Rest extends []
		? never
		: Rest | SubSpec<Rest>
	: number[]
type SubtractLengths<A extends readonly any[], B extends readonly any[]> = A extends [
	...B,
	...infer Rest,
]
	? Rest
	: never

function assertElementSize(given: any, expected: number) {
	if (given !== expected)
		throw new ParameterError(`Element size mismatch: ${given} received while expecting ${expected}`)
}

function bufferPosition(index: readonly number[], size: readonly number[]): number {
	let pos = 0
	//for (let i = size.length - 1; i >= 0; i--) {
	for (let i = 0; i < size.length; i++) {
		if (index[i] >= size[i]) throw new ParameterError(`Index out of range (${index[i]}/${size[i]})`)
		pos *= size[i]
		pos += index[i]
	}
	return pos
}

function nextXdIndex(index: number[], size: readonly number[]): boolean {
	for (let i = index.length - 1; i >= 0; i--) {
		index[i]++
		if (index[i] < size[i]) return true
		index[i] = 0
	}
	return false
}

function dot(a: readonly number[], b: readonly number[]): number {
	let sum = 0
	const length = Math.min(a.length, b.length)
	for (let i = 0; i < length; i++) sum += a[i] * b[i]
	return sum
}

function prod(a: readonly number[], product = 1): number {
	return a.reduce((a, b) => a * b, product)
}

/* TODO: more complex/"asynchronous" inference
Here: Checking the first element if multi-dimensional - throw if ArrayBuffer is given and not all inferred
*/
function inferSizes<
	Inferences extends AnyInference,
	Element,
	SizesSpec extends readonly SizeSpec<Inferences>[],
>(
	bytesPerElement: number,
	data: InputXD<Element, SizesSpec>,
	size: SizesSpec,
	inferences: Inferences,
	reason: string,
	reasons: Record<string, string>
) {
	let rv: number[] = []
	switch (size.length) {
		case 0:
			return []
		case 1:
			if (data && typeof data === 'object' && 'byteLength' in data) {
				if (data.byteLength % bytesPerElement !== 0)
					throw new InferenceValidationError(
						`Size mismatch in dimension 1: ${data.byteLength} is not a multiple of ${bytesPerElement} (bytes)`
					)
				rv = [data.byteLength / bytesPerElement]
			} else if (Array.isArray(data)) {
				rv = [data.length]
			}
			assertSize(rv, size, inferences, reason, reasons)
			break
		default:
			if (data instanceof ArrayBuffer) {
				const unknown = size.filter((v) => typeof v === 'string' && inferences[v] === undefined)
				if (unknown.length)
					throw new InferenceValidationError(
						`Cannot infer size of a raw ArrayBuffer. Unknown inferences: ${unknown.join(', ')}`
					)
			} else if (Array.isArray(data)) {
				assertSize([data.length], size.slice(0, 1), inferences, reason, reasons)
				// TODO: case when data.length === 0
				rv = [
					data.length,
					...inferSizes(bytesPerElement, data[0], size.slice(1), inferences, reason, reasons),
				]
			}
			break
	}
	return rv
}

function writeInputData<Element, SizesSpec extends readonly number[]>(
	target: ArrayBuffer,
	bytesPerElement: number,
	offset: number,
	data: InputXD<Element, any>,
	sizes: SizesSpec,
	write: (index: number, value: Element) => void
) {
	if (data instanceof ArrayBuffer) {
		data = {
			buffer: data,
			byteOffset: 0,
			byteLength: data.byteLength,
		}
	}
	if (typeof data === 'object' && data && 'buffer' in data) {
		// offset & length of the input
		const { buffer, byteOffset, byteLength } = data as ArrayBufferView
		offset *= bytesPerElement
		// TODO: validate size
		const round = [3, 2, 1].find((i) => ((offset | byteOffset) & ((1 << i) - 1)) === 0)
		if (!round) new Uint8Array(target).set(new Uint8Array(buffer, byteOffset, byteLength), offset)
		else {
			// offset in bytes, length in `elements`
			const typedArray = [undefined, Uint16Array, Uint32Array, BigUint64Array][round] as new (
				buffer: ArrayBuffer,
				offset: number,
				length: number
			) => Uint16Array | Uint32Array | BigUint64Array

			const from = data.byteLength & ~((1 << round) - 1)
			if (from > 0)
				new typedArray(target, offset, from >> round).set(
					//@ts-expect-error typedArray = typedArray, explain this...
					new typedArray(buffer, byteOffset, from >> round)
				)
			const remaining = data.byteLength & ((1 << round) - 1)
			if (remaining)
				new Uint8Array(target, from + offset).set(
					new Uint8Array(buffer, byteOffset + from, remaining)
				)
		}
	} else if (!sizes.length) {
		write(offset, data as Element)
	} else if (Array.isArray(data)) {
		if (data.length !== sizes[0])
			// This is not an inference error because it was inferred with a parent input - so, the whole input is inconsistent
			throw new ParameterError(
				`Size mismatch: ${data.length} elements were expected, got ${sizes[0]} elements`
			)
		const subSizes = sizes.slice(1)
		const stride = prod(subSizes, 1)
		for (const element of data) {
			writeInputData(target, bytesPerElement, offset, element, subSizes, write)
			offset += stride
		}
	} else {
		throw new ParameterError('Invalid input')
	}
}

// Poorly typed but for internal use only
export function toArrayBuffer<
	Inferences extends AnyInference,
	Element,
	SizesSpec extends readonly SizeSpec<Inferences>[],
>(
	bytesPerElement: number,
	data: InputXD<Element, SizesSpec>,
	sizesSpec: SizesSpec,
	writer: (buffer: ArrayBuffer) => (index: number, value: Element) => void,
	inferences: Inferences,
	reason: string,
	reasons: Record<string, string>
): ArrayBuffer {
	const sizes = inferSizes(bytesPerElement, data, sizesSpec, inferences, reason, reasons)
	const buffer = new ArrayBuffer(prod(sizes, bytesPerElement))
	const write = writer(buffer)
	writeInputData(buffer, bytesPerElement, 0, data, sizes, write)
	return buffer
}
const customInspectSymbol = Symbol.for('nodejs.util.inspect.custom')

type IndexableReturn<Element, InputSpec extends readonly number[]> = InputSpec extends [number]
	? Element
	: InputSpec extends [number, ...infer Rest extends number[]]
		? IBufferReader<Element, Rest>
		: IBufferReader<Element, number[]> | Element
// TODO: should have all the info even for children browsing cached & given when constructing sub-buffer as {me, subs} (no more array slice)
export interface IBufferReader<Element = any, InputSpec extends readonly number[] = number[]> {
	/**
	 * Retrieve an element by its index
	 * @param index The index of the element to retrieve
	 * @returns
	 */
	at(...index: InputSpec): Element
	keys(): IterableIterator<InputSpec>
	values(): IterableIterator<Element>
	entries(): Generator<[InputSpec, Element]>
	length: number
	stride: InputSpec
	flatLength: number
	flat(): Element[]
	section<SSpec extends SubSpec<InputSpec>>(
		...sSpec: SSpec
	): IBufferReader<Element, SubtractLengths<InputSpec, SSpec>>
	[key: number]: IndexableReturn<Element, InputSpec>
}
export class BufferReader<Element = any, InputSpec extends readonly number[] = number[]>
	extends Indexable<IndexableReturn<Element, InputSpec>>
	implements IBufferReader<Element, InputSpec>
{
	constructor(
		protected readonly read: Reader<Element>,
		public readonly buffer: ArrayBuffer,
		public readonly sizes: InputSpec,
		public readonly offset: number = 0
	) {
		super()
	}
	/**
	 * Take an element by its index - given `least significant` first:
	 * @param index The index of the element to retrieve
	 * @returns
	 */
	at(...index: InputSpec): Element {
		const { sizes, offset } = this
		if (index.length !== sizes.length)
			throw new InferenceValidationError(
				`Index length mismatch: taking ${index.length}-D index element out of ${sizes.length}-D buffer`
			)
		return this.read(bufferPosition(index, sizes) + offset)
	}
	*keys(): IterableIterator<InputSpec> {
		const { sizes } = this
		if (sizes.some((s) => s === 0)) {
			log.warn('Zero size buffer as output')
			return
		}
		const index: number[] = sizes.map(() => 0)
		do {
			const copy = [...index] as const
			yield copy as InputSpec
		} while (nextXdIndex(index, sizes))
	}
	/**
	 * Retrieves the [index, value] pairs, where `index` is the index array and `value` is the value at that index
	 * @returns Generator<[InputSpec, Element]>
	 */
	*entries(): Generator<[InputSpec, Element]> {
		let pos = this.offset
		for (const index of this.keys()) yield [index as InputSpec, this.read(pos++)]
	}
	// TODO: Cache these
	get flatLength(): number {
		return prod(this.sizes)
	}
	get length(): number {
		return this.sizes[0]
	}
	get stride(): InputSpec {
		const { sizes } = this
		let remaining = this.flatLength
		return [
			...sizes.map((s) => {
				remaining /= s
				return remaining
			}),
		] as const as InputSpec
	}
	*values() {
		const { flatLength: length, offset } = this
		for (let pos = offset; pos < length; ++pos) yield this.read(pos)
	}
	flat(): Element[] {
		return [...this.values()]
	}
	/**
	 * Retrieves a subsection of the buffer.
	 * This does not create a copy of the underlying buffer, it just creates a view on top of it.
	 * @param sSpec The specification of the section to take
	 * @returns
	 * @example myBuffer.section(x,y).at(z,w) == myBuffer.at(x,y,z,w)
	 */
	section<SSpec extends SubSpec<InputSpec>>(
		...sSpec: SSpec
	): IBufferReader<Element, SubtractLengths<InputSpec, SSpec>> {
		const { stride, buffer, sizes } = this
		return new BufferReader(
			this.read,
			buffer,
			sizes.slice(sSpec.length) as SubtractLengths<InputSpec, SSpec>,
			this.offset + dot(sSpec, stride)
		)
	}
	getAtIndex(index: number): any {
		// @ts-expect-error We know InputSpec but only programmatically
		return this.sizes.length === 1 ? this.at(index) : this.section(index)
	}
	// TODO: node displaying a BufferReader throws [Array: Inspection interrupted prematurely. Maximum call stack size exceeded.]
	toString(): string {
		return 'BufferReader'
	}
	valueOf(): Element[] {
		if (this.sizes.length) throw new TypeError('Cannot convert BufferReader to primitive')
		//@ts-expect-error we know it's a number
		return this.at()
	}
	get [Symbol.toStringTag]() {
		return 'BufferReader'
	}

	//[customInspectSymbol](/*depth, inspectOptions, inspect*/) {
	//	return 'BufferReader'
	//}
	// TODO: Implement Array<...>
}
