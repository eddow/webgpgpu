import { Indexable } from '../hacks'
import { type AnyInference, type SizeSpec, assertSize } from '../inference'
import { log } from '../log'
import { InferenceValidationError, ParameterError } from '../types'
import type { InputXD } from './buffable'

export interface Writer<Element> {
	write(index: number, value: Element): void
	writeMany?(index: number, values: Element[]): void
}

function assertElementSize(given: any, expected: number) {
	if (given !== expected)
		throw new ParameterError(`Element size mismatch: ${given} received while expecting ${expected}`)
}
// TODO: Optionally give an ArrayBuffer & position
// Poorly typed but for internal use only
export function toArrayBuffer<
	Inferences extends AnyInference,
	Element,
	SizesSpec extends SizeSpec<Inferences>[],
>(
	bytesPerElement: number,
	data: InputXD<Element, SizesSpec>,
	size: SizesSpec,
	writer: (buffer: ArrayBuffer) => Writer<Element>,
	inferences: Inferences,
	reason: string,
	reasons: Record<string, string>
): ArrayBuffer {
	/*if (data instanceof ArrayBuffer && 'elementSize' in data && data.elementSize !== elementSize)
		assertElementSize(data.elementSize, elementSize)*/

	// #region 0D
	if (size.length === 0) {
		if (data instanceof ArrayBuffer) {
			assertElementSize(data.byteLength, bytesPerElement)
			return data
		}
		const buffer = new ArrayBuffer(bytesPerElement)
		const { write } = writer(buffer)
		write(0, data as Element)
		return buffer
	}
	// #endregion 0D
	// #region 1D
	if (size.length === 1) {
		if (data instanceof ArrayBuffer) {
			if (data.byteLength % bytesPerElement !== 0)
				throw new InferenceValidationError(
					`Size mismatch in dimension 1: ${data.byteLength} is not a multiple of ${bytesPerElement} (bytes)`
				)
			assertSize([data.byteLength / bytesPerElement], size, inferences, reason, reasons)
			return data
		}
		if (!Array.isArray(data)) throw new ParameterError('Input is not an array nor a typed array')
		assertSize([data.length], size, inferences, reason, reasons)
		const buffer = new ArrayBuffer(bytesPerElement * data.length)
		const { write, writeMany } = writer(buffer)
		// We assert all the elements have the same type (no mixed number/ArrayBuffer)
		if (writeMany && typeof data[0] === 'number') writeMany(0, data)
		else {
			let dst = 0
			for (const element of data as Element[]) write(dst++, element)
		}
		return buffer
	}
	// #endregion 1D
	// #region 2~3-D
	throw `Not implemented (dimension ${size.length})`
	/*
	if (data instanceof ArrayBuffer) {
		// TODO: multidimensional inferring
		if (!isTypedArrayXD(data))
			throw new InferenceValidationError(
				`When giving a ${size.length}-D typed array as input, the input must have given dimension. Use \`dimensionedArray\``
			)
		if (data.size.length !== size.length)
			throw new InferenceValidationError(
				`Dimensions mismatch: Typed array of dimension ${data.size.length} is used in a ${size.length}D context`
			)
		assertSize(data.size, size, inferences, reason, reasons)
		return data as Buffer
	}
	if (!Array.isArray(data))
		throw new InferenceValidationError('Input is not an array nor a typed array')
	assertSize([data.length], [size[size.length - 1]], inferences, reason, reasons)
	let rv: Buffer | undefined
	let itemSize: number | undefined
	let dst = 0
	for (const element of data) {
		const subBuffer = elementsToTypedArray(
			specification,
			inferences,
			element,
			size.slice(1),
			reason && `[Slice of] ${reason}`,
			reasons
		)
		if (!rv) {
			itemSize = subBuffer.length
			// TODO: multidimensional inferring
			rv = new bufferType(resolvedSize(size, inferences).reduce((a, b) => a * b, 1))
		} else if (subBuffer.length !== itemSize)
			throw new InferenceValidationError(
				`Size mismatch in dimension ${size.length}: Buffer length ${subBuffer.length} was expected to be ${itemSize}`
			)
		rv.set(subBuffer, dst)
		dst += itemSize
	}
	if (!rv) log.warn('Size 0 buffer created')
	return rv ?? new bufferType(0)*/
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

type IndexableReturn<Element, InputSpec extends readonly number[]> = InputSpec extends [number]
	? Element
	: InputSpec extends [number, ...infer Rest extends number[]]
		? BufferReader<Element, Rest>
		: never
export class BufferReader<
	Element = any,
	InputSpec extends readonly number[] = number[],
> extends Indexable<IndexableReturn<Element, InputSpec>> {
	constructor(
		private readonly read: (index: number) => Element,
		public readonly buffer: ArrayBuffer,
		public readonly size: InputSpec,
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
		const { size, offset } = this
		if (index.length !== size.length)
			throw new InferenceValidationError(
				`Index length mismatch: taking ${index.length}-D index element out of ${size.length}-D buffer`
			)
		return this.read(bufferPosition(index, size) + offset)
	}
	*keys(): IterableIterator<InputSpec> {
		const { size } = this
		if (size.some((s) => s === 0)) {
			log.warn('Zero size buffer as output')
			return
		}
		const index: number[] = size.map(() => 0)
		do {
			const copy = [...index] as const
			yield copy as InputSpec
		} while (nextXdIndex(index, size))
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
		return prod(this.size)
	}
	get length(): number {
		return this.size[0]
	}
	get stride(): InputSpec {
		const { size } = this
		let remaining = this.flatLength
		return [
			...size.map((s) => {
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
	): BufferReader<Element, SubtractLengths<InputSpec, SSpec>> {
		const { stride, buffer, size } = this
		return new BufferReader(
			this.read,
			buffer,
			size.slice(sSpec.length) as SubtractLengths<InputSpec, SSpec>,
			this.offset + dot(sSpec, stride)
		)
	}
	*sections() {
		// TODO: Implement
	}
	getAtIndex(index: number): any {
		// @ts-expect-error We know InputSpec but only programmatically
		return this.size.length === 1 ? this.at(index) : this.section(index)
	}
	// TODO: Implement Array<...>
}

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

const a = [].map
