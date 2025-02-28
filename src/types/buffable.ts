import { type AnyInference, type SizeSpec, assertSize, resolvedSize } from '../inference'
import { log } from '../log'
import { type NumericSizesSpec, isTypedArrayXD } from '../typedArrays'
import { InferenceValidationError, type InputXD, ParameterError } from '../types'

type ValidateSizeSpec<
	Inferences extends AnyInference,
	SizesSpec,
> = SizesSpec extends SizeSpec<Inferences>[] ? unknown : never
export type ValuedBuffable<
	Inferences extends AnyInference = AnyInference,
	Element = any,
	SizesSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	ElementSizeSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
> = {
	buffable: Buffable<Inferences, Element, SizesSpec, ElementSizeSpec>
	value: InputXD<Element, SizesSpec>
} // & ValidateSizeSpec<Inferences, SizesSpec>

/**
 * Interface is needed for type inference
 */
export interface Buffable<
	Inferences extends AnyInference = any,
	Element = any,
	SizesSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	ElementSizeSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
> {
	readonly size: SizesSpec
	toArrayBuffer(
		data: InputXD<Element, SizesSpec>,
		inferences: Inferences,
		reason: string,
		reasons: Record<string, string>
	): ArrayBuffer
	value(
		v: InputXD<Element, SizesSpec>
	): ValuedBuffable<Inferences, Element, SizesSpec, ElementSizeSpec>
	readonly wgslSpecification: string
	readArrayBuffer(
		buffer: ArrayBuffer,
		inferences: AnyInference
	): BufferReader<Element, NumericSizesSpec<SizesSpec>>
	elementByteSize(inferences: Inferences): number
}

function assertElementSize(given: any, expected: number) {
	if (given !== expected)
		throw new ParameterError(`Element size mismatch: ${given} received while expecting ${expected}`)
}
export interface Writer<Element> {
	write(index: number, value: Element): void
	writeMany?(index: number, values: Element[]): void
}

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

function bufferPosition(index: number[], size: number[]): number {
	let pos = 0
	for (let i = 0; i < size.length; i++) {
		if (index[i] >= size[i]) throw new ParameterError(`Index out of range (${index[i]}/${size[i]})`)
		pos *= size[i]
		pos += index[i]
	}
	return pos
}

function nextXdIndex(index: number[], size: number[]): boolean {
	for (let i = index.length - 1; i >= 0; i--) {
		index[i]++
		if (index[i] < size[i]) return true
		index[i] = 0
	}
	return false
}

export class BufferReader<Element = any, InputSpec extends number[] = number[]> {
	constructor(
		private readonly read: (index: number) => Element,
		public readonly size: number[]
	) {}
	at(...index: InputSpec): Element {
		const { size } = this
		if (index.length !== size.length)
			throw new InferenceValidationError(
				`Index length mismatch: taking ${index.length}-D index element out of ${size.length}-D buffer`
			)
		return this.read(bufferPosition(index, size))
	}
	/**
	 * Retrieves the [index, value] pairs, where `index` is the index array and `value` is the value at that index
	 * Index is *not* cloned, so its content will be modified along browsing and should not be stored as-is
	 * @returns Generator<[InputSpec, Element]>
	 */
	*entries(): Generator<[InputSpec, Element]> {
		const { size } = this
		if (size.some((s) => s === 0)) {
			log.warn('Zero size buffer as output')
			return
		}
		const index: number[] = size.map(() => 0)
		let pos = 0
		do yield [index as InputSpec, this.read(pos++)]
		while (nextXdIndex(index, size))
	}
	*values() {
		for (const [_, v] of this.entries()) yield v
	}
	toArray(): Element[] {
		return [...this.values()]
	}
}
