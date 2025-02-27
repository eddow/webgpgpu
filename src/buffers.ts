import { type AnyInference, type SizeSpec, assertSize, resolvedSize } from './inference'
import { log } from './log'
import { type NumericSizesSpec, isTypedArrayXD } from './typedArrays'
import {
	InferenceValidationError,
	type InputXD,
	ParameterError,
	type TypedArray,
	type TypedArrayConstructor,
} from './types'

export interface ElementAccessor<T> {
	read(array: TypedArray, index: number): T
	write(array: TypedArray, index: number, element: T): void
}

type ValidateSizeSpec<
	Inferences extends AnyInference,
	SizesSpec,
> = SizesSpec extends SizeSpec<Inferences>[] ? unknown : never
export type ValuedBuffable<
	Inferences extends AnyInference = AnyInference,
	Buffer extends TypedArray = TypedArray,
	OriginElement = any,
	SizesSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	InputSizesSpec extends SizeSpec<Inferences>[] = [],
	InputSpec extends number[] = number[],
> = {
	buffable: Buffable<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
	value: InputXD<OriginElement, InputSpec, Buffer>
} & ValidateSizeSpec<Inferences, SizesSpec>
/**
 * Interface is needed for type inference
 */
export interface Buffable<
	Inferences extends AnyInference = any,
	Buffer extends TypedArray = TypedArray,
	OriginElement = any,
	SizesSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	InputSizesSpec extends SizeSpec<Inferences>[] = [],
	InputSpec extends number[] = number[],
> {
	readonly elementAccessor: ElementAccessor<OriginElement>
	readonly size: SizesSpec
	readonly bufferType: TypedArrayConstructor<Buffer>
	toTypedArray(
		inferences: Inferences,
		data: InputXD<OriginElement, InputSpec, Buffer>,
		reason: string,
		reasons: Record<string, string>
	): Buffer
	value(
		v: InputXD<OriginElement, InputSpec, Buffer>
	): ValuedBuffable<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
	readonly wgslSpecification: string
	readonly elementSize: number
	readonly transformSize: InputSizesSpec
	readTypedArray(
		buffer: Buffer,
		inferences: AnyInference
	): BufferReader<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec>
}

function assertElementSize(given: any, expected: number) {
	if (given !== expected)
		throw new ParameterError(`Element size mismatch: ${given} received while expecting ${expected}`)
}

// Poorly typed but for internal use only
export function elementsToTypedArray<
	Inferences extends AnyInference,
	Buffer extends TypedArray,
	OriginElement,
	SizesSpec extends SizeSpec<Inferences>[],
	InputSizesSpec extends SizeSpec<Inferences>[],
>(
	specification: Buffable<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, any>,
	inferences: Inferences,
	data: any,
	size: SizesSpec,
	reason: string,
	reasons: Record<string, string>
): Buffer {
	const {
		bufferType,
		elementSize,
		elementAccessor: { write },
		transformSize,
	} = specification
	if (data instanceof bufferType && 'elementSize' in data && data.elementSize !== elementSize)
		assertElementSize(data.elementSize, elementSize)
	// #region 0D
	if (size.length === 0) {
		if (data instanceof bufferType) {
			assertElementSize((data as Buffer).length, elementSize)
			return data as Buffer
		}
		const buffer = new bufferType(elementSize)
		write(buffer, 0, data as OriginElement)
		return buffer
	}
	// #endregion 0D
	// #region 1D
	if (size.length === 1) {
		if (data instanceof bufferType) {
			if ((data as Buffer).length % elementSize !== 0)
				throw new InferenceValidationError(
					`Size mismatch in dimension 1: ${data.length} is not a multiple of ${elementSize}`
				)
			assertSize([(data as Buffer).length / elementSize], size, inferences, reason, reasons)
			return data as Buffer
		}
		if (!Array.isArray(data))
			throw new InferenceValidationError('Input is not an array nor a typed array')
		assertSize([data.length], size, inferences, reason, reasons)
		const rv = new bufferType(data.length * elementSize) as Buffer
		let dst = 0
		// Make the `if` early not to not make it in the loop
		for (const element of data as OriginElement[]) {
			write(rv, dst, element)
			dst += elementSize
		}
		return rv
	}
	// #endregion 1D
	// #region 2~3-D

	if (data instanceof bufferType) {
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
	return rv ?? new bufferType(0)
}

function bufferPosition(index: number[], size: number[], elementSize: number): number {
	let pos = 0
	for (let i = 0; i < size.length; i++) {
		pos *= size[i]
		pos += index[i]
	}
	return pos * elementSize
}

function nextXdIndex(index: number[], size: number[]): boolean {
	for (let i = index.length - 1; i >= 0; i--) {
		index[i]++
		if (index[i] < size[i]) return true
		index[i] = 0
	}
	return false
}

export class BufferReader<
	Inferences extends AnyInference = any,
	Buffer extends TypedArray = TypedArray,
	OriginElement = any,
	SizesSpec extends SizeSpec<Inferences>[] = SizeSpec<Inferences>[],
	InputSizesSpec extends SizeSpec<Inferences>[] = [],
	InputSpec extends number[] = number[],
> {
	constructor(
		public readonly buffable: Buffable<
			Inferences,
			Buffer,
			OriginElement,
			SizesSpec,
			InputSizesSpec,
			InputSpec
		>,
		public readonly buffer: Buffer,
		public readonly size: number[]
	) {}
	at(...index: InputSpec): OriginElement {
		const {
			size,
			buffer,
			buffable: {
				elementSize,
				elementAccessor: { read },
				transformSize,
			},
		} = this
		if (index.length !== size.length)
			throw new InferenceValidationError(
				`Index length mismatch: taking ${index.length}-D index element out of ${size.length}-D buffer`
			)
		const pos = bufferPosition(index, size, elementSize)
		return read(buffer, pos)
	}
	*entries(): Generator<[InputSpec, OriginElement]> {
		const {
			size,
			buffer,
			buffable: {
				elementSize,
				elementAccessor: { read },
			},
		} = this
		if (size.some((s) => s === 0)) {
			log.warn('Zero size buffer as output')
			return
		}
		const index: number[] = size.map(() => 0)
		let pos = 0
		do {
			yield [[...index] as InputSpec, read(buffer, pos)]
			pos += elementSize
		} while (nextXdIndex(index, size))
	}
	values(): OriginElement[] {
		return Array.from(this.entries()).map(([_, v]) => v)
	}
	toArray(): OriginElement[] {
		return [...this.values()]
	}
	// TODO: type
	slice(
		...index: [number, ...number[]]
	): BufferReader<Inferences, Buffer, OriginElement, SizesSpec, InputSizesSpec, InputSpec> {
		const { size, buffer, buffable: specification } = this
		const { elementSize } = specification
		if (!this.size.length)
			throw new InferenceValidationError(`Cannot slice a ${this.size.length}-D buffer`)
		if (index.length > this.size.length)
			throw new InferenceValidationError(
				`Index length mismatch: taking ${index.length}-D index slice out of ${size.length}-D buffer`
			)
		const subSize = size.slice(index.length)
		const subLength = subSize.reduce((a, b) => a * b, 1)
		const pos = bufferPosition(index, size, elementSize) * subLength
		return new BufferReader<
			Inferences,
			Buffer,
			OriginElement,
			SizesSpec,
			InputSizesSpec,
			InputSpec
		>(specification, buffer.subarray(pos, pos + subLength) as Buffer, subSize)
	}
	*slices() {
		const { size } = this
		if (!size.length) throw new InferenceValidationError(`Cannot slice a ${size.length}-D buffer`)
		for (let i = 0; i < size[0]; i++) yield this.slice(i)
	}
}
