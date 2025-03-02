import { expect } from 'chai'
import {
	type AnyInference,
	type AnyInput,
	type ValuedBuffable,
	f32,
	mat2x3f,
	mat3x2f,
	resolvedSize,
	vec2f,
	vec3f,
	vec4f,
} from 'webgpgpu'

describe('MappableIO', () => {
	describe('X->buffer->Y: X~Y', () => {
		function XbY<Inferences extends AnyInference, Input extends AnyInput & any[]>(
			inferences: Inferences,
			{ buffable, value }: ValuedBuffable<Inferences>
		) {
			const array = value as any[]
			// `toArrayBuffer` has to occur first in order to infer size
			const arrayBuffer = buffable.toArrayBuffer(value, inferences)
			expect(arrayBuffer.byteLength).to.equal(
				resolvedSize(buffable.size, inferences).reduce(
					// x * y * ...
					(a, c) => a * c,
					// This part takes padding into account.
					buffable.elementByteSize(inferences)
				)
			)
			const reader = buffable.readArrayBuffer(arrayBuffer, inferences)
			expect(reader).to.deepArrayEqual(array)
		}
		for (const [name, valued] of Object.entries({
			'f32<1>': f32.array('x').value([1, 2, 3, 4, 5]),
			'vec2f<1>': vec2f.array('x').value([
				[1, 2],
				[3, 4],
				[5, 6],
			]),
			'vec3f<1>': vec3f.array('x').value([
				[1, 2, 3],
				[4, 5, 6],
				[7, 8, 9],
			]),
			'vec4f<1>': vec4f.array('x').value([
				[1, 2, 3, 4],
				[5, 6, 7, 8],
				[9, 10, 11, 12],
			]),
			'mat2x3f<1>': mat2x3f.array('x').value([
				[
					[1, 2, 3],
					[4, 5, 6],
				],
				[
					[7, 8, 9],
					[10, 11, 12],
				],
				[
					[13, 14, 15],
					[16, 17, 18],
				],
			]),
			'mat3x2f<1>': mat3x2f.array('x').value([
				[
					[1, 2],
					[3, 4],
					[5, 6],
				],
				[
					[7, 8],
					[9, 10],
					[11, 12],
				],
				[
					[13, 14],
					[15, 16],
					[17, 18],
				],
			]),
			'f32<2>': f32.array('x', 'y').value([
				[1, 2, 3, 4, 5],
				[6, 7, 8, 9, 10],
				[11, 12, 13, 14, 15],
			]),
		}))
			it(name, () => {
				XbY({ x: undefined, y: undefined, z: undefined, w: undefined }, valued)
			})
	})
})

/*

			const inferences = { 'size.x': undefined, 'size.y': undefined }
			const type = f32.array('size.x', 'size.y')
			const source = [
				[1, 2, 3, 4, 5],
				[6, 7, 8, 9, 10],
			]
*/
