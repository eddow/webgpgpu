import { expect } from 'chai'
import { f32, mat2x2f, mat3x3f, Struct, u32, vec2f, vec3f, vec4f } from 'webgpgpu'

describe('struct', () => {
	describe('bytesPerAtomic', () => {
		it('single f32 pads to 16', () => {
			expect(new Struct('S', { a: f32 }).bytesPerAtomic).to.equal(16)
		})
		it('two f32 fields occupy 32 bytes', () => {
			expect(new Struct('S', { a: f32, b: f32 }).bytesPerAtomic).to.equal(32)
		})
		it('vec3f occupies 16 bytes', () => {
			expect(new Struct('S', { v: vec3f }).bytesPerAtomic).to.equal(16)
		})
		it('vec2f + f32 sorted largest-first', () => {
			// vec2f (8) sorted before f32 (4); each padded to 16
			expect(new Struct('S', { s: f32, v: vec2f }).bytesPerAtomic).to.equal(32)
		})
		it('empty struct', () => {
			expect(new Struct('S', {}).bytesPerAtomic).to.equal(0)
		})
	})

	describe('round-trip', () => {
		function roundTrip<V extends Record<string, any>>(
			s: Struct<any, any>,
			value: V
		): Record<string, any> {
			const buf = s.toArrayBuffer(value, {})
			return s.readArrayBuffer(buf, {}).at()
		}

		it('single scalar', () => {
			const s = new Struct('S', { a: f32 })
			const { a } = roundTrip(s, { a: 42.5 })
			expect(a).to.equal(42.5)
		})

		it('two scalars', () => {
			const s = new Struct('S', { x: f32, y: f32 })
			const { x, y } = roundTrip(s, { x: 1.5, y: 2.5 })
			expect(x).to.equal(1.5)
			expect(y).to.equal(2.5)
		})

		it('unsigned integer', () => {
			const s = new Struct('S', { n: u32 })
			const { n } = roundTrip(s, { n: 7 })
			expect(n).to.equal(7)
		})

		it('vec2f field', () => {
			const s = new Struct('S', { v: vec2f })
			const { v } = roundTrip(s, { v: [3, 4] })
			expect(Array.from(v)).to.deep.equal([3, 4])
		})

		it('vec3f field (padding)', () => {
			const s = new Struct('S', { v: vec3f })
			const { v } = roundTrip(s, { v: [1, 2, 3] })
			expect(Array.from(v)).to.deep.equal([1, 2, 3])
		})

		it('vec4f field', () => {
			const s = new Struct('S', { v: vec4f })
			const { v } = roundTrip(s, { v: [5, 6, 7, 8] })
			expect(Array.from(v)).to.deep.equal([5, 6, 7, 8])
		})

		it('mixed scalar + vector', () => {
			const s = new Struct('S', { a: f32, v: vec3f })
			const r = roundTrip(s, { a: 10, v: [1, 2, 3] })
			expect(r.a).to.equal(10)
			expect(Array.from(r.v)).to.deep.equal([1, 2, 3])
		})

		it('multiple vectors', () => {
			const s = new Struct('S', { p: vec3f, q: vec2f })
			const r = roundTrip(s, { p: [10, 20, 30], q: [40, 50] })
			expect(Array.from(r.p)).to.deep.equal([10, 20, 30])
			expect(Array.from(r.q)).to.deep.equal([40, 50])
		})

		it('mat2x2f field', () => {
			const s = new Struct('S', { m: mat2x2f })
			const val = [
				[1, 2],
				[3, 4],
			]
			const { m } = roundTrip(s, { m: val })
			expect(Array.from(m[0])).to.deep.equal([1, 2])
			expect(Array.from(m[1])).to.deep.equal([3, 4])
		})

		it('mat3x3f + scalar (non-power-of-2 stride)', () => {
			const s = new Struct('S', { a: f32, m: mat3x3f })
			const mat = [
				[1, 2, 3],
				[4, 5, 6],
				[7, 8, 9],
			]
			const r = roundTrip(s, { a: 99, m: mat })
			expect(r.a).to.equal(99)
			for (let c = 0; c < 3; c++) expect(Array.from(r.m[c])).to.deep.equal(mat[c])
		})
	})

	describe('wgsl', () => {
		it('generates valid struct syntax', () => {
			const s = new Struct('Params', { a: f32, v: vec3f })
			const wgsl = s.wgsl
			expect(wgsl).to.include('struct Params {')
			expect(wgsl).to.not.include('var ')
			expect(wgsl).to.not.include('@offset')
			expect(wgsl).to.include('v: vec3f,')
			expect(wgsl).to.include('a: f32,')
		})
	})
})
