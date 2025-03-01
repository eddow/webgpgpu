import commonjs from '@rollup/plugin-commonjs'
import resolve from '@rollup/plugin-node-resolve'
import typescript from '@rollup/plugin-typescript'

export default {
	output: {
		format: 'iife',
		sourcemap: true,
		name: 'webgpgpuTest',
		globals: {
			chai: 'chai',
		},
	},
	external: ['chai'],
	plugins: [
		resolve(),
		commonjs({
			include: 'node_modules/**',
			ignore: ['js-base64'],
			sourceMap: false,
		}),
		typescript({
			tsconfig: './tests/tsconfig.json',
			paths: {
				webgpgpu: ['../src/client/index.ts'],
			},
		}),
	],
}
