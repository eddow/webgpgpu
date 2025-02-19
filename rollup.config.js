import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import typescript from '@rollup/plugin-typescript'
import pluginDts from 'rollup-plugin-dts'
import { rm } from 'node:fs/promises'

// clean out the destination folder
await rm('lib', { recursive: true, force: true })
const banner = `/*
	GpGpu Compute Shader
*/`
const input = {client: 'src/client/index.ts', server: 'src/server/index.ts'}
const plugins = [
	resolve(),
	commonjs(),
	typescript({
		tsconfig: './src/tsconfig.json'
	})
]
export default  [
	{
		input,
		output: {
			banner,
			dir: 'lib'
		},
		plugins: [
			...plugins,
			pluginDts()
		]
	},
	{
		input,
		output: {
			banner,
			dir: 'lib',
			entryFileNames: '[name].mjs',
			sourcemap: true,
			format: 'esm'
		},
		plugins
	},
	{
		input,
		output: {
			banner,
			dir: 'lib',
			entryFileNames: '[name].js',
			sourcemap: true,
			format: 'cjs',
			exports: 'named'
		},
		plugins
	}
]
