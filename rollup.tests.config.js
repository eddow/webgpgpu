import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import typescript from '@rollup/plugin-typescript'

import { globSync } from 'glob';

const testFiles = globSync('tests/**/*.test.ts'); // Collect all test files
/*
export default {
  input: testFiles, // Entry point for test files
  output: {
    file: 'tests/build/*.test.js',
    format: 'iife', // Needed for browser execution
    sourcemap: true,
  },
  plugins: [
    resolve(), // Allows importing dependencies
    commonjs(), // Converts CommonJS modules to ES6
    typescript({
      tsconfig: './tests/tsconfig.json'
    })
  ]
};
*/

const input = testFiles
const plugins = [
	resolve(),
	commonjs(),
	typescript({
		tsconfig: './tests/tsconfig.json'
	})
]
const external = (id) => id.includes('node_modules')

export default  [
	{
		input,
		output: {
			dir: 'tests/build'
		},
		plugins,
		external
	}
]