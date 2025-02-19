import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		globals: true, // Enables Jest-like API (describe, it, expect)
		environment: "node", // Runs tests in a Node.js environment
		coverage: {
			provider: "v8", // Enables coverage reports
		},
	},
});
