const {defineConfig} = require("@playwright/test");

module.exports = defineConfig({
  testDir: __dirname,
  testMatch: "**/*.spec.js",
  timeout: 45_000,
  expect: {timeout: 10_000},
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  use: {
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
});
