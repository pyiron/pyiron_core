/** @type {import('jest').Config} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  testMatch: ['**/?(*.)+(spec|test).[tj]s?(x)'],
  testPathIgnorePatterns: ['/__tests__/.*\\.js$', '/node_modules/'],
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov'],
  transform: {
    '^.+\\.(t|j)sx?$': ['ts-jest', { tsconfig: 'tsconfig.json' }],
  },
  setupFilesAfterEnv: ['<rootDir>/setupTests.ts'],
  
  // ✅ Mock ALL CSS, including node_modules packages like @xyflow/react
  moduleNameMapper: {
    '.+\\.(css|less|scss)$': '<rootDir>/__mocks__/styleMock.js',
  },

  // ✅ Ensure node_modules CSS is not attempted to be transformed
  transformIgnorePatterns: [
    'node_modules/(?!.*\\.(css|less|scss)$)',
  ],
};