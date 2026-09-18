import { defineWorkersConfig } from "@cloudflare/vitest-pool-workers/config";

export default defineWorkersConfig({
  test: {
    globals: true,
    poolOptions: {
      workers: {
        wrangler: { configPath: "./wrangler.jsonc" },
        isolatedStorage: false,
        miniflare: {
          bindings: {
            HOOK_SECRET: "test-hook-secret",
            CLIENT_TOKEN: "test-client-token",
          },
        },
      },
    },
  },
});
