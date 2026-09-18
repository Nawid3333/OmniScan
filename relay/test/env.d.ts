declare module "cloudflare:test" {
  interface ProvidedEnv {
    HUB: DurableObjectNamespace;
    HOOK_SECRET: string;
    CLIENT_TOKEN: string;
  }
}
