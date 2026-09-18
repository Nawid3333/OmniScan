import { EventHub } from "./hub.js";
import type { Env } from "./types.js";

export { EventHub } from "./hub.js";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    if (path === "/health") {
      const id = env.HUB.idFromName("default");
      const stub = env.HUB.get(id);
      return stub.fetch(request);
    }

    if (path.startsWith("/hooks/extractpics/")) {
      const id = env.HUB.idFromName("default");
      const stub = env.HUB.get(id);
      return stub.fetch(request);
    }

    if (!isAuthorized(request, env)) {
      return new Response("Unauthorized", { status: 401 });
    }

    if (path === "/events" || path === "/ack" || path === "/ws") {
      const id = env.HUB.idFromName("default");
      const stub = env.HUB.get(id);
      return stub.fetch(request);
    }

    return new Response("Not found", { status: 404 });
  },
} satisfies ExportedHandler<Env>;

function isAuthorized(request: Request, env: Env): boolean {
  const auth = request.headers.get("Authorization") ?? "";
  const prefix = "Bearer ";
  if (!auth.startsWith(prefix)) {
    return false;
  }
  const token = auth.slice(prefix.length);
  return constantTimeEqual(token, env.CLIENT_TOKEN);
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }
  let result = 0;
  for (let i = 0; i < a.length; i += 1) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}
