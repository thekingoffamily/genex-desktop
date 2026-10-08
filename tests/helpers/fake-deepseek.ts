/**
 * A real HTTP server speaking the part of DeepSeek's API the engine uses: the key-gated model list
 * and the OpenAI-compatible `chat/completions` SSE stream. pi-ai does real HTTP and SSE against it,
 * so the engine's provider wiring, key handling and failure mapping are exercised for real, with no
 * network and no account.
 */
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { type FakeReply, writeChatCompletion } from "./fake-ollama.ts";

/** A key this server accepts. */
export const GOOD_KEY = "sk-0123456789abcdef0123456789abcdef";

export interface FakeDeepSeekOptions {
  /** The catalog's `data`. */
  models?: unknown[];
  /** Replace the whole `/models` answer: a status and a raw body. */
  catalog?: { status: number; body: string };
  /** Keys `/models` and `/chat/completions` accept; any other is refused with 401. */
  keys?: string[];
  replies?: FakeReply[];
}

export interface FakeDeepSeek {
  baseUrl: string;
  requests: Array<{ path: string; body: unknown; authorization: string | undefined }>;
  close: () => Promise<void>;
}

/** DeepSeek lists exactly its two text models; the metadata is the engine's own. */
export const CATALOG = [
  { id: "deepseek-chat", object: "model", owned_by: "deepseek" },
  { id: "deepseek-reasoner", object: "model", owned_by: "deepseek" },
];

export async function startFakeDeepSeek(options: FakeDeepSeekOptions = {}): Promise<FakeDeepSeek> {
  const keys = options.keys ?? [GOOD_KEY];
  const replies = [...(options.replies ?? [])];
  const requests: FakeDeepSeek["requests"] = [];
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      let body: unknown = null;
      try {
        body = raw ? JSON.parse(raw) : null;
      } catch {
        body = raw;
      }
      const url = req.url ?? "/";
      requests.push({ path: url, body, authorization: req.headers.authorization });
      const json = (value: unknown, status = 200) => {
        res.writeHead(status, { "content-type": "application/json" });
        res.end(JSON.stringify(value));
      };
      const keyGiven = req.headers.authorization?.replace(/^Bearer /, "") ?? "";
      const accepted = keys.includes(keyGiven);
      if (url.startsWith("/models")) {
        if (options.catalog) {
          res.writeHead(options.catalog.status, { "content-type": "application/json" });
          return res.end(options.catalog.body);
        }
        if (!accepted) return json({ error: { message: "Authentication Fails", code: 401 } }, 401);
        return json({ object: "list", data: options.models ?? CATALOG });
      }
      if (url.startsWith("/chat/completions")) {
        if (!accepted) return json({ error: { message: "Authentication Fails", code: 401 } }, 401);
        return writeChatCompletion(res, body, replies.shift() ?? { text: "ok" });
      }
      json({ error: `unhandled ${url}` }, 404);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    requests,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections?.();
        server.close(() => resolve());
      }),
  };
}
