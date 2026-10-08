/**
 * DeepSeek: a metered API engine on the studio's own pi-ai loop, its key kept in the secret store.
 *
 * Runs pi-ai against a real local HTTP server (`helpers/fake-deepseek.ts`), so the provider wiring,
 * the key in the Authorization header, streaming, tool calls, usage and failure mapping are
 * exercised for real, with no network and no account.
 */
import assert from "node:assert/strict";
import path from "node:path";
import { after, describe, it } from "node:test";
import { DeepSeekEngine, cleanDeepSeekKey, deepSeekModels } from "../../src/substrate/engines/deepseek.ts";
import { EngineRegistry } from "../../src/substrate/engines/registry.ts";
import { EngineError } from "../../src/substrate/engines/types.ts";
import { memoryKeyStore, type ApiKeyStore } from "../../src/substrate/provider-keys.ts";
import { SecretStorageUnavailableError } from "../../src/substrate/secrets.ts";
import { SecretStorageIssue } from "../../src/shared/secret-storage.ts";
import { CATALOG, GOOD_KEY, startFakeDeepSeek, type FakeDeepSeek } from "../helpers/fake-deepseek.ts";
import type { FakeReply } from "../helpers/fake-ollama.ts";
import { tmpDir } from "../helpers/tmp.ts";

const servers: FakeDeepSeek[] = [];
after(async () => {
  await Promise.all(servers.map((server) => server.close()));
});

async function fake(options?: Parameters<typeof startFakeDeepSeek>[0]): Promise<FakeDeepSeek> {
  const server = await startFakeDeepSeek(options);
  servers.push(server);
  return server;
}

async function engineFor(server: FakeDeepSeek, keys: ApiKeyStore = memoryKeyStore(GOOD_KEY)) {
  const root = await tmpDir("deepseek-");
  return new DeepSeekEngine({ root: path.join(root, "engine"), keys, baseUrl: server.baseUrl });
}

const CHAT = "deepseek-chat";
const REASONER = "deepseek-reasoner";

describe("DeepSeek's catalog", () => {
  it("reads DeepSeek's ids and gives each the engine's own capabilities", () => {
    const rows = deepSeekModels({ object: "list", data: CATALOG });
    assert.deepEqual(
      rows.map((row) => row.id),
      [CHAT, REASONER],
    );
    const [chat, reasoner] = rows;
    assert.equal(chat?.label, "DeepSeek Chat");
    assert.equal(chat?.supportsTools, true);
    assert.equal(chat?.supportsThinking, false);
    assert.equal(chat?.efforts, undefined, "a model that does not think has no dial");
    assert.equal(reasoner?.label, "DeepSeek Reasoner");
    assert.equal(reasoner?.supportsThinking, true);
    assert.deepEqual(reasoner?.efforts, ["low", "medium", "high"]);
  });

  it("refuses a catalog that is not one, and skips entries it cannot read", () => {
    for (const body of [null, {}, { data: "x" }, "[]"])
      assert.throws(() => deepSeekModels(body), /could not be read/, JSON.stringify(body));
    const rows = deepSeekModels({ data: [null, 7, { id: 3 }, { id: CHAT }] });
    assert.deepEqual(
      rows.map((row) => row.id),
      [CHAT],
    );
  });

  it("keeps nothing stale as fresh when the read fails", async () => {
    const broken = await fake({ catalog: { status: 200, body: "{not json" } });
    const engine = await engineFor(broken);
    await engine.refreshModels(true);
    assert.deepEqual(await engine.models(), []);
    assert.equal(engine.catalogSnapshot().state, "unavailable");
    assert.equal(engine.catalogSnapshot().problem?.code, "malformed");
  });
});

describe("DeepSeek's key", () => {
  it("says what to do at each step: no key, a refused key, a good key, a locked store", async () => {
    const server = await fake();
    const none = await engineFor(server, memoryKeyStore(null));
    const missing = await none.status();
    assert.equal(missing.code, "needs_login");
    assert.match(missing.remedy ?? "", /Settings › Model Providers/);

    const refused = await (await engineFor(server, memoryKeyStore("sk-not-a-key-this-server-takes"))).status();
    assert.equal(refused.code, "needs_login");
    assert.match(refused.detail, /did not accept/);

    assert.equal((await (await engineFor(server)).status()).code, "ready");

    const locked: ApiKeyStore = {
      read: async () => {
        throw new SecretStorageUnavailableError(SecretStorageIssue.NoKeyring);
      },
      write: async () => {},
      clear: async () => {},
    };
    const lockedStatus = await (await engineFor(server, locked)).status();
    assert.equal(lockedStatus.code, "error");
    assert.match(lockedStatus.detail, /secret store is locked/);
  });

  it("saves a pasted key only when DeepSeek takes it, and forgets it on request", async () => {
    const server = await fake();
    const keys = memoryKeyStore(null);
    const engine = await engineFor(server, keys);
    assert.equal((await engine.saveKey("sk-wrong-key-0000000000000000")).code, "needs_login");
    assert.equal(await keys.read(), null, "a refused key is not kept");
    assert.equal((await engine.saveKey(`  ${GOOD_KEY}\n`)).code, "ready");
    assert.equal(await keys.read(), GOOD_KEY, "kept trimmed");
    assert.equal((await engine.clearKey()).code, "needs_login");
    assert.equal(await keys.read(), null);
  });

  it("refuses what cannot be a key before asking anyone, and stores nothing (hostile input)", async () => {
    const server = await fake();
    const keys = memoryKeyStore(null);
    const engine = await engineFor(server, keys);
    const hostile: unknown[] = ["", "   ", "short", `${GOOD_KEY}\nX-Injected: 1`, `${GOOD_KEY} ${GOOD_KEY}`, 42, null];
    for (const value of hostile) {
      assert.equal(cleanDeepSeekKey(value), null, JSON.stringify(value)?.slice(0, 40));
      const status = await engine.saveKey(value);
      assert.equal(status.code, "needs_login");
      assert.doesNotMatch(JSON.stringify(status), /sk-0123/, "the answer never carries a key");
    }
    assert.equal(await keys.read(), null);
    assert.equal(server.requests.length, 0, "DeepSeek was never asked about a value that cannot be a key");
  });
});

describe("DeepSeek completions", () => {
  it("streams a reply with the key as a bearer and the model's own id", async () => {
    const server = await fake({ replies: [{ text: "Hello from DeepSeek" }] });
    const engine = await engineFor(server);
    const deltas: string[] = [];
    const response = await engine.complete({
      model: CHAT,
      messages: [{ role: "user", content: "hi" }],
      onDelta: (delta) => deltas.push(delta),
    });
    assert.equal(response.message.content, "Hello from DeepSeek");
    assert.equal(deltas.join(""), "Hello from DeepSeek");
    assert.equal(response.engine, "deepseek");
    assert.equal(response.usage.engine, "deepseek");
    const call = server.requests.find((request) => request.path.startsWith("/chat/completions"));
    assert.equal(call?.authorization, `Bearer ${GOOD_KEY}`);
    assert.equal((call?.body as { model?: string } | undefined)?.model, CHAT);
  });

  it("runs tool calls and maps a throttled or refused request by its status", async () => {
    const server = await fake({
      replies: [{ toolCalls: [{ id: "call-1", name: "probe", arguments: { depth: 2 } }] }],
    });
    const engine = await engineFor(server);
    const tools = [{ name: "probe", description: "probe", parameters: { type: "object", properties: {} } }];
    const response = await engine.complete({ model: CHAT, messages: [{ role: "user", content: "go" }], tools });
    assert.deepEqual(response.message.tool_calls, [{ id: "call-1", name: "probe", arguments: { depth: 2 } }]);

    const rows: Array<[FakeReply, string]> = [
      [{ httpStatus: 429, body: '{"error":{"message":"Rate limit exceeded"}}' }, "rate_limit"],
      [{ httpStatus: 401, body: '{"error":{"message":"bad key"}}' }, "auth"],
      [{ httpStatus: 502, body: '{"error":{"message":"upstream"}}' }, "unavailable"],
      [{ cutAfter: "half a rep" }, "unavailable"],
    ];
    for (const [reply, kind] of rows) {
      const failing = await engineFor(await fake({ replies: [reply] }));
      await assert.rejects(
        failing.complete({ model: CHAT, messages: [{ role: "user", content: "x" }] }),
        (err) => err instanceof EngineError && err.kind === kind && err.engine === "deepseek",
        JSON.stringify(reply),
      );
    }
  });

  it("says the key is missing as a sign-in failure, without sending anything", async () => {
    const server = await fake();
    const engine = await engineFor(server, memoryKeyStore(null));
    await engine.refreshModels(true);
    await assert.rejects(
      engine.complete({ model: CHAT, messages: [{ role: "user", content: "x" }] }),
      (err) => err instanceof EngineError && err.kind === "auth",
    );
    assert.equal(server.requests.filter((request) => request.path.includes("chat/completions")).length, 0);
  });
});

describe("DeepSeek in the registry", () => {
  it("is described as a session engine that needs its key, and is never a fallback", async () => {
    const server = await fake();
    const registry = new EngineRegistry();
    registry.register(await engineFor(server, memoryKeyStore(null)));
    const [described] = await registry.describe();
    assert.equal(described?.id, "deepseek");
    assert.equal(described?.kind, "direct");
    assert.equal(described?.supportsSessions, true);
    assert.equal(described?.status.code, "needs_login");
    assert.equal(described?.provider?.billing, "metered");
    assert.equal(described?.defaultModel, null, "no model is picked for the person");

    const ready = new EngineRegistry();
    ready.register(await engineFor(server));
    assert.deepEqual(await ready.fallbackFor("claude-code", { kind: "rate_limit" }), []);
    assert.equal(await ready.firstReady(), null);
  });
});
