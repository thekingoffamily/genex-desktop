/**
 * One-command development launcher: finds a Node 24, builds, then starts Electron.
 *
 * `npm run dev` runs this with whatever Node npm itself uses, so a machine whose default Node is
 * older still starts the app: the launcher locates a Node 24 (this process, `GENEX_NODE`, a system
 * install, or a portable copy under `~/node24/`) and re-runs the build and Electron with it. The
 * portable path matches the one the Windows setup uses; nothing is installed or changed.
 */
import { spawnSync } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function exists(file) {
  try {
    return statSync(file).isFile();
  } catch {
    return false;
  }
}

/** `node.exe` files one level inside `dir` (a portable zip extracts into its own folder). */
function portableNodes(dir) {
  try {
    return readdirSync(dir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(dir, entry.name, "node.exe"));
  } catch {
    return [];
  }
}

/** A Node 24 executable, or null: this process, an override, a system install or a portable copy. */
function node24() {
  if (Number(process.versions.node.split(".")[0]) >= 24) return process.execPath;
  const candidates = [
    process.env.GENEX_NODE,
    path.join(process.env.ProgramFiles ?? "", "nodejs", "node.exe"),
    ...portableNodes(path.join(os.homedir(), "node24")),
  ];
  for (const file of candidates) {
    if (!file || !exists(file)) continue;
    const probe = spawnSync(file, ["-p", "process.versions.node"], { encoding: "utf8" });
    if (probe.status === 0 && Number(String(probe.stdout).trim().split(".")[0]) >= 24) return file;
  }
  return null;
}

const node = node24();
if (!node) {
  console.error("Genex needs Node 24. Install it (nvm use, or https://nodejs.org), or set GENEX_NODE to its node.exe.");
  process.exit(1);
}

const env = { ...process.env, PATH: `${path.dirname(node)}${path.delimiter}${process.env.PATH ?? ""}` };
if (process.argv.includes("--unsandboxed")) {
  // srt-win's ACL stamp can hang on Windows; this runs the app without the process sandbox.
  env.GENEX_UNSANDBOXED = "1";
  console.log("Genex dev: running without the process sandbox (--unsandboxed).");
}

function run(args) {
  const result = spawnSync(node, args, { cwd: ROOT, stdio: "inherit", env });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

run([path.join(ROOT, "scripts", "build.mjs")]);
run([path.join(ROOT, "node_modules", "electron", "cli.js"), "."]);
