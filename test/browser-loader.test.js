import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { after, before, test } from "node:test";

import { loadBrowserSegmenter } from "../src/browser.js";

const modelDirectory = path.resolve("models/default");
const modelJsonUrl = "https://pkuseg.test/models/default/model.json";
let originalFetch;

before(async () => {
  originalFetch = globalThis.fetch;
  const [metadata, binary] = await Promise.all([
    readFile(path.join(modelDirectory, "model.json")),
    readFile(path.join(modelDirectory, "model.bin")),
  ]);
  globalThis.fetch = async (url) => {
    const value = String(url);
    if (value.endsWith("model.json")) {
      return new Response(metadata, {
        headers: { "content-type": "application/json" },
      });
    }
    if (value.endsWith("model.bin")) {
      return new Response(binary, {
        headers: { "content-type": "application/octet-stream" },
      });
    }
    return new Response("not found", { status: 404 });
  };
});

after(() => {
  globalThis.fetch = originalFetch;
});

test("browser entry loads model metadata and binary through fetch", async () => {
  const segmenter = await loadBrowserSegmenter({ modelJsonUrl });
  assert.deepEqual(segmenter.cut("我爱北京天安门"), [
    "我",
    "爱",
    "北京",
    "天安门",
  ]);
});
