import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

async function sha256(filePath) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(filePath)) {
    digest.update(chunk);
  }
  return digest.digest("hex");
}

const modelsRoot = path.resolve("models");
const entries = await readdir(modelsRoot, { withFileTypes: true });
let verified = 0;
for (const entry of entries) {
  if (!entry.isDirectory()) {
    continue;
  }
  const directory = path.join(modelsRoot, entry.name);
  const metadata = JSON.parse(
    await readFile(path.join(directory, "model.json"), "utf8"),
  );
  const binaryPath = path.join(directory, metadata.binary ?? "model.bin");
  const actual = await sha256(binaryPath);
  if (actual !== metadata.sha256) {
    throw new Error(
      `${entry.name} SHA-256 mismatch: ${actual} != ${metadata.sha256}`,
    );
  }
  console.log(`${entry.name}: ${actual} OK`);
  verified += 1;
}
if (verified === 0) {
  throw new Error("No models found to verify");
}
