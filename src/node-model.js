import { readFile } from "node:fs/promises";
import path from "node:path";
import { Model } from "./model.js";

export async function loadModel(modelDirectory) {
  const directory = path.resolve(modelDirectory);
  const metadataPath = path.join(directory, "model.json");
  const metadata = JSON.parse(await readFile(metadataPath, "utf8"));
  const binaryPath = path.join(directory, metadata.binary ?? "model.bin");
  const bytes = await readFile(binaryPath);
  return Model.fromBytes(bytes, metadata);
}
