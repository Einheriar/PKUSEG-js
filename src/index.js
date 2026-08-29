import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { loadModel } from "./node-model.js";
import { Postag } from "./postag.js";
import { Segmenter } from "./segmenter.js";

const DEFAULT_MODEL_DIRECTORY = fileURLToPath(
  new URL("../models/default/", import.meta.url),
);
const DEFAULT_POSTAG_DIRECTORY = fileURLToPath(
  new URL("../models/postag/", import.meta.url),
);

async function loadUserDictionary(value) {
  if (value === "default" || value === null || Array.isArray(value)) {
    return value;
  }
  if (typeof value !== "string") {
    throw new TypeError(
      "userDictionary must be 'default', null, an array, or a file path",
    );
  }

  const contents = await readFile(value, "utf8");
  return contents
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const fields = line.split("\t");
      return fields.length > 1 ? [fields[0], fields[1]] : fields[0];
    });
}

export async function loadSegmenter({
  modelDirectory = DEFAULT_MODEL_DIRECTORY,
  userDictionary = "default",
  postag = false,
  postagModelDirectory = DEFAULT_POSTAG_DIRECTORY,
} = {}) {
  const model = await loadModel(modelDirectory);
  const loadedDictionary = await loadUserDictionary(userDictionary);
  const useDefaultDictionary = loadedDictionary !== null;
  const entries = Array.isArray(loadedDictionary) ? loadedDictionary : [];
  const tagger = postag
    ? new Postag(await loadModel(postagModelDirectory))
    : null;

  return new Segmenter(model, {
    userDictionary: entries,
    useDefaultDictionary,
    tagger,
  });
}

export { Model } from "./model.js";
export { loadModel } from "./node-model.js";
export { Postag } from "./postag.js";
export { Segmenter } from "./segmenter.js";
