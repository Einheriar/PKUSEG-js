import { Model } from "./model.js";
import { Postag } from "./postag.js";
import { Segmenter } from "./segmenter.js";

export async function loadModelFromUrl(modelJsonUrl) {
  const metadataUrl = new URL(modelJsonUrl, globalThis.location?.href);
  const metadataResponse = await fetch(metadataUrl);
  if (!metadataResponse.ok) {
    throw new Error(
      `Unable to load PKUSEG metadata: HTTP ${metadataResponse.status}`,
    );
  }
  const metadata = await metadataResponse.json();
  const binaryUrl = new URL(metadata.binary ?? "model.bin", metadataUrl);
  const binaryResponse = await fetch(binaryUrl);
  if (!binaryResponse.ok) {
    throw new Error(
      `Unable to load PKUSEG model: HTTP ${binaryResponse.status}`,
    );
  }
  return Model.fromBytes(await binaryResponse.arrayBuffer(), metadata);
}

export async function loadBrowserSegmenter({
  modelJsonUrl,
  userDictionary = "default",
  postagModelJsonUrl = null,
} = {}) {
  if (modelJsonUrl === undefined) {
    throw new TypeError("modelJsonUrl is required in browser environments");
  }
  if (
    userDictionary !== "default" &&
    userDictionary !== null &&
    !Array.isArray(userDictionary)
  ) {
    throw new TypeError(
      "Browser userDictionary must be 'default', null, or an array",
    );
  }

  const [model, postagModel] = await Promise.all([
    loadModelFromUrl(modelJsonUrl),
    postagModelJsonUrl === null
      ? Promise.resolve(null)
      : loadModelFromUrl(postagModelJsonUrl),
  ]);
  return new Segmenter(model, {
    userDictionary: Array.isArray(userDictionary) ? userDictionary : [],
    useDefaultDictionary: userDictionary !== null,
    tagger: postagModel === null ? null : new Postag(postagModel),
  });
}

export { Model } from "./model.js";
export { Postag } from "./postag.js";
export { Segmenter } from "./segmenter.js";
