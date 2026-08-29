import {
  normalizePostagToken,
  postagFeaturesAt,
} from "./feature-extractor.js";
import { decodeViterbi } from "./inference.js";

export class Postag {
  constructor(model) {
    if (model.metadata.task !== "postag") {
      throw new Error(`Expected a postag model, got ${model.metadata.task}`);
    }
    this.model = model;
    this.tags = model.metadata.tags;
  }

  tag(words) {
    if (words.length === 0) {
      return [];
    }
    const normalized = words.map(normalizePostagToken);
    const featureLists = normalized.map((_, index) =>
      postagFeaturesAt(this.model, normalized, index),
    );
    return decodeViterbi(featureLists, this.model).map(
      (identifier) => this.tags[identifier],
    );
  }
}
