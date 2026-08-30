import {
  normalizePostagToken,
  postagFeaturesInto,
} from "./feature-extractor.js";
import { createViterbiScratch, decodeViterbiFused } from "./inference.js";

export class Postag {
  constructor(model) {
    if (model.metadata.task !== "postag") {
      throw new Error(`Expected a postag model, got ${model.metadata.task}`);
    }
    this.model = model;
    this.tags = model.metadata.tags;
    this._scratch = null;
  }

  // Per-tagger reusable buffers, same single-threaded rationale as Segmenter.
  _tagScratch() {
    if (this._scratch === null) {
      const scratch = { nodes: [], viterbi: createViterbiScratch() };
      scratch.fill = (node, out) =>
        postagFeaturesInto(this.model, scratch.nodes, node, out);
      this._scratch = scratch;
    }
    return this._scratch;
  }

  tag(words) {
    if (words.length === 0) {
      return [];
    }
    const scratch = this._tagScratch();
    const nodes = scratch.nodes;
    for (let index = 0; index < words.length; index += 1) {
      nodes[index] = normalizePostagToken(words[index]);
    }
    // See Segmenter: keep nodes.length at the exact word count.
    nodes.length = words.length;

    const states = decodeViterbiFused(
      this.model,
      words.length,
      scratch.fill,
      scratch.viterbi,
    );
    const tags = new Array(words.length);
    for (let index = 0; index < words.length; index += 1) {
      tags[index] = this.tags[states[index]];
    }
    return tags;
  }
}
