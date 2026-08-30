import { UserDictionary, applyDefaultDictionary } from "./dictionary.js";
import {
  normalizedSegmentationCharacter,
  segmentationFeaturesInto,
} from "./feature-extractor.js";
import { createViterbiScratch, decodeViterbiFused } from "./inference.js";

export class Segmenter {
  constructor(model, { userDictionary = [], useDefaultDictionary = true, tagger = null } = {}) {
    if (model.metadata.task !== "segmentation") {
      throw new Error(
        `Expected a segmentation model, got ${model.metadata.task}`,
      );
    }
    this.model = model;
    this.tags = model.metadata.tags;
    this.userDictionary = new UserDictionary(userDictionary);
    this.useDefaultDictionary = useDefaultDictionary;
    this.tagger = tagger;
    this._scratch = null;
  }

  // Per-segmenter reusable buffers. cut() is synchronous and single-threaded,
  // so one scratch set per segmenter cannot be reentered.
  _segScratch() {
    if (this._scratch === null) {
      const scratch = {
        nodes: [],
        offsets: new Int32Array(1024),
        viterbi: createViterbiScratch(),
      };
      scratch.fill = (node, out) =>
        segmentationFeaturesInto(this.model, scratch.nodes, node, out);
      this._scratch = scratch;
    }
    return this._scratch;
  }

  _cutFragment(text) {
    const scratch = this._segScratch();
    const nodes = scratch.nodes;
    let offsets = scratch.offsets;

    // Normalize in a single code-point pass while recording the UTF-16
    // offset of each character, so words can later be assembled by slicing
    // the original text instead of concatenating per-character strings.
    let count = 0;
    let offset = 0;
    for (const character of text) {
      if (count >= offsets.length) {
        const grown = new Int32Array(offsets.length * 2);
        grown.set(offsets);
        offsets = grown;
        scratch.offsets = grown;
      }
      nodes[count] = normalizedSegmentationCharacter(character);
      offsets[count] = offset;
      offset += character.length;
      count += 1;
    }
    if (count === 0) {
      return [];
    }
    // Drop stale entries from previous longer calls; the feature extractor
    // reads nodes.length for its context-window bounds.
    nodes.length = count;

    const states = decodeViterbiFused(this.model, count, scratch.fill, scratch.viterbi);

    const words = [];
    let wordStart = 0;
    for (let index = 1; index < count; index += 1) {
      if (this.tags[states[index]].includes("B")) {
        words.push(text.slice(offsets[wordStart], offsets[index]));
        wordStart = index;
      }
    }
    words.push(text.slice(offsets[wordStart]));
    return words;
  }

  _cutWords(text) {
    const trimmed = text.trim();
    if (trimmed.length === 0) {
      return { words: [], userTags: [] };
    }

    const output = [];
    const userTags = [];
    // Match the user dictionary against the whole text first so entries
    // containing spaces can still hit (upstream bug #109); whitespace is
    // only used to split the remaining non-dictionary fragments.
    const { parts, isWords, tags } = this.userDictionary.split(trimmed);
    for (let index = 0; index < parts.length; index += 1) {
      if (isWords[index]) {
        output.push(parts[index]);
        userTags.push(tags[index]);
        continue;
      }

      for (const piece of parts[index].split(/\s+/u)) {
        if (piece.length === 0) {
          continue;
        }

        const cutWords = this._cutFragment(piece);
        const processed = this.useDefaultDictionary
          ? applyDefaultDictionary(cutWords, this.model)
          : cutWords;
        for (const word of processed) {
          output.push(word);
          userTags.push("");
        }
      }
    }
    return { words: output, userTags };
  }

  cut(text) {
    if (typeof text !== "string") {
      throw new TypeError("Segmenter.cut expects a string");
    }
    const { words, userTags } = this._cutWords(text);
    if (this.tagger === null) {
      return words;
    }

    const tags = this.tagger.tag(words);
    for (let index = 0; index < userTags.length; index += 1) {
      if (userTags[index]) {
        tags[index] = userTags[index];
      }
    }
    return words.map((word, index) => [word, tags[index]]);
  }
}
