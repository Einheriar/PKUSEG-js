import { UserDictionary, applyDefaultDictionary } from "./dictionary.js";
import {
  normalizeSegmentationText,
  segmentationFeaturesAt,
} from "./feature-extractor.js";
import { decodeViterbi } from "./inference.js";

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
  }

  _cutFragment(text) {
    const originalCharacters = Array.from(text);
    if (originalCharacters.length === 0) {
      return [];
    }
    const normalized = normalizeSegmentationText(text);
    const featureLists = normalized.map((_, index) =>
      segmentationFeaturesAt(this.model, normalized, index),
    );
    const states = decodeViterbi(featureLists, this.model);

    const words = [];
    let currentWord = originalCharacters[0];
    for (let index = 1; index < originalCharacters.length; index += 1) {
      if (this.tags[states[index]].includes("B")) {
        words.push(currentWord);
        currentWord = originalCharacters[index];
      } else {
        currentWord += originalCharacters[index];
      }
    }
    words.push(currentWord);
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
        output.push(...processed);
        userTags.push(...processed.map(() => ""));
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
