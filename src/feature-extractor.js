import { PairHasher } from "./hash.js";

const KEYWORDS = new Set(Array.from("-._,|/*:"));
const NUMBERS = new Set(
  Array.from("0123456789.几二三四五六七八九十千万亿兆零１２３４５６７８９０％"),
);
const LETTERS = new Set(
  Array.from(
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ" +
      "ａｂｃｄｅｆｇｈｉｇｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ" +
      "／・－",
  ),
);

// Cache-key strides must exceed every interned id they combine with.
// Distinct normalized characters are bounded by Unicode itself, and the word
// interning caps below stay far under these strides, so composite numeric
// keys cannot collide.
const ID_STRIDE = 2 ** 21;
const WORD_CAP = 500_000;
const WW_CAP = 500_000;
const NORM_TOKEN_CAP = 500_000;

function renameKeyword(character) {
  return KEYWORDS.has(character) ? "&" : character;
}

function computeNormalizedSegmentationCharacter(character) {
  const renamed = renameKeyword(character);
  if (NUMBERS.has(renamed)) {
    return "**Num";
  }
  if (LETTERS.has(renamed)) {
    return "**Letter";
  }
  return renamed;
}

// Character normalization depends only on the static sets above and the
// distinct-character space is tiny, so one shared unbounded map is safe.
const normalizedCharCache = new Map();

export function normalizedSegmentationCharacter(character) {
  let normalized = normalizedCharCache.get(character);
  if (normalized === undefined) {
    normalized = computeNormalizedSegmentationCharacter(character);
    normalizedCharCache.set(character, normalized);
  }
  return normalized;
}

export function normalizeSegmentationText(text) {
  return Array.from(text, normalizedSegmentationCharacter);
}

function pushKnownFeature(model, output, feature) {
  const identifier = model.lookupFeature(feature);
  if (identifier !== -1) {
    output.push(identifier);
  }
}

function pushId(output, identifier) {
  if (identifier !== -1) {
    output.push(identifier);
  }
}

function internSegNode(model, node) {
  const cache = model.optCache;
  let id = cache.nodeIds.get(node);
  if (id === undefined) {
    id = cache.nodeStrs.length;
    cache.nodeIds.set(node, id);
    cache.nodeStrs.push(node);
  }
  return id;
}

// Single-character context features (c., c-1., c1., c-2., c2.) repeat for
// every occurrence of the same character, so they are memoized per model.
function feat1(model, kind, prefix, node, nodeId) {
  if (nodeId >= ID_STRIDE) {
    return model.lookupFeature(prefix + node);
  }
  const cache = model.optCache;
  const key = kind * ID_STRIDE + nodeId;
  let id = cache.feat1.get(key);
  if (id === undefined) {
    id = model.lookupFeature(prefix + node);
    cache.feat1.set(key, id);
  }
  return id;
}

// Two-character context features (c-1c., cc1., c-2c-1.) keyed by char pair.
function feat2(model, kind, prefix, a, aId, b, bId) {
  if (aId >= ID_STRIDE || bId >= ID_STRIDE) {
    return model.lookupFeature(`${prefix}${a}.${b}`);
  }
  const cache = model.optCache;
  const key = (kind * ID_STRIDE + aId) * ID_STRIDE + bId;
  let id = cache.feat2.get(key);
  if (id === undefined) {
    id = model.lookupFeature(`${prefix}${a}.${b}`);
    cache.feat2.set(key, id);
  }
  return id;
}

const windowHasher = new PairHasher();
const emptyHasher = new PairHasher();
emptyHasher.digest();
const EMPTY_FIRST = emptyHasher.hashFirst;
const EMPTY_SECOND = emptyHasher.hashSecond;

// Hash and membership-check a character window without materializing the
// window string. Returns 0 for a miss ("**noWord"), otherwise the interned
// gram id. Only hits are interned, so the table is bounded by the model's
// unigram table itself and needs no size cap.
function gramIdFor(model, nodes, start, length) {
  const allLength = nodes.length;
  const inBounds = start >= 0 && start < allLength && start + length <= allLength;
  let first;
  let second;
  if (inBounds) {
    windowHasher.reset();
    for (let index = start; index < start + length; index += 1) {
      windowHasher.feed(nodes[index]);
    }
    windowHasher.digest();
    first = windowHasher.hashFirst;
    second = windowHasher.hashSecond;
  } else {
    first = EMPTY_FIRST;
    second = EMPTY_SECOND;
  }
  if (!model.hasUnigramHash(first, second)) {
    return 0;
  }

  const cache = model.optCache;
  let bySecond = cache.gramIds.get(first);
  if (bySecond !== undefined) {
    const known = bySecond.get(second);
    if (known !== undefined) {
      return known;
    }
  }
  const value = inBounds ? nodes.slice(start, start + length).join("") : "";
  const id = cache.gramStrs.length;
  if (bySecond === undefined) {
    bySecond = new Map();
    cache.gramIds.set(first, bySecond);
  }
  bySecond.set(second, id);
  cache.gramStrs.push(value);
  return id;
}

// w-1./w1. word features keyed by interned gram id.
function gramFeature(model, kind, prefix, gramId) {
  const cache = model.optCache;
  const key = kind * ID_STRIDE + gramId;
  let id = cache.gramFeat.get(key);
  if (id === undefined) {
    id = model.lookupFeature(prefix + cache.gramStrs[gramId]);
    cache.gramFeat.set(key, id);
  }
  return id;
}

// ww.l./ww.r. pair features keyed by interned gram ids; the pair space grows
// with corpus diversity, so entries are capped and overflow falls back to a
// direct (uncached) table lookup with identical results.
function wwFeature(model, direction, prefix, prefixId, suffixId) {
  const cache = model.optCache;
  const top = direction === 0 ? cache.wwL : cache.wwR;
  const inner = top.get(prefixId);
  if (inner !== undefined) {
    const hit = inner.get(suffixId);
    if (hit !== undefined) {
      return hit;
    }
  }
  const id = model.lookupFeature(
    `${prefix}${cache.gramStrs[prefixId]}*${cache.gramStrs[suffixId]}`,
  );
  if (cache.wwEntries < WW_CAP) {
    if (inner === undefined) {
      top.set(prefixId, new Map([[suffixId, id]]));
    } else {
      inner.set(suffixId, id);
    }
    cache.wwEntries += 1;
  }
  return id;
}

export function segmentationFeaturesAt(model, nodes, index) {
  return segmentationFeaturesInto(model, nodes, index, []);
}

// Reused-buffer variant: fills `features` (reset in place) so the fused
// Viterbi path does not allocate one array per character.
export function segmentationFeaturesInto(model, nodes, index, features) {
  features.length = 0;
  features.push(0);
  const length = nodes.length;
  const current = nodes[index];
  const currentId = internSegNode(model, current);

  pushId(features, feat1(model, 0, "c.", current, currentId));

  let previous = null;
  let previousId = -1;
  if (index > 0) {
    previous = nodes[index - 1];
    previousId = internSegNode(model, previous);
    pushId(features, feat1(model, 1, "c-1.", previous, previousId));
    pushId(features, feat2(model, 0, "c-1c.", previous, previousId, current, currentId));
  }

  if (index + 1 < length) {
    const next = nodes[index + 1];
    const nextId = internSegNode(model, next);
    pushId(features, feat1(model, 2, "c1.", next, nextId));
    pushId(features, feat2(model, 1, "cc1.", current, currentId, next, nextId));
  }

  if (index > 1) {
    const previousPrevious = nodes[index - 2];
    const previousPreviousId = internSegNode(model, previousPrevious);
    pushId(features, feat1(model, 3, "c-2.", previousPrevious, previousPreviousId));
    pushId(
      features,
      feat2(model, 2, "c-2c-1.", previousPrevious, previousPreviousId, previous, previousId),
    );
  }

  if (index + 2 < length) {
    const nextNext = nodes[index + 2];
    pushId(features, feat1(model, 4, "c2.", nextNext, internSegNode(model, nextNext)));
  }

  const wordMaximum = model.metadata.wordMaximum ?? 6;
  const wordMinimum = model.metadata.wordMinimum ?? 2;
  if (model.metadata.wordFeature === false) {
    return features;
  }

  const sizeCount = wordMaximum - wordMinimum + 1;
  // Gram ids in the same order the original loops produced window strings:
  // size wordMaximum down to wordMinimum; 0 stands for "**noWord".
  const prefixesIncludingCurrent = new Array(sizeCount);
  for (let size = wordMaximum; size >= wordMinimum; size -= 1) {
    const slot = wordMaximum - size;
    const gramId = gramIdFor(model, nodes, index - size + 1, size);
    prefixesIncludingCurrent[slot] = gramId;
    if (gramId !== 0) {
      pushId(features, gramFeature(model, 0, "w-1.", gramId));
    }
  }

  const suffixesIncludingCurrent = new Array(sizeCount);
  for (let size = wordMaximum; size >= wordMinimum; size -= 1) {
    const slot = wordMaximum - size;
    const gramId = gramIdFor(model, nodes, index, size);
    suffixesIncludingCurrent[slot] = gramId;
    if (gramId !== 0) {
      pushId(features, gramFeature(model, 1, "w1.", gramId));
    }
  }

  const prefixesExcludingCurrent = new Array(sizeCount);
  for (let size = wordMaximum; size >= wordMinimum; size -= 1) {
    prefixesExcludingCurrent[wordMaximum - size] = gramIdFor(
      model,
      nodes,
      index - size,
      size,
    );
  }

  const suffixesExcludingCurrent = new Array(sizeCount);
  for (let size = wordMaximum; size >= wordMinimum; size -= 1) {
    suffixesExcludingCurrent[wordMaximum - size] = gramIdFor(
      model,
      nodes,
      index + 1,
      size,
    );
  }

  for (let outer = 0; outer < sizeCount; outer += 1) {
    for (let inner = 0; inner < sizeCount; inner += 1) {
      pushId(
        features,
        wwFeature(
          model,
          0,
          "ww.l.",
          prefixesExcludingCurrent[outer],
          suffixesIncludingCurrent[inner],
        ),
      );
    }
  }

  for (let outer = 0; outer < sizeCount; outer += 1) {
    for (let inner = 0; inner < sizeCount; inner += 1) {
      pushId(
        features,
        wwFeature(
          model,
          1,
          "ww.r.",
          prefixesIncludingCurrent[outer],
          suffixesExcludingCurrent[inner],
        ),
      );
    }
  }

  return features;
}

function computeNormalizedPostagToken(token) {
  const renamedCharacters = Array.from(token, renameKeyword);
  if (
    renamedCharacters.length > 0 &&
    renamedCharacters.every((character) => NUMBERS.has(character))
  ) {
    return "**Num";
  }
  return renamedCharacters.join("");
}

// Token normalization is model-independent; capped so a huge corpus cannot
// grow the map without bound (overflow just recomputes).
const normalizedTokenCache = new Map();

export function normalizePostagToken(token) {
  let normalized = normalizedTokenCache.get(token);
  if (normalized === undefined) {
    normalized = computeNormalizedPostagToken(token);
    if (normalizedTokenCache.size < NORM_TOKEN_CAP) {
      normalizedTokenCache.set(token, normalized);
    }
  }
  return normalized;
}

function internPosWord(model, word) {
  const cache = model.optCache;
  let id = cache.wordIds.get(word);
  if (id === undefined) {
    if (cache.wordStrs.length >= WORD_CAP) {
      return -1;
    }
    id = cache.wordStrs.length;
    cache.wordIds.set(word, id);
    cache.wordStrs.push(word);
  }
  return id;
}

// Word-intrinsic features (w., tr1.pre/post) in their original push order.
function posIntrinsic(model, word, wordId) {
  const cache = model.optCache;
  if (wordId >= 0) {
    const known = cache.wordIntrinsic.get(wordId);
    if (known !== undefined) {
      return known;
    }
  }
  const ids = [];
  pushKnownFeature(model, ids, `w.${word}`);
  const wordCharacters = Array.from(word);
  for (let size = 1; size < 4; size += 1) {
    if (wordCharacters.length >= size) {
      pushKnownFeature(
        model,
        ids,
        `tr1.pre.${size}.${wordCharacters.slice(0, size).join("")}`,
      );
      pushKnownFeature(
        model,
        ids,
        `tr1.post.${size}.${wordCharacters.slice(-size).join("")}`,
      );
    }
  }
  if (wordId >= 0) {
    cache.wordIntrinsic.set(wordId, ids);
  }
  return ids;
}

// Word-context features (tr1.w-1., tr1.w1., tr1.w-2., tr1.w2.) by word id.
function posFeat1(model, kind, prefix, word, wordId) {
  if (wordId < 0) {
    return model.lookupFeature(prefix + word);
  }
  const cache = model.optCache;
  const key = kind * ID_STRIDE + wordId;
  let id = cache.posFeat1.get(key);
  if (id === undefined) {
    id = model.lookupFeature(prefix + word);
    cache.posFeat1.set(key, id);
  }
  return id;
}

// Word-pair features (tr1.w_-1_0., tr1.w_0_1.) by word id pair.
function posFeat2(model, kind, prefix, a, aId, b, bId) {
  if (aId < 0 || bId < 0) {
    return model.lookupFeature(`${prefix}${a}.${b}`);
  }
  const cache = model.optCache;
  const key = (kind * ID_STRIDE + aId) * ID_STRIDE + bId;
  let id = cache.posFeat2.get(key);
  if (id === undefined) {
    id = model.lookupFeature(`${prefix}${a}.${b}`);
    cache.posFeat2.set(key, id);
  }
  return id;
}

function posConstants(model) {
  const cache = model.optCache;
  if (cache.posConst === null) {
    cache.posConst = {
      prev1: model.lookupFeature("tr1.w-1.BOS"),
      next1: model.lookupFeature("tr1.w1.EOS"),
      prev2: model.lookupFeature("tr1.w-2.BOS"),
      next2: model.lookupFeature("tr1.w2.EOS"),
      pairPrev: model.lookupFeature("tr1.w_-1_0.BOS"),
      pairNext: model.lookupFeature("tr1.w_0_1.EOS"),
    };
  }
  return cache.posConst;
}

export function postagFeaturesAt(model, nodes, index) {
  return postagFeaturesInto(model, nodes, index, []);
}

// Reused-buffer variant, same rationale as segmentationFeaturesInto.
export function postagFeaturesInto(model, nodes, index, features) {
  features.length = 0;
  features.push(0);
  const length = nodes.length;
  const word = nodes[index];
  const wordId = internPosWord(model, word);
  const constants = posConstants(model);

  for (const id of posIntrinsic(model, word, wordId)) {
    features.push(id);
  }

  let previous = null;
  let previousId = -1;
  if (index > 0) {
    previous = nodes[index - 1];
    previousId = internPosWord(model, previous);
    pushId(features, posFeat1(model, 0, "tr1.w-1.", previous, previousId));
  } else {
    pushId(features, constants.prev1);
  }

  let next = null;
  let nextId = -1;
  if (index < length - 1) {
    next = nodes[index + 1];
    nextId = internPosWord(model, next);
    pushId(features, posFeat1(model, 1, "tr1.w1.", next, nextId));
  } else {
    pushId(features, constants.next1);
  }

  if (index > 1) {
    const previousPrevious = nodes[index - 2];
    pushId(
      features,
      posFeat1(model, 2, "tr1.w-2.", previousPrevious, internPosWord(model, previousPrevious)),
    );
  } else {
    pushId(features, constants.prev2);
  }

  if (index + 2 < length) {
    const nextNext = nodes[index + 2];
    pushId(
      features,
      posFeat1(model, 3, "tr1.w2.", nextNext, internPosWord(model, nextNext)),
    );
  } else {
    pushId(features, constants.next2);
  }

  if (index > 0) {
    pushId(features, posFeat2(model, 0, "tr1.w_-1_0.", previous, previousId, word, wordId));
  } else {
    pushId(features, constants.pairPrev);
  }

  if (index < length - 1) {
    pushId(features, posFeat2(model, 1, "tr1.w_0_1.", word, wordId, next, nextId));
  } else {
    pushId(features, constants.pairNext);
  }

  return features;
}
