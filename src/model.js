import { hashPair } from "./hash.js";

const MAGIC = "PKUSEGJS";
const FORMAT_VERSION = 1;
const HEADER_BYTES = 64;

function align(value, alignment = 8) {
  return Math.ceil(value / alignment) * alignment;
}

class FeatureTable {
  constructor(firstHashes, secondHashes, identifiers) {
    this.firstHashes = firstHashes;
    this.secondHashes = secondHashes;
    this.identifiers = identifiers;
    this.mask = identifiers.length - 1;
  }

  lookup(text) {
    const [first, second] = hashPair(text);
    return this.lookupHash(first, second);
  }

  lookupHash(first, second) {
    let slot = first & this.mask;

    for (let probes = 0; probes < this.identifiers.length; probes += 1) {
      const storedIdentifier = this.identifiers[slot];
      if (storedIdentifier === 0) {
        return -1;
      }
      if (
        this.firstHashes[slot] === first &&
        this.secondHashes[slot] === second
      ) {
        return storedIdentifier - 1;
      }
      slot = (slot + 1) & this.mask;
    }

    throw new Error("Corrupt PKUSEG feature table: probe cycle exhausted");
  }
}

class MembershipTable {
  constructor(firstHashes, secondHashes, occupied) {
    this.firstHashes = firstHashes;
    this.secondHashes = secondHashes;
    this.occupied = occupied;
    this.mask = occupied.length === 0 ? 0 : occupied.length - 1;
  }

  has(text) {
    if (this.occupied.length === 0) {
      return false;
    }

    const [first, second] = hashPair(text);
    return this.hasHash(first, second);
  }

  hasHash(first, second) {
    if (this.occupied.length === 0) {
      return false;
    }

    let slot = first & this.mask;
    for (let probes = 0; probes < this.occupied.length; probes += 1) {
      if (this.occupied[slot] === 0) {
        return false;
      }
      if (
        this.firstHashes[slot] === first &&
        this.secondHashes[slot] === second
      ) {
        return true;
      }
      slot = (slot + 1) & this.mask;
    }

    throw new Error("Corrupt PKUSEG membership table: probe cycle exhausted");
  }
}

function copyToAlignedBytes(bytes) {
  if ((bytes.byteOffset & 7) === 0) {
    return bytes;
  }
  return Uint8Array.from(bytes);
}

export class Model {
  static fromBytes(source, metadata) {
    const original =
      source instanceof Uint8Array ? source : new Uint8Array(source);
    const bytes = copyToAlignedBytes(original);
    if (bytes.byteLength < HEADER_BYTES) {
      throw new Error("PKUSEG model is shorter than its header");
    }

    const magic = String.fromCharCode(...bytes.subarray(0, MAGIC.length));
    if (magic !== MAGIC) {
      throw new Error(`Invalid PKUSEG model magic: ${JSON.stringify(magic)}`);
    }

    const header = new DataView(bytes.buffer, bytes.byteOffset, HEADER_BYTES);
    const version = header.getUint32(8, true);
    const headerBytes = header.getUint32(12, true);
    if (version !== FORMAT_VERSION || headerBytes !== HEADER_BYTES) {
      throw new Error(
        `Unsupported PKUSEG model format ${version}/${headerBytes}`,
      );
    }

    const featureTableSize = header.getUint32(16, true);
    const unigramTableSize = header.getUint32(20, true);
    const dictionaryTableSize = header.getUint32(24, true);
    const weightCount = header.getUint32(28, true);
    const nFeature = header.getUint32(32, true);
    const nTag = header.getUint32(36, true);

    for (const [name, size] of [
      ["feature", featureTableSize],
      ["unigram", unigramTableSize],
      ["dictionary", dictionaryTableSize],
    ]) {
      if (size !== 0 && (size & (size - 1)) !== 0) {
        throw new Error(`${name} table size is not a power of two: ${size}`);
      }
    }

    const base = bytes.byteOffset;
    let offset = HEADER_BYTES;
    const uint32View = (length) => {
      const view = new Uint32Array(bytes.buffer, base + offset, length);
      offset += length * Uint32Array.BYTES_PER_ELEMENT;
      return view;
    };
    const uint8View = (length) => {
      const view = new Uint8Array(bytes.buffer, base + offset, length);
      offset += length;
      return view;
    };

    const featureFirst = uint32View(featureTableSize);
    const featureSecond = uint32View(featureTableSize);
    const featureIdentifiers = uint32View(featureTableSize);

    const unigramFirst = uint32View(unigramTableSize);
    const unigramSecond = uint32View(unigramTableSize);
    const unigramOccupied = uint8View(unigramTableSize);
    offset = align(offset);

    const dictionaryFirst = uint32View(dictionaryTableSize);
    const dictionarySecond = uint32View(dictionaryTableSize);
    const dictionaryOccupied = uint8View(dictionaryTableSize);
    offset = align(offset);

    const weights = new Float64Array(bytes.buffer, base + offset, weightCount);
    offset += weightCount * Float64Array.BYTES_PER_ELEMENT;
    if (offset > bytes.byteLength) {
      throw new Error("Truncated PKUSEG model data");
    }
    if (weightCount !== nTag * (nFeature + nTag)) {
      throw new Error(
        `Inconsistent model shape: ${weightCount} != ${nTag} * (${nFeature} + ${nTag})`,
      );
    }
    if (metadata.nFeature !== nFeature || metadata.nTag !== nTag) {
      throw new Error("Binary model and metadata disagree about model shape");
    }
    for (const vector of metadata.hashVectors ?? []) {
      const actual = hashPair(vector.text);
      if (actual[0] !== vector.hash[0] || actual[1] !== vector.hash[1]) {
        throw new Error(
          `Model hash implementation mismatch for ${JSON.stringify(vector.text)}`,
        );
      }
    }

    return new Model({
      bytes,
      metadata,
      nFeature,
      nTag,
      weights,
      featureTable: new FeatureTable(
        featureFirst,
        featureSecond,
        featureIdentifiers,
      ),
      unigramTable: new MembershipTable(
        unigramFirst,
        unigramSecond,
        unigramOccupied,
      ),
      dictionaryTable: new MembershipTable(
        dictionaryFirst,
        dictionarySecond,
        dictionaryOccupied,
      ),
    });
  }

  constructor({
    bytes,
    metadata,
    nFeature,
    nTag,
    weights,
    featureTable,
    unigramTable,
    dictionaryTable,
  }) {
    this._bytes = bytes;
    this.metadata = metadata;
    this.nFeature = nFeature;
    this.nTag = nTag;
    this.weights = weights;
    this.featureTable = featureTable;
    this.unigramTable = unigramTable;
    this.dictionaryTable = dictionaryTable;
    // Per-model memoization for the hot feature-lookup path. All entries are
    // pure cache: every miss falls back to the same table lookup the engine
    // always used, so outputs stay byte-identical. Maps are size-capped by
    // the feature extractor and degrade to direct lookups past the cap.
    this.optCache = {
      nodeIds: new Map(), // normalized segmentation char -> integer id
      nodeStrs: [], // integer id -> normalized char
      feat1: new Map(), // single-char context feature -> featureId | -1
      feat2: new Map(), // two-char context feature -> featureId | -1
      gramIds: new Map(), // window hash first -> Map(second -> gramId)
      gramStrs: ["**noWord"], // gramId -> window string; id 0 means no hit
      gramFeat: new Map(), // w-1./w1. feature by gramId -> featureId | -1
      wwL: new Map(), // prefixGramId -> Map(suffixGramId -> featureId | -1)
      wwR: new Map(),
      wwEntries: 0,
      normToken: new Map(), // raw postag token -> normalized token
      wordIds: new Map(), // normalized postag word -> integer id
      wordStrs: [],
      wordIntrinsic: new Map(), // wordId -> known intrinsic featureIds
      posFeat1: new Map(), // word-context feature -> featureId | -1
      posFeat2: new Map(), // word-pair feature -> featureId | -1
      posConst: null, // cached BOS/EOS feature ids
      edgeScores: null, // transition matrix, built once per model
    };
  }

  lookupFeature(text) {
    return this.featureTable.lookup(text);
  }

  lookupFeatureHash(first, second) {
    return this.featureTable.lookupHash(first, second);
  }

  hasUnigram(text) {
    return this.unigramTable.has(text);
  }

  hasUnigramHash(first, second) {
    return this.unigramTable.hasHash(first, second);
  }

  hasDictionaryWord(text) {
    return this.dictionaryTable.has(text);
  }
}
