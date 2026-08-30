// Viterbi decoding over per-node feature-id lists. decodeViterbi keeps the
// original allocating behavior; decodeViterbiFused is the hot-path variant
// that computes features on the fly into reused scratch buffers, avoiding
// the per-node feature-array materialization entirely.

export function createViterbiScratch() {
  return {
    nodeScores: new Float64Array(0),
    maximumScores: new Float64Array(0),
    previousTags: new Int32Array(0),
    states: new Int32Array(0),
    features: [],
  };
}

function ensureScratchCapacity(scratch, nodeCount, tagCount) {
  const cells = nodeCount * tagCount;
  if (scratch.nodeScores.length < cells) {
    let capacity = Math.max(1024, scratch.nodeScores.length);
    while (capacity < cells) {
      capacity *= 2;
    }
    scratch.nodeScores = new Float64Array(capacity);
    scratch.maximumScores = new Float64Array(capacity);
    scratch.previousTags = new Int32Array(capacity);
    scratch.states = new Int32Array(nodeCount);
  } else if (scratch.states.length < nodeCount) {
    scratch.states = new Int32Array(nodeCount);
  }
}

// The transition matrix depends only on model weights, so it is built once
// per model instead of once per cut call.
export function modelEdgeScores(model) {
  const cache = model.optCache ?? null;
  let edgeScores = cache === null ? null : cache.edgeScores;
  if (edgeScores === null) {
    const tagCount = model.nTag;
    edgeScores = new Float64Array(tagCount * tagCount);
    edgeScores.fill(1);
    const backoff = model.nFeature * tagCount;
    for (let tag = 0; tag < tagCount; tag += 1) {
      for (let previousTag = 0; previousTag < tagCount; previousTag += 1) {
        edgeScores[previousTag * tagCount + tag] +=
          model.weights[backoff + tag * tagCount + previousTag];
      }
    }
    if (cache !== null) {
      cache.edgeScores = edgeScores;
    }
  }
  return edgeScores;
}

// Shared backward pass + backtrace. Reads nodeScores[0 .. nodeCount*tagCount)
// and writes maximumScores/previousTags/states. Identical arithmetic to the
// original implementation; only buffer ownership differs.
function viterbiBackward(model, nodeCount, nodeScores, maximumScores, previousTags, states) {
  const tagCount = model.nTag;
  const edgeScores = modelEdgeScores(model);
  const lastOffset = (nodeCount - 1) * tagCount;
  for (let tag = 0; tag < tagCount; tag += 1) {
    maximumScores[lastOffset + tag] = nodeScores[lastOffset + tag];
  }

  for (let node = nodeCount - 2; node >= 0; node -= 1) {
    const rowOffset = node * tagCount;
    const nextOffset = (node + 1) * tagCount;
    for (let tag = 0; tag < tagCount; tag += 1) {
      let bestPreviousTag = 0;
      let bestScore =
        maximumScores[nextOffset] +
        nodeScores[rowOffset + tag] +
        edgeScores[tag * tagCount];

      for (let previousTag = 1; previousTag < tagCount; previousTag += 1) {
        const score =
          maximumScores[nextOffset + previousTag] +
          nodeScores[rowOffset + tag] +
          edgeScores[tag * tagCount + previousTag];
        if (score >= bestScore) {
          bestScore = score;
          bestPreviousTag = previousTag;
        }
      }

      maximumScores[rowOffset + tag] = bestScore;
      previousTags[rowOffset + tag] = bestPreviousTag;
    }
  }

  let bestTag = 0;
  let bestScore = maximumScores[0];
  for (let tag = 1; tag < tagCount; tag += 1) {
    if (bestScore < maximumScores[tag]) {
      bestScore = maximumScores[tag];
      bestTag = tag;
    }
  }

  states[0] = bestTag;
  for (let node = 1; node < nodeCount; node += 1) {
    bestTag = previousTags[(node - 1) * tagCount + bestTag];
    states[node] = bestTag;
  }
}

export function decodeViterbi(featureLists, model) {
  const nodeCount = featureLists.length;
  const tagCount = model.nTag;
  if (nodeCount === 0) {
    return [];
  }

  const nodeScores = new Float64Array(nodeCount * tagCount);
  for (let node = 0; node < nodeCount; node += 1) {
    const rowOffset = node * tagCount;
    for (const feature of featureLists[node]) {
      const weightOffset = feature * tagCount;
      for (let tag = 0; tag < tagCount; tag += 1) {
        nodeScores[rowOffset + tag] += model.weights[weightOffset + tag];
      }
    }
  }

  const maximumScores = new Float64Array(nodeCount * tagCount);
  const previousTags = new Int32Array(nodeCount * tagCount);
  const states = new Int32Array(nodeCount);
  viterbiBackward(model, nodeCount, nodeScores, maximumScores, previousTags, states);
  return Array.from(states);
}

// Fused variant: fillFeatures(nodeIndex, out) appends the node's feature ids
// into the reused `out` array. Returns the scratch-owned states buffer, valid
// until the next call with the same scratch — consume it immediately.
export function decodeViterbiFused(model, nodeCount, fillFeatures, scratch) {
  const tagCount = model.nTag;
  if (nodeCount === 0) {
    return new Int32Array(0);
  }

  ensureScratchCapacity(scratch, nodeCount, tagCount);
  const { nodeScores, maximumScores, previousTags, states, features } = scratch;
  const cellCount = nodeCount * tagCount;
  nodeScores.fill(0, 0, cellCount);

  for (let node = 0; node < nodeCount; node += 1) {
    const rowOffset = node * tagCount;
    features.length = 0;
    fillFeatures(node, features);
    for (let index = 0; index < features.length; index += 1) {
      const weightOffset = features[index] * tagCount;
      for (let tag = 0; tag < tagCount; tag += 1) {
        nodeScores[rowOffset + tag] += model.weights[weightOffset + tag];
      }
    }
  }

  viterbiBackward(model, nodeCount, nodeScores, maximumScores, previousTags, states);
  return states;
}
