export function decodeViterbi(featureLists, model) {
  const nodeCount = featureLists.length;
  const tagCount = model.nTag;
  if (nodeCount === 0) {
    return [];
  }

  const nodeScores = new Float64Array(nodeCount * tagCount);
  const edgeScores = new Float64Array(tagCount * tagCount);
  edgeScores.fill(1);

  for (let node = 0; node < nodeCount; node += 1) {
    const rowOffset = node * tagCount;
    for (const feature of featureLists[node]) {
      const weightOffset = feature * tagCount;
      for (let tag = 0; tag < tagCount; tag += 1) {
        nodeScores[rowOffset + tag] += model.weights[weightOffset + tag];
      }
    }
  }

  const backoff = model.nFeature * tagCount;
  for (let tag = 0; tag < tagCount; tag += 1) {
    for (let previousTag = 0; previousTag < tagCount; previousTag += 1) {
      edgeScores[previousTag * tagCount + tag] +=
        model.weights[backoff + tag * tagCount + previousTag];
    }
  }

  const maximumScores = new Float64Array(nodeCount * tagCount);
  const previousTags = new Int32Array(nodeCount * tagCount);
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

  const states = new Int32Array(nodeCount);
  states[0] = bestTag;
  for (let node = 1; node < nodeCount; node += 1) {
    bestTag = previousTags[(node - 1) * tagCount + bestTag];
    states[node] = bestTag;
  }
  return Array.from(states);
}
