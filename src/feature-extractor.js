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

function renameKeyword(character) {
  return KEYWORDS.has(character) ? "&" : character;
}

function normalizedSegmentationCharacter(character) {
  const renamed = renameKeyword(character);
  if (NUMBERS.has(renamed)) {
    return "**Num";
  }
  if (LETTERS.has(renamed)) {
    return "**Letter";
  }
  return renamed;
}

export function normalizeSegmentationText(text) {
  return Array.from(text, normalizedSegmentationCharacter);
}

function sliceCharacters(characters, start, length) {
  const allLength = characters.length;
  if (start < 0 || start >= allLength || start + length >= allLength + 1) {
    return "";
  }
  return characters.slice(start, start + length).join("");
}

function pushKnownFeature(model, output, feature) {
  const identifier = model.lookupFeature(feature);
  if (identifier !== -1) {
    output.push(identifier);
  }
}

export function segmentationFeaturesAt(model, nodes, index) {
  const features = [0];
  const length = nodes.length;
  const current = nodes[index];

  pushKnownFeature(model, features, `c.${current}`);

  if (index > 0) {
    const previous = nodes[index - 1];
    pushKnownFeature(model, features, `c-1.${previous}`);
    pushKnownFeature(model, features, `c-1c.${previous}.${current}`);
  }

  if (index + 1 < length) {
    const next = nodes[index + 1];
    pushKnownFeature(model, features, `c1.${next}`);
    pushKnownFeature(model, features, `cc1.${current}.${next}`);
  }

  if (index > 1) {
    const previousPrevious = nodes[index - 2];
    pushKnownFeature(model, features, `c-2.${previousPrevious}`);
    pushKnownFeature(
      model,
      features,
      `c-2c-1.${previousPrevious}.${nodes[index - 1]}`,
    );
  }

  if (index + 2 < length) {
    pushKnownFeature(model, features, `c2.${nodes[index + 2]}`);
  }

  const wordMaximum = model.metadata.wordMaximum ?? 6;
  const wordMinimum = model.metadata.wordMinimum ?? 2;
  if (model.metadata.wordFeature === false) {
    return features;
  }

  const prefixesIncludingCurrent = [];
  for (let size = wordMaximum; size >= wordMinimum; size -= 1) {
    const value = sliceCharacters(nodes, index - size + 1, size);
    if (model.hasUnigram(value)) {
      pushKnownFeature(model, features, `w-1.${value}`);
      prefixesIncludingCurrent.push(value);
    } else {
      prefixesIncludingCurrent.push("**noWord");
    }
  }

  const suffixesIncludingCurrent = [];
  for (let size = wordMaximum; size >= wordMinimum; size -= 1) {
    const value = sliceCharacters(nodes, index, size);
    if (model.hasUnigram(value)) {
      pushKnownFeature(model, features, `w1.${value}`);
      suffixesIncludingCurrent.push(value);
    } else {
      suffixesIncludingCurrent.push("**noWord");
    }
  }

  const prefixesExcludingCurrent = [];
  for (let size = wordMaximum; size >= wordMinimum; size -= 1) {
    const value = sliceCharacters(nodes, index - size, size);
    prefixesExcludingCurrent.push(
      model.hasUnigram(value) ? value : "**noWord",
    );
  }

  const suffixesExcludingCurrent = [];
  for (let size = wordMaximum; size >= wordMinimum; size -= 1) {
    const value = sliceCharacters(nodes, index + 1, size);
    suffixesExcludingCurrent.push(
      model.hasUnigram(value) ? value : "**noWord",
    );
  }

  for (const prefix of prefixesExcludingCurrent) {
    for (const suffix of suffixesIncludingCurrent) {
      pushKnownFeature(model, features, `ww.l.${prefix}*${suffix}`);
    }
  }

  for (const prefix of prefixesIncludingCurrent) {
    for (const suffix of suffixesExcludingCurrent) {
      pushKnownFeature(model, features, `ww.r.${prefix}*${suffix}`);
    }
  }

  return features;
}

export function normalizePostagToken(token) {
  const renamedCharacters = Array.from(token, renameKeyword);
  if (
    renamedCharacters.length > 0 &&
    renamedCharacters.every((character) => NUMBERS.has(character))
  ) {
    return "**Num";
  }
  return renamedCharacters.join("");
}

export function postagFeaturesAt(model, nodes, index) {
  const features = [0];
  const length = nodes.length;
  const word = nodes[index];
  const wordCharacters = Array.from(word);

  pushKnownFeature(model, features, `w.${word}`);

  for (let size = 1; size < 4; size += 1) {
    if (wordCharacters.length >= size) {
      pushKnownFeature(
        model,
        features,
        `tr1.pre.${size}.${wordCharacters.slice(0, size).join("")}`,
      );
      pushKnownFeature(
        model,
        features,
        `tr1.post.${size}.${wordCharacters.slice(-size).join("")}`,
      );
    }
  }

  pushKnownFeature(
    model,
    features,
    index > 0 ? `tr1.w-1.${nodes[index - 1]}` : "tr1.w-1.BOS",
  );
  pushKnownFeature(
    model,
    features,
    index < length - 1 ? `tr1.w1.${nodes[index + 1]}` : "tr1.w1.EOS",
  );
  pushKnownFeature(
    model,
    features,
    index > 1 ? `tr1.w-2.${nodes[index - 2]}` : "tr1.w-2.BOS",
  );
  pushKnownFeature(
    model,
    features,
    index < length - 2 ? `tr1.w2.${nodes[index + 2]}` : "tr1.w2.EOS",
  );
  pushKnownFeature(
    model,
    features,
    index > 0
      ? `tr1.w_-1_0.${nodes[index - 1]}.${word}`
      : "tr1.w_-1_0.BOS",
  );
  pushKnownFeature(
    model,
    features,
    index < length - 1
      ? `tr1.w_0_1.${word}.${nodes[index + 1]}`
      : "tr1.w_0_1.EOS",
  );

  return features;
}
