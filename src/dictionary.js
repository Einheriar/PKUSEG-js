class TrieNode {
  constructor() {
    this.isWord = false;
    this.userTag = "";
    this.children = new Map();
  }
}

function normalizeEntry(entry) {
  if (typeof entry === "string") {
    return [entry.trim(), ""];
  }
  if (Array.isArray(entry) && entry.length === 2) {
    return [String(entry[0]).trim(), String(entry[1]).trim()];
  }
  throw new TypeError("User dictionary entries must be strings or [word, tag]");
}

export class UserDictionary {
  constructor(entries = []) {
    this.root = new TrieNode();
    for (const entry of entries) {
      const [word, tag] = normalizeEntry(entry);
      if (word.length > 0) {
        this.insert(word, tag);
      }
    }
  }

  insert(word, tag = "") {
    let node = this.root;
    for (const character of word) {
      let child = node.children.get(character);
      if (child === undefined) {
        child = new TrieNode();
        node.children.set(character, child);
      }
      node = child;
    }
    node.isWord = true;
    node.userTag = tag;
  }

  split(text) {
    const characters = Array.from(text);
    const parts = [];
    const isWords = [];
    const tags = [];
    let last = 0;
    let index = 0;

    while (index < characters.length) {
      let node = this.root;
      let longestEnd = -1;
      let userTag = "";

      for (let cursor = index; cursor < characters.length; cursor += 1) {
        node = node.children.get(characters[cursor]);
        if (node === undefined) {
          break;
        }
        if (node.isWord) {
          longestEnd = cursor + 1;
          userTag = node.userTag;
        }
      }

      if (longestEnd !== -1) {
        if (last !== index) {
          parts.push(characters.slice(last, index).join(""));
          isWords.push(false);
          tags.push("");
        }
        parts.push(characters.slice(index, longestEnd).join(""));
        isWords.push(true);
        tags.push(userTag);
        last = longestEnd;
        index = longestEnd;
      } else {
        index += 1;
      }
    }

    if (last < characters.length) {
      parts.push(characters.slice(last).join(""));
      isWords.push(false);
      tags.push("");
    }
    return { parts, isWords, tags };
  }
}

export function applyDefaultDictionary(words, model) {
  const sentence = words.slice();
  for (let mergeSize = 7; mergeSize >= 2; mergeSize -= 1) {
    let end = sentence.length - mergeSize;
    if (end < 0) {
      continue;
    }

    let index = 0;
    while (index <= end) {
      const merged = sentence.slice(index, index + mergeSize).join("");
      let shouldMerge = false;
      if (model.hasDictionaryWord(merged)) {
        const alreadySeparated = sentence
          .slice(index, index + mergeSize)
          .every((word) => model.hasDictionaryWord(word));
        shouldMerge = !alreadySeparated;
      }

      if (shouldMerge) {
        sentence.splice(index, mergeSize, merged);
        index += 1;
        end = sentence.length - mergeSize;
      } else {
        index += 1;
      }
    }
  }
  return sentence;
}
