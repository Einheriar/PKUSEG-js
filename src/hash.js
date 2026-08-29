function rotateLeft32(value, bits) {
  return ((value << bits) | (value >>> (32 - bits))) >>> 0;
}

/**
 * Hash a JavaScript string by Unicode code point, matching tools/export_model.py.
 * Two independently mixed 32-bit values make accidental model false positives
 * vanishingly unlikely without retaining millions of JavaScript strings.
 */
export function hashPair(text) {
  let first = 0x811c_9dc5;
  let second = 0x9747_b28c;
  let length = 0;

  for (const character of text) {
    const codePoint = character.codePointAt(0);

    first = Math.imul((first ^ codePoint) >>> 0, 0x0100_0193) >>> 0;

    let mixed = Math.imul(codePoint, 0xcc9e_2d51) >>> 0;
    mixed = rotateLeft32(mixed, 15);
    mixed = Math.imul(mixed, 0x1b87_3593) >>> 0;
    second = (second ^ mixed) >>> 0;
    second = rotateLeft32(second, 13);
    second = (Math.imul(second, 5) + 0xe654_6b64) >>> 0;
    length += 1;
  }

  second = (second ^ length) >>> 0;
  second = (second ^ (second >>> 16)) >>> 0;
  second = Math.imul(second, 0x85eb_ca6b) >>> 0;
  second = (second ^ (second >>> 13)) >>> 0;
  second = Math.imul(second, 0xc2b2_ae35) >>> 0;
  second = (second ^ (second >>> 16)) >>> 0;

  return [first >>> 0, second >>> 0];
}
