import { performance } from "node:perf_hooks";
import { loadSegmenter } from "../src/index.js";

const loadStarted = performance.now();
const segmenter = await loadSegmenter();
const loadMilliseconds = performance.now() - loadStarted;

const texts = [
  "我爱北京天安门，中国人民银行今天发布公告。",
  "机器学习与自然语言处理正在快速发展。",
  "患者出现慢性支气管炎，建议及时就医。",
];
const input = Array.from({ length: 1_000 }, (_, index) => texts[index % texts.length]);
const characterCount = input.reduce(
  (total, text) => total + Array.from(text).length,
  0,
);
const started = performance.now();
let tokenCount = 0;
for (const text of input) {
  tokenCount += segmenter.cut(text).length;
}
const elapsedSeconds = (performance.now() - started) / 1_000;

console.log(
  JSON.stringify(
    {
      modelLoadMilliseconds: Math.round(loadMilliseconds * 100) / 100,
      characterCount,
      tokenCount,
      elapsedSeconds: Math.round(elapsedSeconds * 10_000) / 10_000,
      charactersPerSecond: Math.round(characterCount / elapsedSeconds),
      residentMemoryMiB:
        Math.round((process.memoryUsage().rss / 1024 / 1024) * 100) / 100,
    },
    null,
    2,
  ),
);
