import { loadSegmenter } from "../src/index.js";

const segmenter = await loadSegmenter({ postag: true });
console.log(segmenter.cut("我爱北京天安门"));
