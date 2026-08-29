import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { before, test } from "node:test";

import { loadModel, loadSegmenter } from "../src/index.js";

const defaultModelDirectory = fileURLToPath(
  new URL("../models/default/", import.meta.url),
);
const cases = [
  "",
  "   \n\t",
  "我爱北京天安门",
  "中国人民银行今天发布公告",
  "机器学习与自然语言处理正在快速发展。",
  "新版本Python3.12已经发布。",
  "自然语言处理（NLP）很有趣！",
  "2026年8月29日，天气不错。",
  "南京市长江大桥",
  "患者出现慢性支气管炎，建议及时就医。",
  "这个帖子真的很好玩！！！",
  "故宫博物院值得参观",
  "A股上涨3.14%，成交额超过1万亿元。",
  "hello-world@example.com",
  "第一段  第二段\t第三段",
  "繁體中文與简体中文混合测试",
  "表情😀也能参与分词吗？",
  "𠀀是一个扩展区汉字",
  "一二三四五六七八九十",
  "北京大学生前来应聘",
];
const expected = [
  [],
  [],
  ["我", "爱", "北京", "天安门"],
  ["中国", "人民", "银行", "今天", "发布", "公告"],
  ["机器", "学习", "与", "自然", "语言", "处理", "正在", "快速", "发展", "。"],
  ["新", "版本", "Python3.12", "已经", "发布", "。"],
  ["自然", "语言", "处理", "（", "NLP", "）", "很", "有趣", "！"],
  ["2026年", "8月", "29日", "，", "天气", "不错", "。"],
  ["南京市", "长江", "大桥"],
  ["患者", "出现", "慢性", "支气管炎", "，", "建议", "及时", "就医", "。"],
  ["这个", "帖子", "真的", "很", "好玩", "！", "！", "！"],
  ["故宫", "博物院", "值得", "参观"],
  ["A股", "上涨", "3.14%", "，", "成交额", "超过", "1万亿", "元", "。"],
  ["hello-world", "@", "example.com"],
  ["第一", "段", "第二", "段", "第三", "段"],
  ["繁體", "中文", "與", "简体", "中文", "混合", "测试"],
  ["表情", "😀", "也", "能", "参与", "分词", "吗", "？"],
  ["𠀀", "是", "一个", "扩展区", "汉字"],
  ["一二三四五六七八九十"],
  ["北京", "大学生", "前来", "应聘"],
];

let segmenter;
let taggingSegmenter;

before(async () => {
  [segmenter, taggingSegmenter] = await Promise.all([
    loadSegmenter(),
    loadSegmenter({ postag: true }),
  ]);
});

test("loads the converted model with its exact shape", async () => {
  const model = await loadModel(defaultModelDirectory);
  assert.equal(model.nFeature, 2_357_840);
  assert.equal(model.nTag, 5);
  assert.equal(model.weights.length, 11_789_225);
  assert.equal(model.lookupFeature("$$"), 0);
  assert.notEqual(model.lookupFeature("c.我"), -1);
  assert.equal(model.lookupFeature("feature.that.does.not.exist"), -1);
});

test("matches Python pkuseg segmentation on the golden corpus", () => {
  assert.deepEqual(cases.map((value) => segmenter.cut(value)), expected);
});

test("matches Python pkuseg part-of-speech output", () => {
  assert.deepEqual(taggingSegmenter.cut("我爱北京天安门"), [
    ["我", "r"],
    ["爱", "v"],
    ["北京", "ns"],
    ["天安门", "ns"],
  ]);
  assert.deepEqual(taggingSegmenter.cut("患者出现慢性支气管炎"), [
    ["患者", "n"],
    ["出现", "v"],
    ["慢性", "b"],
    ["支气管炎", "n"],
  ]);
});

test("supports in-memory user dictionaries and user POS tags", async () => {
  const custom = await loadSegmenter({
    userDictionary: ["自然语言处理", ["北京天安门", "custom"]],
    postag: true,
  });
  assert.deepEqual(custom.cut("我喜欢自然语言处理"), [
    ["我", "r"],
    ["喜欢", "v"],
    ["自然语言处理", "n"],
  ]);
  assert.deepEqual(custom.cut("我爱北京天安门"), [
    ["我", "r"],
    ["爱", "v"],
    ["北京天安门", "custom"],
  ]);
});

test("supports UTF-8 user dictionary files", async () => {
  const custom = await loadSegmenter({
    userDictionary: fileURLToPath(new URL("./user-dict.txt", import.meta.url)),
  });
  assert.deepEqual(custom.cut("我喜欢自然语言处理"), [
    "我",
    "喜欢",
    "自然语言处理",
  ]);
});

test("user dictionary dead-end falls back to the longest matched word", async () => {
  // Upstream bug #137: '车' is a prefix of '车在中国'; when the text follows
  // the longer path and dies, only the matched word must be emitted.
  const custom = await loadSegmenter({ userDictionary: ["车", "车在中国"] });
  assert.deepEqual(custom.cut("电动车在上海"), ["电动", "车", "在", "上海"]);
});

test("user dictionary entries containing spaces still match", async () => {
  // Upstream bug #109: dictionary matching must run before whitespace
  // splitting, otherwise entries like 'Color OS' can never hit.
  const custom = await loadSegmenter({ userDictionary: ["Color OS", "前摄像头"] });
  assert.deepEqual(custom.cut("Color OS"), ["Color OS"]);
  assert.deepEqual(custom.cut("Color OS 的前摄像头很好用"), [
    "Color OS",
    "的",
    "前摄像头",
    "很",
    "好",
    "用",
  ]);
});

test("whitespace inside non-dictionary text is still dropped", async () => {
  assert.deepEqual(segmenter.cut("第一段  第二段\t第三段"), [
    "第一",
    "段",
    "第二",
    "段",
    "第三",
    "段",
  ]);
});

test("can disable the bundled default dictionary", async () => {
  const withoutDefaultDictionary = await loadSegmenter({
    userDictionary: null,
  });
  assert.deepEqual(segmenter.cut("这是我的世界"), ["这是", "我", "的", "世界"]);
  assert.deepEqual(withoutDefaultDictionary.cut("这是我的世界"), [
    "这",
    "是",
    "我",
    "的",
    "世界",
  ]);
});

test("rejects non-string input", () => {
  assert.throws(() => segmenter.cut(null), /expects a string/u);
});
