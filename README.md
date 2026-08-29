# PKUSEG-js

A pure-JavaScript inference engine for PKUSEG Chinese word segmentation and
POS tagging.

PKUSEG `0.0.25` 的纯 JavaScript 推理引擎，提供中文分词与词性标注。运行时不
依赖 Python、NumPy、Cython 或任何原生扩展，Node.js 18+ 与现代浏览器均可使用。

这是一个**推理引擎**，不是完整的 pkuseg 移植：

- **已实现**：默认模型中文分词、词性标注及用户词性覆盖、用户词典、原版默认
  词典后处理、自定义/领域模型加载、Python 模型到 JS 格式的离线转换；
- **没有移植**：模型训练。推荐继续使用 Python 版训练，再用本项目的工具导出
  为 JS 推理模型。

## 安装

尚未发布到 npm，可暂时直接从 GitHub 安装：

```bash
npm install github:Einheriar/PKUSEG-js
```

要求 Node.js 18 或更高版本。模型文件不随仓库分发，见[获取模型](#获取模型)。

## 获取模型

仓库只包含代码与转换工具。预转换的默认分词模型（约 156 MiB）与词性模型
（约 51 MiB）通过 [GitHub Releases](https://github.com/Einheriar/PKUSEG-js/releases)
分发：下载模型压缩包，解压到仓库或包的根目录，得到 `models/default/` 与
`models/postag/` 两个目录，`loadSegmenter` 的默认路径即可直接工作；也可以
解压到任意位置，通过 `modelDirectory` 指定。

同样可以用[自带工具](#使用其他领域模型)从 Python 版模型自行转换。

## 快速开始

先把模型解压到 `models/` 下（见[获取模型](#获取模型)），然后可直接运行示例：

```bash
node example/basic.js
```

```js
import { loadSegmenter } from "pkuseg-js";

const segmenter = await loadSegmenter();
console.log(segmenter.cut("我爱北京天安门"));
// [ '我', '爱', '北京', '天安门' ]

const withPostag = await loadSegmenter({ postag: true });
console.log(withPostag.cut("我爱北京天安门"));
// [ [ '我', 'r' ], [ '爱', 'v' ], [ '北京', 'ns' ], [ '天安门', 'ns' ] ]
```

## API

### `loadSegmenter(options)`（Node.js）

返回 `Promise<Segmenter>`。

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `modelDirectory` | `models/default/` | 包含 `model.json` 与 `model.bin` 的目录 |
| `userDictionary` | `"default"` | 见[用户词典](#用户词典) |
| `postag` | `false` | 是否同时加载词性模型 |
| `postagModelDirectory` | `models/postag/` | 词性模型目录 |

`segmenter.cut(text)`：未开启 `postag` 时返回 `string[]`；开启后返回
`[词, 词性][]`。

### `loadBrowserSegmenter(options)`（浏览器）

从 `pkuseg-js/browser` 导入，该入口不导入任何 Node.js 模块：

```js
import { loadBrowserSegmenter } from "pkuseg-js/browser";

const segmenter = await loadBrowserSegmenter({
  modelJsonUrl: new URL("./models/default/model.json", import.meta.url),
  postagModelJsonUrl: new URL("./models/postag/model.json", import.meta.url),
});
```

- `modelJsonUrl` 必填；`postagModelJsonUrl` 可选；
- `userDictionary` 仅支持 `"default"`、`null` 或数组，不支持文件路径；
- 默认二进制模型约 156 MiB，词性模型约 51 MiB。生产网页应在 Web Worker 中
  加载，并使用 HTTP 缓存或 IndexedDB 缓存模型；低内存移动设备不建议同时
  加载两个模型。

## 用户词典

`userDictionary` 有四种取值：

- `"default"`（默认）：启用原版自带的默认词典后处理；
- 数组：在默认词典后处理的基础上追加词条，支持 `"词"` 与 `["词", "词性"]`
  两种形式；
- UTF-8 文件路径（仅 Node.js）：一行一个词，词与可选词性之间用制表符分隔；
- `null`：完全关闭默认词典后处理。

```js
const segmenter = await loadSegmenter({
  userDictionary: ["自然语言处理", ["北京天安门", "custom"]],
  postag: true,
});
```

## 使用其他领域模型

仓库只附带默认分词模型与词性模型。原版 pkuseg 的医药、旅游等领域模型需要
自行转换。转换工具只需 Python 3 与 numpy（直接读取模型文件，不需要安装
pkuseg）：

```bash
pip install numpy

python tools/export_model.py \
  --model-dir <pkuseg模型目录> \
  --dictionary <pkuseg词典.pkl> \
  --output models/<目标目录>
```

- `--model-dir` 指向包含 `features.pkl` 与 `weights.npz` 的目录。pip 安装的
  pkuseg 位于 `site-packages/pkuseg/models/`，上游仓库亦有同样结构；
- `--dictionary` 可重复传入，把领域词典与默认词典一起写入推理模型；
- 词性模型会根据 `features.pkl` 的结构自动识别，无需指定 `--task`。

转换工具会执行 Pickle 反序列化，只能用于可信来源的模型文件，不要转换来源
不明的 Pickle。

## 模型格式

原版 `features.pkl` 展开约 235 万条特征和 514 万条训练 bigram，Python 首次
加载峰值约 1.35 GiB。导出器会：

1. 删除推理路径完全不使用的 bigram；
2. 把特征、unigram 和词典转换为双 32 位开放寻址哈希表；
3. 保持原版 Float64 权重，不做有损量化；
4. 在 `model.json` 中记录二进制 SHA-256 与跨语言哈希测试向量，供加载时校验。

JS 推理保持与原模型相同的数值路径：`test/cases.json` 与原项目 `example.txt`
的所有输入在 Python 与 JavaScript 两边的分词及词性输出逐项一致，同时把默认
模型运行峰值降到约 210 MiB。

## 测试与性能

```bash
npm test               # 分词、词性、用户词典、Unicode 边界等单元测试
npm run verify-models  # 校验附带模型的格式与 SHA-256
npm run benchmark      # 本机基准
```

参考性能（Node.js 22 的一次基准）：默认模型加载约 90–100 ms，常驻内存约
212 MiB，吞吐约 2.8 万字符/秒。具体数值随硬件与磁盘缓存状态变化。

## 致谢与许可

本项目基于北京大学
[pkuseg-python](https://github.com/lancopku/pkuseg-python)，代码沿用 MIT
许可证。原始 PKUSEG 预训练模型使用的数据可能有额外的研究用途限制；将模型
发布到 npm、CDN 或用于商业产品前，应单独确认模型与训练数据的再分发/使用
权限。
