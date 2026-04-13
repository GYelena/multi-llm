# 角色与训练数据约定（草案）

本文与当前代码对齐：`protocol.py`（DAG / 任务类型）、`planner.py`（OpenAIJsonPlanner 的 JSON 约束）、`experts.py`（A/B/C 的 system prompt 与 user 负载）。

---

## 1. 中心 Planner（DAG 规划）

### 职责（训模型时要固定住的）

- **只做一件事**：根据用户自然语言问题，输出 **无环依赖图** 上的任务节点列表。
- **不做**：直接回答用户最终问题、替专家写长文、编造执行结果。

### 模型侧约束（与 `OpenAIJsonPlanner._system_prompt` 一致）

- 输出 **唯一一个 JSON 对象**（可被 `json.loads` 解析；若模型夹带说明文字，编排器会尝试截取第一个 `{` 到最后一个 `}` 再解析）。
- 顶层字段：**`nodes`**（数组），每个元素形状如下（camelCase）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `nodeId` | string | 非空、全局唯一，如 `T1`、`T2` |
| `taskType` | string | `retrieve` \| `reason` \| `write` \| `verify` |
| `expert` | string | `A` \| `B` \| `C` |
| `dependencies` | string[] | 必须先完成的 `nodeId` |
| `inputRefs` | string[] | 本节点可读的产物节点 id（通常 ⊆ dependencies 的闭包） |
| `budget` | object | `maxTokens`、`maxSeconds` 正整数 |

### 校验规则（训数据时必须满足）

- `dependencies` 里出现的 id **必须**在 `nodes` 里存在。
- 图 **无环**（拓扑可执行）。
- 至少 **一个** 节点。

### 三域（数学 / 金融 / 常识）规划启发（给标注/写 prompt 用）

| 域 | 倾向 DAG 形态 | 备注 |
|----|----------------|------|
| **数学** | 常：`retrieve`(可选) → `reason` → `write`；证明/多步可加 `verify` | 简单口算可省 `retrieve`；复杂应用题可保留 A 做「已知条件整理」 |
| **金融** | 常：`retrieve` → `reason` → `write`；政策对比可加 `verify` | A 偏「事实/定义/口径」；B 做推理与风险提示；C 对用户可读表述 |
| **常识** | 常：`retrieve` → `reason` → `write` 或 `reason` → `write` | 强事实核查时保留 A；概念解释可缩短链 |

### Planner 训练样本「输入 / 输出」建议

- **输入**：用户 `query`（与线上一致）。
- **输出（gold）**：满足上表 schema 的 JSON 对象（即 `{"nodes":[...]}`）。
- 见 `data_train_examples/planner.example.jsonl`。

---

## 2. 专家 A / B / C（执行节点）

### 与代码一致的 system prompt（OpenAI 适配器）

- **A**：`You are Expert A (factual retrieval and grounding). Return concise evidence-oriented results.`
- **B**：`You are Expert B (reasoning). Return structured step-by-step reasoning.`
- **C**：`You are Expert C (writing). Produce clear and faithful final responses from context.`

### 单次调用时模型看到的 user 内容（JSON 字符串）

编排器发送的是 **一个 JSON 字符串**（`OpenAIExpertAdapter._build_user_content`），结构为：

```json
{
  "nodeId": "T2",
  "taskType": "reason",
  "query": "<用户原始问题>",
  "context": { "<上游 nodeId>": <该节点 payload 或空对象>, "...": "..." }
}
```

- `context` 的 key 来自已完成的 `inputRefs` 对应产物；失败节点可能对应 `{}`。
- **训专家模型**时：建议把 **同一套 system** + **上述 user JSON** 作为输入；**assistant** 为希望模型产出的正文（线上会放进 `choices[0].message.content`，再存 `payload.text`）。

### 职责边界（建议写进标注指南）

| 专家 | 主要产出取向 | 应避免 |
|------|----------------|--------|
| **A** | 短证据、要点、可引用陈述；标明不确定性 | 长篇最终用户答案、复杂多步推导独占 |
| **B** | 分步推理、检查条件、中间结论；可与 A 矛盾时说明假设 | 完全忽视 A 的 `context` |
| **C** | 面向用户的统稿；忠实于 `context`、少编造 | 引入 `context` 中不存在的新「事实」而不加说明 |

### 三域侧重点（给标注）

- **数学**：B 写清步骤与检验；C 写最终答案与单位；A 可整理题设。
- **金融**：A 给定义/口径/常见事实；B 给逻辑链与风险；C 给合规友好表述（非投资建议免责声明可由 C 统一处理若你有模板）。
- **常识**：A 给关键事实句；B 给「为何如此」的推理；C 给通俗总结。

### 专家训练样本文件

- 见 `data_train_examples/expert.example.jsonl`（字段说明见该目录 `README.txt`）。

---

## 3. 最小 JSONL Schema 汇总

### 3.1 Planner SFT / 偏好数据（文件级）

每行一个 JSON 对象，**必填**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 可选但强烈建议，便于去重与回归 |
| `domain` | string | 可选：`math` \| `finance` \| `common` |
| `query` | string | 用户问题（与线上一致） |
| `dag` | object | **等于** planner 应输出的 JSON：`{"nodes":[...]}` |

可选：`notes`（标注备注）、`source`（题库来源）。

### 3.2 专家 SFT（文件级）

每行一个 JSON 对象，**必填**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 可选 |
| `domain` | string | 可选 |
| `expert` | string | `A` \| `B` \| `C` |
| `nodeId` | string | 与 DAG 中一致，如 `T2` |
| `taskType` | string | `retrieve` \| `reason` \| `write` \| `verify` |
| `query` | string | 用户原始问题 |
| `context` | object | 与线上一致：nodeId → 上游 payload（可简化为要点） |
| `assistant` | string | 希望模型输出的正文（映射到线上 assistant `content`） |

可选：`system_override`（仅当你要做角色消融实验时覆盖默认 system）。

---

## 4. 与线上一致性的自检清单（训前）

1. **Planner**：每条 gold `dag` 能否被 `parse_dag_plan` 接受（无重复 id、依赖存在、无环）。
2. **专家**：`expert`+`taskType` 与 DAG 中该节点一致；`context` 的 key 与 `inputRefs` 对齐。
3. **域标签**：`domain` 若填写，建议在 val 中分层抽样，避免泄漏。

---

## 5. 建议的目录布局（你可自建，不强制进仓库）

```
data/
  math/
    planner.train.jsonl
    planner.val.jsonl
    experts.train.jsonl
  finance/
    ...
  common/
    ...
```

---

## 6. 修订记录

- 草案：与 `muti-llm/orchestrator` 当前实现同步；后续若改 `TaskType` / planner schema，请同步改本文与 `data_train_examples/`。
