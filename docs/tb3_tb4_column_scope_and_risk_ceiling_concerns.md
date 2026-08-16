# MAEDA 侧对附录 P 两项未决问题的顾虑

**背景**:`ECOSYSTEM_INTEGRATION_PLAN.md` 附录 P(协调侧起草的 TB3+TB4 候选 schema)里，多数字段标了 `[答]`/`[选]` 并给出候选理由，MAEDA 侧审阅后基本认可。但有两项——`column_scope.mode` 和 `force_on` 是否要能突破 `needs_review`——分量较重、影响面较大，MAEDA 不想单方面拍板，写成本文档交给 orchestrator 判断。

风险分级映射表归属（P.4.1）MAEDA 已经确认同意草案的建议（记在 cleaner 仓库，MAEDA 只消费结果），不在本文档讨论范围。

---

## 1. `column_scope.mode`：`restrict` 还是 `advisory`

### 1.1 两个选项的字面定义（附录 P.2）

- `restrict`：cleaner 只清洗 `column_scope.columns` 里的列，其余列不动
- `advisory`：优先处理这些列，但不排除其他列

### 1.2 MAEDA 的顾虑：`restrict` 的失败模式是静默的

`column_scope.columns` 来自 P.1 的确定性列名对账（精确匹配 → 大小写不敏感 → 词表别名匹配，纯字符串匹配，不调 LLM）。这套对账**注定不是 100% 召回**——用户问题里提到的业务术语如果既不精确匹配列名、也不在词表别名里（词表本身是 O.3.2 点名"从零设计"，覆盖率未知；仓库里目前完全没有词表文件），会落进 `unresolved_mentions`，不会进入 `column_scope.columns`。

在 `restrict` 语义下，这意味着：**如果对账漏判了一个用户实际关心、且数据本身很脏的列，这一列会完全不被清洗，而且没有任何环节会报告这件事**——`resolution_status` 可能仍是 `full`（如果被漏掉的 mention 本身没有被解析出 unresolved 记录，而是压根没被 LLM 从问题里提取出来，属于 intent parsing 阶段就丢失，不是 P.1 对账阶段的 `unresolved`），或者即使 `resolution_status=partial`、`unresolved_mentions` 里有记录，下游是否真的把这条信息转成用户可见的 caveat，取决于 P.5 的 O.4.1（Insight 结构化 caveat 字段）**这一项本身也还没定案**——两个未决项叠在一起，`restrict` 的风险会被放大。

`advisory` 的代价是"省不下多少 token/范围"，这是已知的、可衡量的成本；`restrict` 的代价是"未知覆盖率的对账质量直接决定数据是否被处理"，这是一个更难在上线前评估的风险。按 MAEDA 项目现有的保守倾向（附录 B.1 的设计原文："v1 宁可不触发，也不做无依据的有损修改"），MAEDA 倾向 `advisory`。

### 1.3 但这不是没有代价的选择，供 orchestrator 权衡

- `advisory` 会让 TB3 想要的"用 intent 收窄清洗范围以省 token/避免动不该动的列"这个初衷打折扣——如果 cleaner 侧后续的 deep profiling（阶段 3 待实现的"预算受控、类型感知"画像）本身就是 token 敏感的，`advisory` 等于没有真正收窄输入给 LLMPlanner 的画像范围
- `restrict` 的风险可以被"先把词表覆盖率做扎实、上线前用真实数据集测对账召回率"缓解，不是无法接受，只是意味着 P.1 的对账质量本身需要一个验收标准（目前草案没有给出——比如"对账召回率 ≥ 多少才允许上 restrict"），这个验收标准怎么定也是一个待展开的问题
- 折中方案（本文档提出，非既有草案条目）：`resolution_status=full` 时用 `restrict`，`partial`/`absent` 时自动退化为 `advisory`——用"对账是否完整"这个已有信号做门控，而不是全局二选一。这样至少把"漏判导致静默不处理"的风险限制在"对账本身就报告不完整"的那些情况，配合 O.4.1 的 caveat 字段可以让用户至少看到"部分列未收窄清洗范围"的提示。这个折中方案是否可行，需要 orchestrator/cleaner 侧确认 cleaner 能否接受一个随请求变化的 scope 语义，而不是固定配置。

**MAEDA 的立场**：不反对 `restrict`，但反对在没有对账质量验收标准、且 caveat 字段（O.4.1）还未定案的情况下就定 `restrict`。如果 orchestrator 判断这两个前提可以后补（先冻结 `restrict`，验收标准和 caveat 字段作为阶段 3 的后续任务），MAEDA 也可以接受，只是想确保这不是被默认忽略的风险，而是一个被看到、被权衡过的决定。

---

## 2. `force_on` 是否要能突破 `needs_review`；`MAEDA_CLEANER_RISK_CEILING` 现在加还是以后加

### 2.1 附录 P.5(O.4.2)的候选立场

草案建议 `force_on` 严格限定为"强制调用 cleaner"，不覆盖 `needs_review` 的阻断；如果确实需要"强制执行到底"的实验档，另设 `MAEDA_CLEANER_RISK_CEILING`（`safe`/`lossy`/`all`），与三态开关正交。MAEDA 认可"`force_on` 不应默认突破安全边界"这个方向，顾虑只在"这个新开关要不要现在就加"。

### 2.2 MAEDA 的初始顾虑：不要为假设性需求预先设计

MAEDA 项目的 `CLAUDE.md` 明确写："Don't add features...for hypothetical future requirements." 按这个原则，`MAEDA_CLEANER_RISK_CEILING` 在没有具体消费方之前不应该加——这是 MAEDA 提出"先不加"的初始理由。

### 2.3 但这条原则可能不适用于这个具体场景，orchestrator 需要判断

重新看了一遍阶段 4 的方法论要求后，MAEDA 认为这个初始判断可能站不住：阶段 4 的"组件净增益"实验（定案文档"两类实验必须分开做"那一条）要求 `force_on` vs `force_off` 在**同一个 case、同一个输入、配对 trial**下对比。如果某个 trial 的清洗过程中触发了 `needs_review`（比如高风险操作被拦下），`force_on` 分支这一轮实际执行的清洗步骤会比另一些没触发 `needs_review` 的 trial 少——**这不是"cleaner 有没有用"的净增益信号，是"这条数据这次有没有撞上风险分级阈值"的信号，两者混在一份 pass@k/方差统计里会污染结果**，而这正是定案 #15 当初被提出、要把"组件增益"和"路由准确性"两类实验分开做的同一类问题的又一个变种：`needs_review` 造成的执行差异如果不被识别和控制，实验对照就不干净。

`MAEDA_CLEANER_RISK_CEILING`（或类似机制）如果要等到"阶段 4 真的跑起来发现这个问题"才补，届时可能已经产生了一批需要作废重跑的实验数据。这不是纯粹的假设性需求——它是阶段 4 方法论明确要求的"配对 trial 必须可比"这条约束，在 TB4 引入 `needs_review` 分支后自然推出的推论，只是眼下阶段 4 还没开始，还没有真正撞上。

### 2.4 MAEDA 的立场

MAEDA 没有把握判断这条推论是否成立——阶段 4 的实验设计细节不在 MAEDA 这一轮的核实范围内，"是否需要现在就加这个开关"取决于阶段 4 的 A/B 方法论具体怎么处理 `needs_review` 造成的执行差异，而这件事 orchestrator 或负责阶段 4 设计的一方可能有更完整的视角。MAEDA 的建议是：**如果阶段 4 的方法论已经有别的机制处理"trial 因 needs_review 而执行不完整"这个问题（比如按 E3 的终止状态分类、把这类 trial 标记为不适用而非计入净增益统计），那么 `MAEDA_CLEANER_RISK_CEILING` 现在确实不必加，用回归 E3 的既有机制即可；如果没有，这个开关最好现在就定下接口形状（哪怕暂不实现），避免阶段 4 开工后被迫回头改契约。**

---

## 3. 给 orchestrator 的两个具体问题

1. `column_scope.mode`：接受 MAEDA 提出的"`full` 用 restrict、`partial`/`absent` 退化为 advisory"折中方案，还是维持草案的全局单值配置（如果维持单值，选哪个）？
2. `MAEDA_CLEANER_RISK_CEILING`：阶段 4 的 A/B 方法论是否已经用 E3 的终止状态分类处理了"`needs_review` 导致的 trial 执行差异"这个问题？如果是，这个开关可以按原计划推迟；如果不是，建议现在就把接口形状定下来（哪怕暂不实现）。
