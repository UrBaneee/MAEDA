# Handoff: MAEDA → Data Cleaner / RAG-MCP-Server

**发出方**：MAEDA
**接收方**：`agentic-data-cleaner-v2`、`rag-framework`（同一人的两个子系统项目）
**日期**：2026-08-12
**修订**：2026-08-14——三方协商 + 外部评审（codex）后定案，第 2、3.1、4、6 节据此更新；初版第 2 节的 Data Cleaner 契约与已核实的服务端源码不符，已重写。
**权威文件**：`~/ECOSYSTEM_INTEGRATION_PLAN.md`（v3）**是执行顺序与分工的唯一权威**，本文档与之冲突处以它为准。本文档的定位是 MAEDA 侧的背景论证与消费端契约细节，不重复论证已定的事。
**用途**：今天在 MAEDA 上做了一轮 grill-me（架构合理性拷打），本文档把和这两个子系统直接相关的结论抽出来，供三份文档（本文档 + 两个子系统各自已有的 handoff）对齐后开一次联调会议。

**先读什么**：三份文档已经存在部分重叠但视角不同，建议顺序：
1. 本文档（MAEDA 视角：今天定了什么、MCP 契约的权威定义）
2. `~/agentic-data-cleaner-v2/HANDOFF_TO_MAEDA.md`（Data Cleaner 视角：端口/参数名不一致的具体证据）
3. `~/rag-framework/HANDOFF_MAEDA_INTEGRATION.md`（RAG 视角：策展层/检索层分工、事件脚本方案、注入点前移提案——这份最长最重）

**另需知晓**：`docs/anthropic_agent_eval_improvement_plan.md` 是 MAEDA 侧一份既有的 eval harness 改造方案（基于 Anthropic《Demystifying evals for AI agents》）。它与本次集成的交集很大——**两个子系统的 A/B 评测会跑在改造后的 harness 上，而不是现在这个版本**。凡涉及评测设计的条目（6.4、6.5），本文档已按该方案修正，并标注了对应节号（写作「eval 方案 §N」）。

---

## 1. 今天 MAEDA 侧 grill-me 的结论（六项待办，按依赖顺序）

评判标准定为：AI Eval / GenAI DS 岗面试官，能扛住三层追问。核心决定：

1. **找第二标注员**——eval 的人工标注目前是单标注员（作者本人），需要一份 inter-annotator 一致性数据，公开写成 limitation。与两个子系统无关，异步进行。
2. **把 MCP 连接做实**——本文档主题，见下文。
3. **Analysis Agent 改造为真 tool-calling**——现状是 plan-then-execute（LLM 一次性写完计划，Python 代码机械执行），要改成 LLM 看到工具结果后自主决定下一步。外层图公开改口叫 "workflow" 而非 "multi-agent"。与两个子系统无直接接口影响，但改造后的 Analysis Agent 如果自主决定要不要调用 Data Cleaner/RAG，接口形状不变，只是调用时机从"图节点固定触发"变成"LLM 决定触发"——**这点需要在会上确认两个子系统的工具是否要以 tool-calling 形式暴露给 Analysis Agent，还是继续只在固定的图节点（`connect_and_profile_data` / `retrieve_domain_knowledge`）里调用**。倾向后者：数据质量剖析和知识检索仍然是确定性该发生的步骤，不需要 LLM 自己决定"要不要查数据质量"。
4. **改造后重跑 eval v2**——现有 answer key/agreement 数字冻结为 v1，人工标注继续作为 judge 校准集复用（但按 eval 方案 §9.2，需定期补新样本重新校准，不永久依赖这一批）。**前提**：必须先有多 trial 支持，见 6.4 D0。
5. **主张-代码对齐审计**——包括本文档第 5 节列的 README 失实描述。
6. **README 重构收尾**——单一版本，并列 "Evaluation Methodology" 和 "Orchestration Design" 两个深潜区块。

**与本次会议直接相关的是第 2 项**。第 1、3、4、5、6 项列出是为了让子系统方知道 MAEDA 侧同期还在动什么，避免联调会上对 MAEDA 现状有过时假设。

MAEDA 侧的完整待办（含从两份子系统文档接下来的条目）见第 6 节。

---

## 2. MCP 契约：MAEDA 消费端的权威定义

以下契约逐字段标出哪些是"错了会静默改变行为"的硬约束。RAG 部分仍以 MAEDA 消费端为验收基准；**Data Cleaner 部分已按协商定案重写**——初版从 MAEDA 客户端代码反推的字段与 cleaner 服务端真实签名不符，且"子系统对照本契约验收、不让 MAEDA 迁就"这个立场对这些字段是反的：协商已定这些字段由 **MAEDA 迁就服务端**（改 `dataset_path`、去掉 `plan` 回传），cleaner 侧唯一的接口让步是新增 `has_critical_issues: bool` 顶层字段。

### 传输层
- MCP streamable-http，官方 `mcp` SDK 的 `ClientSession`，每次调用走完整 `initialize` 握手。FastMCP 起 streamable-http 天然满足；裸 JSON-RPC 会挂（`src/mcp_client/client.py` 顶部注释记录过 406/400 的教训）。
- 返回值走 `structuredContent`（dict/Pydantic 模型直接返回即可），退路是首个 text block 里的 JSON 字符串。
- 健康检查 5 秒超时做 `initialize + list_tools`，超时即判不可用、全程 fallback。

### Data Cleaner —— 参数平铺（协商 + 评审定稿）
正常路径**只调三个工具**：`profile_dataset {dataset_path}` → `clean_dataset {dataset_path, planner_mode, max_rounds=1}` → `validate_quality {dataset_path}`

与初版的四处差异及理由：

- **参数名是 `dataset_path` 不是 `path`**：cleaner 服务端四个工具签名（`mcp_app.py:105/176/302/466`）的必填参数全部是 `dataset_path`。MAEDA 现发的 `{"path": ...}` 会被 FastMCP 参数校验直接拒绝，不是静默兼容。MAEDA 迁就，改 `src/mcp_client/data_cleaner.py`。
- **彻底不调 `get_cleaning_plan`**（评审定案，比"保留用于展示"更进一步）：MAEDA 侧 `CleaningPlan.to_dict()` 的结构与服务端 `validate_plan()` 要求的内部结构（顶层 `intent`，每 step 含 `step_id`/`mcp_tool`/`inputs`/`params`/`effects`/`guards`，`mcp_tool` 限白名单）完全不是一回事，怎么序列化都过不了校验；而保留它"仅用于展示"会造成**双重规划**——两个独立生成的计划、额外 LLM 成本与延迟，且展示/decision trace 里的计划与实际执行的不是同一个，违背决策追踪的真实性。定案：执行计划取自 `clean_dataset` 返回值的 `plan_steps`（cleaner 侧 C3 将该字段升为稳定契约）。`get_cleaning_plan` 仅在未来加入人工确认环节时，配合 cleaner 新增的 `execute_plan(plan_id/plan_path)` 使用。
- **`planner_mode="llm"` 是 MAEDA 集成路径的硬依赖**：intent-driven 清洗必须 LLM planner；cleaner 运行环境需配置 `ANTHROPIC_API_KEY`。**经 Pydantic Settings 配置**（如 `DATA_CLEANER_PLANNER_MODE`）按环境选择，**不在调用点或脚本里散落硬编码**；启动时校验选 `llm` 时 API key 与服务能力存在，否则在进入耗时流程前明确失败。分阶段：连通性冒烟配 `rule`，intent 路径切 `llm`。
- **`max_rounds` 固定传 1**：外层循环（最多 3 轮）由 MAEDA 图层控制，不交给服务端内部多轮。

硬约束：
- 触发清洗回环的判断改读 `profile_dataset` 返回的**顶层 `has_critical_issues: bool`**（cleaner 新增字段）。初版写的"`quality_issues[].severity == "critical"`"是错的：cleaner 真实返回的 `quality_issues` 是**字符串列表**（`"missing_values"`、`"possible_duplicates"` 等），MAEDA `models.py:43` 的 `.get("severity")` 在真实数据下会直接抛 `AttributeError`。⚠️ **但不能因此把 `quality_issues` 简单收窄为 `list[str]`**——MAEDA 自己的 pandas fallback（`src/mcp_client/fallback.py:264` 起）产出的是结构化 dict（`column`/`issue`/`severity`/`detail`），Insight Agent 也按 dict 渲染，收窄会让 MCP 与 fallback 两条路径语义分裂。定案：在 MCP 边界归一化成统一模型 `QualityIssue{code, severity?, column?, detail?, source: cleaner|fallback}`，cleaner 的字符串映射到 `code`，fallback 的 dict 保留字段。
- **`has_critical_issues` 在阶段 1 只是临时 v1 语义**（仅结构性问题：缺失率/重复/异常值），**TB4 冻结最终语义前不算稳定契约**；它仅作**清洗前触发条件**，清洗后的退出条件用 `validate_quality.passed`。
- **cleaner 的错误响应必须在解析前检查**：cleaner 内部异常返回 `{"error": true, "error_type", "message"}` 且**协议层 `isError=False`**。不检查就会把失败解析成 `row_count=0, columns=[]`，看起来像"空数据集/不需要清洗"。四个工具统一在解析前检查。
- `clean_dataset` 返回的 `cleaned_path` 必须是 MAEDA 进程能直接读到的文件路径。⚠️ **"路径不同"不等于"清洗成功"**：成功条件是**路径不同 + 内容确有变化 + validation 通过**三者共同构成（见 6.2 M8）。
- `estimated_impact`/`changes_summary` 的类型要**以一次真实服务响应或工具 JSON Schema 验证后定型**，不只依据 handoff 文字。

### 同机部署假设（显式声明）
`cleaned_path` 的可读性硬约束、以及未来人工确认路径上的 `plan_path`，全部默认 **MAEDA 与 cleaner 共享同一文件系统**（当前本地联调即如此）。跨容器/跨主机部署前，这部分契约必须重新设计（届时评估 artifact URI / 对象存储，不把本地绝对路径固化为公共契约），不能沿用。

### RAG-MCP-Server —— 参数包在 `input` 下
`retrieve` / `retrieve_with_metadata` 发送 `{"input": {"query", "top_k", "collection"?}}`；`list_collections {}`。

硬约束：
- 顶层必须有 `chunks` 键。
- `retrieve_with_metadata` 的每个 chunk 必须填 `source_file`，否则该 chunk 会被引用列表直接丢弃（`nodes.py:231-232`），insight 的引用溯源演示会是空的。
- **返回值的 `error` 字段必须检查**（与 cleaner 同理）：rag 服务端内部失败时返回 `error` + 空 `chunks` 且协议层 `isError=False`，不检查则真实检索故障与"合法零命中"无法区分——**静默错误不是 fallback**。这是迁移期双轨：rag 侧 R3 正式修复（内部故障改抛 tool error + 稳定错误码）落地后，MAEDA 再移除此检查。

---

## 3. 与两个子系统已有 handoff 的对照：哪些已对齐，哪些需要会上定

### 3.1 Data Cleaner —— 两处不一致均已定案

`agentic-data-cleaner-v2/HANDOFF_TO_MAEDA.md` 独立核对过一遍契约，发现的两个具体问题现均有结论：

- **端口**：**已定**。cleaner 显式起在 **8001**、rag-framework 起在 **8002**（rag 侧自行改 docker-compose），MAEDA `settings.py` 默认值（8001/8002）**零改动**。
- **参数名**：**已核实，不是"疑似"**。结论：MAEDA 发的 `{"path": ...}` 会被 FastMCP 参数校验直接拒绝（缺必填 `dataset_path`），返回 `isError=True` → MAEDA 侧抛 `MCPToolError` → `fallback.py` 只捕获 `MCPConnectionError`，**结果是整条 pipeline 崩溃**——不是 422、也不是优雅降级。修复见第 2 节定稿契约（改 `dataset_path`）与 6.2 B3（补 `MCPToolError` 捕获）。

### 3.2 RAG-MCP-Server —— 已有一份详尽的独立分析，需要 MAEDA 做四项回应

`rag-framework/HANDOFF_MAEDA_INTEGRATION.md` 篇幅很长，核心是提出"策展层（口径解释，归 MAEDA，确定性注入）vs 检索层（业务事件，归 rag-framework，检索）"两层结构，并指出当前检索路径退化成纯 BM25（`_run_retrieval` 里 `embedding_provider` 没传，向量检索和重排器都没启用）。

该文档第 8 节列了四项需要 MAEDA 拍板的事，逐条给出我的初步判断供会上讨论：

- **8.1 连通性**（阻塞项，其余全部依赖它）——直接对应本文档第 2 项待办，会上先解决。
- **8.2 RAG 注入点是否前移到 planner 阶段**——该文档建议先做完 5.1 的口径词表，再评估是否还有必要前移（口径词表能覆盖的部分，前移检索的边际价值会下降）。同意这个先后顺序，**本次联调先不做前移**，留到词表做完后再看。
- **8.3 检索查询构造方式**（当前是纯字符串拼接 vs 改用 LLM 生成检索问句）——建议按该文档提议，4.2 的检索层评测同时跑两种构造方式用数据决定，不在会上纯靠讨论定。
- **8.4 事件脚本归属**——这是双方共享的评测 ground truth 产物（不是语料，是语料和数据的上游规格），且有严格的"必须先写完、两边各自独立渲染"的硬约束（防止先看数据再编语料的自证问题）。**归属已定**：放 rag-framework 仓库，YAML 格式 + `version` 字段 + 内容哈希；MAEDA 只读引用，消费前校验哈希，不一致即报错而非静默用旧版本。

此外该文档第 5 节列了五项 MAEDA 侧要做的具体工作（口径词表存储方案、未知数据源降级处理、`generate_demo_data.py` 改造、`embedding_provider` 传参、端到端 A/B 评测），已经给出了选定方案（YAML + 按 schema 过滤注入，附三态漂移检测），采纳，不在此重复。

### 3.3 rag-framework 顺带发现的两个 MAEDA/Cleaner 缺陷（与本次集成无关，但影响评测可信度）

`rag-framework` 文档第 6 节报告了两处和 RAG 集成无关、但会污染未来 A/B 评测结果的既有 bug，供子系统会议之外单独排期修：

- `src/tools/data_connector.py:209`——多表数据源未指定表名时硬编码取 `tables[0]`，不参照用户问题选表。demo 数据库 `ecommerce_orders.db` 是多表的，若评测跑用错表，归因准确率的信号会被这个噪声污染，需要在跑 8.1 之后的端到端评测之前修好或至少确认每次选中的表。
- `src/agents/intent_parser.py`——`parse_intent`（图第 1 步）读取的 `schema_summary` 在该步执行时恒为空串（真正赋值发生在第 2 步），任何依赖 schema 判断"revenue 指哪一列"之类的逻辑从未拿到过真实数据。纯时序问题，不涉及 RAG，MAEDA 自行判断是否后移。

---

## 4. 建议的会议议程（依赖关系决定顺序）

阶段与 TB 编号以 `ECOSYSTEM_INTEGRATION_PLAN.md` v3 为准，此处只记 MAEDA 视角的要点。

1. **★ TB0 传输冒烟**（轻量，不要求端到端）：端口与参数名均已定案（见 3.1），只剩执行——cleaner 起 8001、rag 起 8002（MAEDA 配置不动）。验收是 `initialize` + `list_tools` + **校验必需工具存在** + **每个必需工具用最小合法参数真调一次**。⚠️ 健康检查成功只证明**可连接**，不证明**契约兼容**，所以工具级业务错误在 TB0 可接受，契约正确性归 TB1。**TB0 不过，阶段 1 不开始。**
2. **★ TB1 契约 E2E**（strict 模式）：阶段 1 的契约修复（M1–M8）完成后才做。验收：无未捕获异常、无非预期 fallback、`mcp_call_log` 里两个子系统均 `mode="mcp"`、**清洗回环真实收敛**（清洗后文件被下一轮消费、re-profile 报告更新、validate 通过后退出）、rag trace 证明向量候选与 reranker 真实参与。
   > 拆 TB0/TB1 是为了消除初版计划的自相矛盾：原 TB1 既要求"无未捕获异常"，又把 `has_critical_issues` 崩溃列为预期行为；而修复它的 M2 在阶段 1、却又规定 TB1 不过不进阶段 1。
3. **★ TB4 清洗质量语义闭环**（MAEDA + cleaner，**已提前为阶段 2 首项**）：把 C1 的临时 v1 语义冻结为正式契约。议题：`has_critical_issues` 的准确含义——**全表平均指标的稀释问题**（分析必需列严重缺失可能被大量干净列稀释）、结构性问题与 intent-specific 可用性问题如何组合、**intent 缺失时的基线判定**；哪些问题触发自动清洗 / 哪些只产生 caveat / 哪些必须人工确认；`validate_quality` 改造为面向 intent 的可用性校验与上述闭环同时设计；**与 M7 联动的多轮停止规则**。⚠️ 这两者必须联合设计——**单方改会破坏 router 行为**。
4. **★ TB3 intent payload schema**（三方）：cleaner 的 intent-driven 清洗需要列语义/口径在**清洗前**（图第 [2] 步）可用；按 rag 文档两层原则这是策展层内容，确定性注入、不走检索。解法是把词表按-schema-过滤的结果注入第 [2] 步传给 cleaner 的 intent payload（连带影响见 6.3）。需定 intent 字段名/结构/版本字段、词表条目形态、LLMPlanner 如何消费、intent 引用不存在列/重名列/大小写不匹配列时的行为。
5. **★ TB5 事件脚本评审与冻结**（三方）：归属已定（rag-framework 仓库），会上评审初稿、冻结、打 tag、生成 sidecar manifest（SHA-256）。硬约束不变：脚本冻结前不许生成数据与语料，两边独立渲染互不参照。
6. 会后并行：MAEDA 做口径词表 + RAG 条件路由 + D0/E4 + E1/E2/E3；cleaner 做 intent-driven cleaning；rag-framework 生成脏语料——三边基本可独立推进。
7. **端到端 A/B 评测**排在最后（阶段 4），前提 TB1/TB2 通过、TB5 产物渲染完成、D0+E4 完成、E1/E2 修复。

---

## 5. 顺带发现：README 当前的一处失实描述

`~/MAEDA/README.md:45` 现在写着"Currently exercised 0% of the time"。这句话是本次联调要修正的目标状态本身，此处只是标记它是一处需要在集成做实后回来更新的具体位置，不是新发现——两个子系统的 handoff 文档都已经引用过这句话作为触发本次整个联调的问题陈述。

---

## 6. MAEDA 侧完整待办清单

本节把三份文档里所有落在 MAEDA 身上的工作合并成一张表，供子系统方核对"我等的那件事 MAEDA 排在第几"。来源列标明该条来自今天的 grill-me、还是两份子系统文档的哪一节。

**所有行号均已对着当前代码核实过**（子系统文档里有两处行号已漂移或描述不准，见 6.5）。

### 6.1 阻塞项：连通性（必须最先做）

| # | 事项 | 位置 | 来源 |
|---|---|---|---|
| A1 | 端口对齐：**已定**——cleaner 显式起在 `8001`，MAEDA 默认值零改动 | `src/config/settings.py:34-35` | 协商定案 |
| A2 | 参数名：**已定**——MAEDA 客户端改发 `dataset_path`（结论已有：`{"path"}` 被 FastMCP 校验拒绝 → `MCPToolError` → pipeline 崩溃，见 3.1，无需再实测） | `src/mcp_client/data_cleaner.py` | 协商定案 |
| A3 | RAG 服务端口：**已定**——rag-framework 自行改 docker-compose 起在 `8002`，MAEDA 默认值零改动 | `src/config/settings.py:37-38` | 协商定案 |
| A4 | **TB0 传输冒烟**：新增 `scripts/check_ecosystem.py`，对两个子系统做 `initialize`+`list_tools`+校验必需工具存在+每个必需工具用最小合法参数真调一次 | `scripts/check_ecosystem.py`（新增） | 评审新增 |
| A5 | **TB1 契约 E2E**：阶段 1 的 M1–M8 完成后，strict 模式跑完整流程，确认 `mode="mcp"`、无未捕获异常、清洗回环真实收敛 | — | 评审新增 |

A1–A3 已定案：**子系统改启动命令，MAEDA 侧配置零改动**。

**A4/A5 是初版 A4 拆开的结果**——初版把"能连上"和"契约兼容"混为一谈，导致计划自相矛盾（既要求无未捕获异常，又把 `has_critical_issues` 崩溃当预期）。**健康检查成功只证明可连接，不证明契约兼容**：A4 只验传输与工具存在性（工具级业务错误可接受），A5 才验契约。A4 不过不进阶段 1；A5 在阶段 1 的接口修复全部完成后才做，且**必须在 strict 模式下通过**。

### 6.2 接口层改动

| # | 事项 | 位置 | 来源 |
|---|---|---|---|
| B1 | 传 `embedding_provider`，否则检索退化为纯 BM25（向量检索和重排器都不启用） | `src/mcp_client/rag_server.py:43,64` | rag 文档 5.4 |
| B2 | 若注入点前移（8.2），`rag_collection` 单值 `Optional[str]` 不够用，需改成按用途分别配置 | `src/config/settings.py:40` | rag 文档 8.2 |
| B3（=M1） | **strict/degraded 两档运行模式**，不再无条件 fallback。`MCP_STRICT_MODE` 环境变量：strict（联调/CI，契约/schema/认证类工具错误**直接失败**）、degraded（demo，仅连接与瞬时错误 fallback）。日志带结构化 `error_class`/`recoverable`/`service_reachable` | `src/mcp_client/fallback.py` | 评审定案 |
| B4（=M5） | 检查**两个子系统**的错误响应：RAG 返回检查 `error` 字段；**cleaner 四个工具检查 `error: true`**（初版只写了 RAG，是我方遗漏）。两者协议层都是 `isError=False`，不检查则 cleaner 失败会被解析成 `row_count=0, columns=[]`（看起来像"空数据集/不需要清洗"），rag 失败与"合法零命中"无法区分。命中即抛 `MCPToolError`，进入 B3 的分级处理并留痕 | `src/mcp_client/rag_server.py`、`data_cleaner.py` | 评审扩充 |
| B5（=M2） | `quality_issues` **边界归一化，不收窄为 `list[str]`**：统一为 `QualityIssue{code, severity?, column?, detail?, source: cleaner\|fallback}`，cleaner 字符串映射到 `code`、fallback dict 保留字段；删 `.get("severity")` 崩溃点，触发判断改读 `has_critical_issues` | `src/mcp_client/models.py` | 评审修正 |
| B6（=M3） | `estimated_impact`/`changes_summary` 类型修正为 dict，**以一次真实服务响应或工具 JSON Schema 验证后定型**，不只依据 handoff 文字 | `src/mcp_client/models.py` | 评审新增 |
| B7（=M4） | 删除 `get_cleaning_plan` 调用（避免双重规划）；`clean_dataset` 显式传 `planner_mode`（读 Settings）、`max_rounds=1`、不传 plan；decision trace 记录返回值里的 `plan_steps` | `src/mcp_client/data_cleaner.py`、`src/graph/nodes.py` | 评审定案 |
| B8（=M6） | **超时要改接口、重试要加谓词**：`MCPClient` 现在一个实例只有统一 timeout，无法按工具分层——接口改为 `call_tool(tool, args, timeout=..., deadline=...)`（profile/validate/retrieve 10–15s，clean 60s+），区分连接/初始化/工具执行，并约束**总 deadline**（单次 60s × 重试 3 次的总等待远超 60s）。另**已核实**：tenacity 装饰器对 `MCPToolError` 也会白白重试 3 次，要加 retry predicate**只重试 `MCPConnectionError`** | `src/mcp_client/client.py` | 评审扩充 |
| B9（=M8） | **文件路径防护**：校验 `cleaned_path` 位于允许的 run/workspace 目录、无路径穿越、文件存在可读且格式受支持；记录输入输出内容哈希与文件大小。成功条件是"路径不同 + 内容确有变化 + validation 通过"**三者共同构成**，而不只是路径不同 | `src/graph/nodes.py` 或工具层 | 评审新增 |
| B10 | `planner_mode` **配置化**：经 Pydantic Settings（如 `DATA_CLEANER_PLANNER_MODE`）按环境选择，不在调用点或脚本里散落硬编码；启动时校验选 `llm` 时 API key 与服务能力存在，否则在进入耗时流程前明确失败 | `src/config/settings.py` | 评审新增 |

**B1 建议由 rag-framework 服务端配默认值而非 MAEDA 逐次传参**（检索配置属于检索服务的职责）——定案改为**显式枚举契约**（`embedding_provider: Literal["server_default","none","openai"]` 等），不用 null/bool 哨兵，未传字段归一化为 `server_default`。B2 依赖 8.2 的决定，暂缓。

**B3 的定案理由**（评审核心洞察，成立）：工具错误可能是参数错、schema 不兼容、认证失败、版本错配——**联调阶段最该暴露的契约错误，反而会被无条件 fallback 掩盖**。不建策略框架，一个环境变量两档即可。连带要求：TB 验收标准是 `mode="mcp"`，**不能只以"pipeline 没崩"为成功**。

### 6.2.1 M7：清洗回环重写（评审最有价值的发现，已核实属实）

**这是三方此前都漏掉的真 bug，不是设计分歧。** `connect_and_profile_node`（`src/graph/nodes.py`）现在每轮都从 `data_sources[0]` 读**原始路径**：清洗成功后只更新了 `active_source` 和 `schema_summary`，**没有把 `cleaned_path` 写回 source**；而且最后存进 `data_quality_report` 的是**清洗前**的 report。

后果：首轮 `has_critical_issues=true` 后 router 重进同一节点，**下一轮仍然 profile 并清洗原始文件**，空转到三轮上限；且达上限后照常进入分析，**不代表质量已合格**。

定案（进阶段 1，作为 TB1 前置）：

- 清洗成功后把 `cleaned_path` **写回 `data_sources[0].path` 与 `active_source`**
- 对清洗后的文件**重新 profile**，新报告写入 `data_quality_report`，**router 只读最新报告**
- 接入 `validate_quality` 作为清洗后退出条件——**已核实它现在从未被图调用过**，只有客户端方法存在
- **仅当**输出文件存在、可读、确实被 MAEDA 接管时才置 `cleaning_applied=true`
- 达最大轮次仍未通过时，写入明确的 **terminal caveat 或 error，不得静默按 `ready` 处理**
- 连续两轮输出路径、内容哈希或质量结果无变化时**提前停止**

接入位置：`profile → decide → clean → validate cleaned file →（passed → re-profile/analysis | failed → next round/terminal caveat）`。validation 与 re-profile 结果均写入 state 和 decision trace。多轮停止规则与 TB4 联动。

### 6.3 策展层（口径词表）——MAEDA 独立工作，不阻塞子系统

方案已在 rag 文档 5.1 选定：**YAML 存储 + 按真实 schema 过滤后注入**，附 `full`/`partial`/`absent` 三态漂移检测写入 state。注入点**修订为两处**（初版只写了第 [3] 步）：cleaner 提出 intent-driven 清洗需要列语义/口径在**清洗前**可用，按 rag 文档两层原则这是策展层内容、确定性注入不走检索，因此除第 [3] 步 `plan_analysis` 的 prompt（`src/agents/analysis_agent.py:166-177` 拼 prompt 处）外，词表按-schema-过滤的结果还要注入第 [2] 步 `connect_and_profile_data` 传给 cleaner 的 intent payload。连带影响：**词表过滤时机要提前到第 [2] 步**（schema 一抽出来就过滤，不能等到 [3]）。payload 具体格式与 cleaner 联合设计（见第 4 节议程议题一）。

配套的未知上传数据源处理（rag 文档 5.2）分两件事：

- **(a) 防瞎编**：词表状态 `absent` 时在 planner prompt 加显式约束（不得基于列名推断业务含义）。改 `src/config/agent_prompts.py`。
- **(b) 报告内 caveat**：⚠️ **rag 文档说"MAEDA 的 insight 已有 confidence 字段与 caveat 机制，挂在那里最自然"——这句只对了一半。** `Insight` dataclass（`src/agents/insight_agent.py:36-44`）有 `confidence`，但**没有 caveat 字段**；`insight_agent.py:400` 处的 `caveat` 是把 stats_tool 产出的工具级警告（`src/tools/stats_tool.py:333`）拼进 evidence 字符串，不是一个结构化的 caveat 通道。所以 (b) 需要**新增字段**，不是挂在现有机制上，工作量比 rag 文档估计的大。

两条明确不做（rag 文档 5.2 已论证，采纳）：不因词表缺失机械压低 `confidence`（会污染后续 RAG on/off 的 A/B 测量）；不做成 guardrail 阻断（这是低保证输出，不是不安全输出）。

### 6.4 评测相关

| # | 事项 | 说明 | 来源 |
|---|---|---|---|
| D0 | **多 trial 支持**：`run_eval.py` 增加 `--trials`/`--suite`/`--concurrency`，报告 success rate、`pass@k`、`pass^k` 及方差 | **新增的硬前提**。现状是单次 trial，任何"改造前后对比"或"on/off 对比"都是 n=1 vs n=1，过不了本项目已测得的噪声地板（`docs/noise_floor.md`）。**D3/D4/D6 全部依赖它** | eval 方案 §7 / Phase C |
| D1 | 事件脚本（ground truth 规格） | **归属已定**：放 rag-framework 仓库，YAML 只含内容与 `version`，SHA-256 放 **sidecar manifest**（避免哈希自引用）。MAEDA 只读引用，除校验哈希外**还要固定 rag 仓库 commit/tag**（自报哈希防不住同路径内容漂移）。硬约束不变：必须先写完（TB5 冻结），数据和语料才能各自独立渲染 | rag 文档 7 + 协商/评审定案 |
| D2 | `generate_demo_data.py` 读事件脚本生成数据 | `scripts/generate_demo_data.py`，依赖 D1 | rag 文档 5.3 |
| D3 | 端到端 A/B 评测：RAG on/off × 归因准确率 / 错误归因率 / 幻觉率 | 依赖 D0+D1+D2。**范围已扩大，见下方 D3 补充** | rag 文档 5.5 + eval 方案 §12 |
| D4 | Data Cleaner on/off 的对照 demo：同一份脏数据 + 同一个问题，无 cleaner 时分析错/失败，有 cleaner 时正确 | cleaner 侧的核心验收标准，也是面试演示点。同样受 D3 补充约束 | cleaner 文档 |
| D5 | 找第二标注员标 30 条子集，补 inter-annotator 一致性 | 与子系统无关，异步进行 | 今日 grill-me |
| D6 | tool-calling 改造后重跑 eval：v1 数字冻结，人工标注继续作为 judge 校准集复用（非"永久金标"，见 eval 方案 §9.2） | 依赖 D0 + 6.6 的改造完成 | 今日 grill-me |

**D1 归属已定，不再是会上决定项**——剩下的是把脚本本身写出来，它仍阻塞 D2 和两边的独立渲染。**D0 是唯一卡住所有评测结论的技术前提**，不需要会上讨论，但必须排在 D3/D4/D6 之前。

#### D3 补充：A/B 评测的范围要比子系统文档写的更大

两份子系统文档给的指标都只测了「该触发时有没有触发」这一半。按 eval 方案 §12，路由类行为必须**正反成对**测，否则一个"永远调用 cleaner / 永远检索"的退化策略也能拿满分：

| 应触发（子系统文档已覆盖） | 不应触发（缺失，需补） |
|---|---|
| 数据质量差时调用 cleaner | 已干净数据不要重复清洗 |
| 需要领域知识时调用 RAG | 纯计算任务不要强行检索 |
| MCP 失败时 fallback | MCP 正常时不要错误降级 |

且报告应给 **precision / recall / 混淆矩阵**，而不是单一 accuracy。

⚠️ **"纯计算任务不要强行检索"这条当前压根测不了**：图现在**无条件**经过 `retrieve_domain_knowledge` 节点。定案补一项（见 6.6 F1）：加明确可测试的路由条件 + 正式的 RAG on/off 配置。

#### D3 补充二：评测方法（评审新增，与 D0 的 pass@k 并存但不互相替代）

- **`pass@k`/`pass^k` 只适合二元成功指标**，不能替代 groundedness、错误归因率这类**连续指标**——后者要报**均值差、方差与 bootstrap 置信区间**。
- **配对设计**：A/B 用同一 case、同一输入、配对 trial。
- **可复现性固定项**：事件脚本版本 + 三仓库 commit/tag + 输入内容哈希 + 运行配置，全部记录。
- **MCP 可用率 / 超时率 / fallback 率作为独立运行指标单独报告**，不与质量指标混算。
- **调参用 dev split，冻结后才跑 holdout。**
- 事件脚本放 rag 仓库可以，但 **MAEDA 不能只校验文件内自报的哈希**，还要**固定 rag 仓库的 commit/tag**——同路径内容可能已漂移，自报哈希防不住。

另需先落地 eval 方案 §10 的失败分类中的 `mcp_error` 与 `environment_error` 两类——**否则子系统超时会被记成「agent 推理失败」，D3/D4 的结论直接失效**。这条对两个子系统方尤其重要：它决定了你们的服务不稳定时，锅会不会错误地算到 MAEDA 头上。

按 eval 方案 §8.3，工具调用次数与顺序属于**诊断指标，默认不判成败**（除非工具选择本身就是被测能力）。这也是 6.6 的 tool-calling 改造不会让现有标注作废的根本原因——outcome-first grading 对路径不确定性免疫。

### 6.5 既有 bug（会污染评测结果，需在 D3/D4 之前修）

| # | 事项 | 位置 | 来源 |
|---|---|---|---|
| E1 | 多表数据源硬编码取 `tables[0]`，不参照用户问题选表 | `src/tools/data_connector.py:213`（rag 文档写的 209 已漂移） | rag 文档 6.1 |
| E2 | **改为 schema-aware intent refinement，不是单纯修时序**：`parse_intent`（图第 1 步）读到的 `schema_summary` 恒为空串——赋值发生在第 2 步 | 读：`src/agents/intent_parser.py:114,194`；写：`src/graph/nodes.py:154`；初值：`src/state/graph_state.py:87` | rag 文档 6.2 + 评审改写 |
| E3 | **改为 EvalRunner 终止状态，不能机械打零分**：`handle_error → persist_run` 直连，**绕过 `run_eval`**，与 CLAUDE.md 第 8 条"eval 在每次执行上运行，无例外"矛盾 | `src/graph/builder.py:96-97`、`src/eval/` | 今日 grill-me + 评审改写 |
| E4 | **Trial 之间未隔离**：各 trial 共享 `data/charts` 与 RunStore，并发时图表互相覆盖、可能读到上一次运行的产物 | 见 eval 方案 §5 / Phase B；涉及 `src/agents/viz_agent.py`、`src/tools/chart_tool.py`、`src/persistence/run_store.py` | eval 方案 §16（列为「只做一件事就做这个」的第一位） |

E1 尤其要紧：demo 数据 `data/demo/ecommerce_orders.db` 是多表的，若某轮评测选错表，归因准确率会混入噪声，分不清是"RAG 没帮上忙"还是"数据源本来就选错了"。

E4 与 E1 同级：D0 的多 trial 一旦并发跑起来，共享 artifact 目录会让不同 trial 的图表互相覆盖，评测结果不可信。**E4 必须与 D0 一起做**，单独上多 trial 反而会放大这个 bug。**隔离范围**（评审补充）不止 charts 与 RunStore，还要含 **cleaned files、临时文件、LLM trace、缓存**——M7 落地后每轮清洗都会产出新文件，这块不隔离会直接串味。

#### E2 补充：为什么不是"把 parse_intent 挪到第 2 步之后"就完了

评审指出更深一层：图的形状决定了 intent parser **首次运行必然看不到真实 schema**；而 cleaner 的 intent-driven cleaning 消费的 intent，本身可能就是在缺 schema 的情况下生成的——**口径词表也补不上这个洞**（词表补的是列语义，补不了"意图当初就没见过这张表"）。

定案采用**两阶段方案**（对现有图改动较小）：第一次 parse 出**粗意图**，拿到 schema 后执行 **schema-aware refinement**，用 LangGraph 节点 + 条件边实现，**两阶段的意图变化写入 decision trace**。

#### E3 补充：失败路径跑 eval ≠ 给失败运行打零分

"失败路径也要跑 eval"这点不变，但修法**不是**把 `handle_error` 直接接到现有的成功态 EvalRunner。EvalRunner 要支持**终止状态**：`success` / `safe refusal` / `pipeline error` / `mcp error` / `environment error` / `guardrail rejection`。

失败运行**只计算适用指标**，答案相关指标标 `not_applicable`——**不能用零分混淆"没有生成答案"和"生成了低质量答案"**，这两者在聚合分里必须可分。eval 自身失败也要被捕获、分类、持久化。

这与 D3 补充里的失败分类（`mcp_error`/`environment_error` 与 agent 推理失败分开记）是同一件事的两端：对两个子系统方尤其重要——**它决定了你们的服务不稳定时，锅会不会错误地算到 MAEDA 头上**。

### 6.6 MAEDA 自身架构改造（与子系统无接口影响，但会改变调用时机）

**Analysis Agent 从 plan-then-execute 改为真 tool-calling 循环。** 现状：`plan()` 发一次 LLM 调用产出完整 JSON 计划，`execute()` 由 Python 代码查 `TOOL_REGISTRY` 字典（`src/agents/analysis_agent.py:105-112`）逐步派发——模型写计划时没见过任何数据结果，之后全程不参与决策。改造后模型每看到一次工具结果再决定下一步。

**对子系统的影响：接口形状不变**。需要会上确认的一点是：两个子系统的工具是否要以 tool-calling 形式暴露给 Analysis Agent，还是继续只在固定图节点（`connect_and_profile_data` / `retrieve_domain_knowledge`）里调用。**倾向后者**——数据质量剖析和知识检索是确定性该发生的步骤，不需要 LLM 自己决定"要不要查数据质量"。

同期还有：外层图对外表述改口为 "workflow"（承认它是确定性编排，引 Anthropic *Building Effective Agents* 的立场作为工程判断）；README 重构为单一版本、并列 "Evaluation Methodology" 与 "Orchestration Design" 两个深潜区块。这两项纯文档，不影响接口。

#### F1：RAG 条件路由 + 正式 on/off 开关（评审新增，对子系统有可见影响）

评审指出一处不一致：**当前图无条件经过 `retrieve_domain_knowledge`**，阶段 4 却要验证"纯计算任务不强行检索"——这条现在压根测不了。定案：

- 加**明确可测试的路由条件**，决定这次要不要检索
- 加**正式的 RAG on/off 配置**（评测时**不许临时改代码**开关）
- 区分 **`skipped` / `mcp` / `fallback` 三种状态**（现在只有后两种，"没调"和"调了但降级"混在一起）
- **路由决策写入 decision trace**
- 配**正反样本与混淆矩阵**（对应 6.4 D3 补充的正反成对要求）

对 rag-framework 的影响：调用量会下降（纯计算类查询不再检索），这是预期行为不是故障；`skipped` 状态在 `mcp_call_log` 里可区分，别把它统计成 fallback 率。

### 6.7 执行顺序

```
A4 TB0 传输冒烟 ──→ 阶段1：B3-B10 契约修复 + M7 清洗回环重写
  (不过不开工)          (M7 是 TB1 前置，不是可选项)
                              │
                              ▼
                        A5 TB1 契约 E2E（strict 模式）
                         (阻塞后续全部)
                              │
        ┌─────────────────────┼── TB4 清洗质量语义（阶段2首项）
        │                     ├── TB3 intent payload schema
        │                     ├── TB5 事件脚本冻结（D1，rag 仓库）
        │                     │
        ├─→ 并行：MAEDA 口径词表(6.3) │ cleaner intent-driven │ rag 脏语料
        │        + F1 RAG 条件路由与 on/off 开关
        │
        ├─→ E1/E2/E3 修 bug（必须早于 D3/D4）
        │
        └─→ D2 ──┐
                  ├─→ D3/D4 端到端评测 → 依据结果决定 8.2/8.3
D0 + E4 ──────────┘
(多 trial + trial 隔离，
 隔离范围含 charts/RunStore/
 cleaned files/临时文件/trace/缓存)

独立并行、不依赖子系统：
   D5 第二标注员
   6.6 tool-calling 改造 ──┐
              D0 + E4 ─────┴─→ D6 重跑 eval（改造前后对比）
```

**M7 的位置要特别注意**：它排在阶段 1、TB1 之前，而不是"以后再优化"。因为 TB1 的验收标准之一就是**清洗回环真实收敛**——M7 不做，TB1 里那条"清洗后文件被下一轮消费"根本无从验证，整个 Data Cleaner 集成看起来能跑（不崩、有 `mode="mcp"`），实际是空转三轮后照常进入分析。

**评审后新增的关键依赖**：`M7` 是 TB1 的前置（见上），`D0 + E4` 是所有对比类结论的前提。它不依赖两个子系统，MAEDA 侧可立即开工，但**它不完成，D3/D4/D6 三项的数字都不可信**——单次 trial 的对比过不了噪声地板，共享 artifact 目录的并发 trial 会互相污染。子系统方据此可以判断：即使连通性当天打通、语料当周就绪，端到端 A/B 的可信结论仍要等 MAEDA 这一步。
