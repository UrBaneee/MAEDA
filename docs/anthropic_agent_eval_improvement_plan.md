# MAEDA Agent Eval 改造方案

> 基于 Anthropic《Demystifying evals for AI agents》的方法，对 MAEDA 现有评测体系进行增量升级。

## 1. 改造目标

MAEDA 已具备较完整的 Eval v2 基础。本次改造不重建现有评测系统，而是把它从“对单次运行计算综合分数”升级为“能够可信测量 Agent 能力、稳定性和回归风险的实验系统”。

完成改造后，评测系统应能回答：

1. 测试任务本身是否明确、可解且公平？
2. Grader 能否正确接受合格解、拒绝错误解？
3. 每次 trial 是否从独立、可复现的环境开始？
4. MAEDA 是偶尔成功，还是能够稳定成功？
5. 一次失败来自 Agent、Task、Grader、MCP 还是基础设施？
6. 修改模型、Prompt、工具或图结构后，是否产生真实提升或回归？

## 2. 当前基础与主要缺口

### 2.1 已有能力

当前项目已经具备：

- 100 条 golden cases，以及固定的 59/41 dev/test split；
- Code-based 与 LLM-based graders；
- 人工标注、judge calibration 和 agreement 计算；
- Judge noise 与 full-pipeline noise 测量；
- Replay cache；
- 每次 pipeline invocation 创建新的 `MAEDAState` 和 `run_id`；
- `decision_trace`、`mcp_call_log`、token、cost 和运行记录；
- Guardrail 和 Eval 在每次执行中运行；
- 回归比较、paired bootstrap 和独立 eval judge 配置。

### 2.2 主要缺口

1. Golden case 只有部分 ground truth，缺少结构化的 success criteria 和 reference outcome。
2. 每次 trial 虽然新建 `MAEDAState`，但图表目录、RunStore 和 Agent/MCP 实例仍有共享部分。
3. 常规 `run_eval.py` 主要执行单次 trial，没有把多次运行和可靠性指标产品化。
4. Dev/test 与 capability/regression/safety 的分类尚未分开。
5. 评分仍混合 outcome、必要流程约束和诊断指标。
6. 失败分类粒度不足，目前主要是 `safe_refusal` 和 `pipeline_error`。
7. 缺少标准化 transcript review queue。
8. 已支持多轮对话，但尚无 multi-turn golden suite。

## 3. 目标评测模型

```text
明确且可解的 Task
        ↓
已验证的 Reference Contract
        ↓
隔离环境中的多个 Trials
        ↓
Outcome-first Grading
        ↓
可靠性与不确定性统计
        ↓
Transcript Review 与失败归因
        ↓
真实失败进入 Regression Suite
```

## 4. Task 与 Reference Contract

### 4.1 扩展 `GoldenTestCase`

在 `src/eval/runner.py` 中扩展 `GoldenTestCase`：

```python
@dataclass
class GoldenTestCase:
    id: str
    query: str
    query_type: str
    expected_metrics: list[str]
    expected_dimensions: list[str]
    ground_truth: dict
    data_source: Optional[dict] = None
    tags: list[str] = field(default_factory=list)
    expected_tools: Optional[list[str]] = None
    expected_chart_types: Optional[list[str]] = None
    split: Optional[str] = None

    # New fields
    suite_type: Literal["capability", "regression", "safety"] = "capability"
    success_criteria: list[dict] = field(default_factory=list)
    reference_outcome: Optional[dict] = None
    forbidden_outcomes: list[dict] = field(default_factory=list)
    expected_behavior: Optional[dict] = None
    scoring_policy: dict = field(default_factory=dict)
    reference_status: Literal[
        "verified",
        "needs_review",
        "data_mismatch",
        "unanswerable",
    ] = "needs_review"
```

### 4.2 Reference 的定位

Reference solution 不应是一篇要求逐字匹配的标准报告，而应是结构化的“已知合格 outcome”：

```yaml
reference_outcome:
  facts:
    - metric: q2_sales_change_pct
      value: -18.24
      tolerance: 0.5
    - metric: primary_driver
      accepted_values: [order_volume, number_of_orders]
  limitations:
    - current data cannot establish why order volume declined
  acceptable_chart_types: [waterfall, bar]
  required_evidence:
    - sales table
    - quarter and region dimensions
```

Reference 的作用包括：

- 证明 task 可解；
- 校验 grader；
- 具体化产品成功标准；
- 设计 partial credit；
- 对照分析 Agent 失败；
- 接纳不同但同样合理的解决路径。

### 4.3 Success Criteria

每个 case 应显式声明 grader 检查的条件：

```yaml
success_criteria:
  - id: correct_change
    type: numeric
    expected: -18.24
    tolerance: 0.5
    required: true

  - id: grounded_driver
    type: semantic
    criterion: primary driver is supported by computed evidence
    required: true

  - id: evidence_boundary
    type: llm_rubric
    criterion: does not present an unsupported causal explanation as fact
    required: true
```

所有 grader 检查的 required 条件都必须在 task 描述或 success criteria 中公开，禁止隐藏要求。

### 4.4 Eval Suite Validator

新增：

```text
scripts/validate_eval_suite.py
```

职责：

- 验证数据源存在；
- 从 fixture 重新计算 ground truth；
- 验证字段、表和维度存在；
- 验证 reference outcome 能通过 required graders；
- 验证 known-bad outcome 会被拒绝；
- 检查 task、reference 和 grader 是否一致；
- 标记 ambiguous、unanswerable 和 data-mismatch cases；
- 检查每个 case 是否有唯一 ID、固定 split 和 suite type。

CI 应先执行 suite validation，再运行 Agent eval。

## 5. Trial 环境隔离

### 5.1 目标目录结构

每个 trial 使用独立目录：

```text
logs/eval_trials/
  <eval_session_id>/
    <case_id>/
      <trial_id>/
        input/
        artifacts/
          charts/
        state.json
        transcript.json
        outcome.json
        grader_results.json
        metadata.json
        runs.db
```

如不需要长期保留某次 trial，可在临时目录中运行，仅在失败或抽样命中时保存产物。

### 5.2 新增 `TrialContext`

新增：

```text
src/eval/trial_context.py
```

建议结构：

```python
@dataclass(frozen=True)
class TrialContext:
    eval_session_id: str
    case_id: str
    trial_id: str
    workspace_dir: Path
    input_dir: Path
    artifacts_dir: Path
    database_path: Path
    seed: Optional[int]
```

`run_one_case()` 接收 `TrialContext`，并把必要字段写入 `MAEDAState`：

```python
state["eval_context"] = {
    "eval_session_id": context.eval_session_id,
    "case_id": context.case_id,
    "trial_id": context.trial_id,
    "suite_type": tc.suite_type,
}
state["artifact_dir"] = str(context.artifacts_dir)
```

### 5.3 文件系统隔离

- 将 source fixture 复制到 trial 的 `input/`，避免修改原始数据；
- `VizAgent` 优先使用 `state["artifact_dir"]`；
- 图表文件名加入 `run_id` 或 trial ID；
- 不同 trial 的 artifact paths 必须互不相交；
- 并发 trial 不得覆盖同名图表。

### 5.4 数据库与共享实例

- Eval trial 优先使用独立 SQLite 文件；
- 若继续使用共享 RunStore，必须保证 Agent 无法读取其他 trial 记录；
- Agent 单例可以保留，但所有运行期可变状态必须位于 `MAEDAState`；
- 审计 MCP client、retry、circuit breaker 和 cache 是否保存跨 trial 状态；
- 增加顺序运行和并发运行的泄漏测试。

### 5.5 隔离验收标准

- 每个 trial 具有唯一 `run_id` 和 `trial_id`；
- `conversation_history` 默认是新的空列表；
- 两个并发 trial 不共享 trace、token、artifact 或数据库状态；
- 同一 case 连续运行不会读取或覆盖上一次结果；
- fixture 在 trial 完成后内容不变；
- Eval replay 必须明确标记为 replay，不能被统计为新的 Agent trial。

## 6. Suite 分层

`split` 和 `suite_type` 是两个独立维度。

### 6.1 Split

- `dev`：允许查看、调试和迭代；
- `test`：holdout，只用于正式 reveal，不用于日常调参。

### 6.2 Suite Type

- `capability`：测试目前仍困难的能力，允许较低通过率；
- `regression`：保护已经稳定具备的行为，应接近 100%；
- `safety`：零容忍要求，不能被平均分稀释。

### 6.3 建议运行层级

#### Smoke Suite

5–10 条，PR 必跑：

- graph 编译和基本路由；
- 基本数据分析；
- guardrail 被调用；
- eval 被调用；
- fallback 不崩溃。

#### Regression Suite

- 意图解析；
- groupby、filter、SQL join；
- grounded report；
- MCP fallback；
- chart generation；
- guardrail 和 trace requirements。

#### Capability Suite

- 多数据源分析；
- 模糊问题澄清；
- 数据不足时说明限制；
- MCP 局部故障；
- 开放式探索；
- RAG 和数据证据冲突。

#### Safety Suite

- 敏感信息泄露；
- SQL 越权；
- prompt injection；
- unsupported causal claims；
- guardrail fail-open；
- 数据不足时编造结果。

## 7. 多 Trial 与可靠性统计

### 7.1 CLI

扩展 `scripts/run_eval.py`：

```bash
poetry run python scripts/run_eval.py \
  --split dev \
  --suite regression \
  --trials 3 \
  --concurrency 4
```

### 7.2 Task Pass Policy

不要用统一的 aggregate threshold 决定所有任务是否成功。每个 case 定义自己的策略：

```yaml
scoring_policy:
  mode: hybrid
  required:
    - factual_accuracy >= 0.9
    - guardrail_executed == true
    - pipeline_error == false
  weighted_threshold: 0.75
```

支持：

- `binary`：所有 required assertions 必须通过；
- `weighted`：综合分达到阈值；
- `hybrid`：required checks 全通过且综合分达标。

MAEDA 默认使用 hybrid。

### 7.3 报告指标

每个 case 报告：

- trial success rate；
- `pass@k`：k 次中至少一次成功；
- `pass^k`：k 次全部成功；
- score mean、median、std、min、max；
- token、cost 和 latency 分布；
- MCP fallback rate；
- failure class 分布。

更多 tasks 测覆盖率；更多 trials 测同一场景的稳定性，二者不可替代。

### 7.4 建议发布门槛

```text
Smoke:      pass^1 = 100%
Regression: pass^3 >= 95%
Capability: 与 baseline 做 paired bootstrap，不使用固定高通过率要求
Safety:     任意 required safety assertion 失败即阻止发布
```

最终阈值应在获得实际分布后校准，而不是直接采用上述示例。

## 8. Outcome-first Grading

将指标分成三层。

### 8.1 Outcome Correctness

最高优先级：

- 数值是否正确；
- 结论是否有数据或 RAG 证据；
- 数据不足时是否说明限制；
- 文件和图表是否真实生成；
- 图表是否使用正确字段和数据；
- guardrail 是否阻止不安全 outcome。

### 8.2 Required Process Constraints

仅强制必要流程：

- guardrail 必须运行；
- eval 必须运行；
- decision trace 必须记录；
- token usage 必须记录；
- 高风险操作必须经过规定检查。

### 8.3 Diagnostic Metrics

默认不直接决定任务是否成功：

- 工具调用次数；
- 普通工具调用顺序；
- graph steps；
- retry count；
- latency 和 cost；
- fallback frequency。

除非工具选择本身就是被测能力，否则不应因 Agent 采用另一条正确路径而判失败。

## 9. Grader 校准与测试

### 9.1 Deterministic Grader

每个 grader 至少覆盖：

- 合格正例；
- 合格但表达不同的正例；
- 已知错误反例；
- 边界误差；
- 缺字段与异常格式；
- grader 不应崩溃的 malformed input。

### 9.2 LLM Grader

- 继续使用人工标注集计算 QWK、Spearman、MAE 和 confusion matrix；
- 分维度调用 judge，避免一个总体印象影响所有评分；
- 允许返回 `Unknown` 或 invalid；
- 记录 judge provider、model、prompt version 和 rubric hash；
- 优先关注 false positive，即错误结果被 grader 放过；
- 定期用新样本重新校准，而不是永久依赖同一批标签。

### 9.3 Grader Acceptance

一个 grader 进入正式 aggregate 前，应满足：

1. Verified reference positives 能通过；
2. Known-bad negatives 能失败；
3. 边界行为符合 rubric；
4. 与人工标注达到事先声明的一致性要求；
5. Prompt 或 judge model 更新后重新生成 calibration report。

## 10. 失败分类

扩展当前错误模型：

```python
FailureClass = Literal[
    "agent_reasoning_error",
    "agent_tool_error",
    "agent_grounding_error",
    "agent_instruction_error",
    "task_invalid",
    "task_ambiguous",
    "grader_false_positive",
    "grader_false_negative",
    "environment_error",
    "mcp_error",
    "rate_limit_error",
    "data_mismatch",
    "expected_safe_refusal",
]
```

每个失败 trial 保存：

```json
{
  "status": "failed",
  "failure_class": "agent_grounding_error",
  "failure_stage": "generate_insights",
  "summary": "Primary driver was not supported by computed findings",
  "review_status": "unreviewed"
}
```

规则或 LLM 可以提出候选分类，但高风险失败和 grader/task 错误应由人工确认。

## 11. Transcript Review

### 11.1 Trial Transcript

每个 trial 保存统一的可观察轨迹：

```json
{
  "trial_id": "...",
  "case_id": "...",
  "events": [
    {
      "sequence": 1,
      "node": "parse_intent",
      "input_summary": {},
      "output_summary": {},
      "tool_calls": [],
      "decision": {},
      "token_usage": {},
      "duration_ms": 940
    }
  ],
  "outcome": {},
  "grader_results": []
}
```

不依赖或保存模型隐藏推理。保存可观察的状态变化、决策摘要、工具调用、证据、结果和成本即可。

### 11.2 Review Queue

新增：

```text
scripts/build_review_queue.py
```

优先抽取：

1. 所有 safety failures；
2. 新的 failure signature；
3. Code grader 与 LLM grader 冲突；
4. aggregate 或关键指标显著下降；
5. pipeline、MCP 或 grader errors；
6. Agent 采用 reference 未覆盖的合理路径；
7. 随机 5% 通过样本，用于发现 false positives。

Review 的产物必须是分类和后续动作，而不只是“已阅读”。

## 12. Balanced Behavioral Cases

对关键行为同时测试应该发生和不应该发生：

| 应触发 | 不应触发 |
|---|---|
| 数据质量差时调用 cleaner | 已干净数据不要重复清洗 |
| 需要领域知识时调用 RAG | 纯计算任务不要强行检索 |
| 信息不足时拒绝推断 | 信息充分时不要过度拒绝 |
| 异常值影响结论时警告 | 正常波动不要误报异常 |
| 有合适数据时生成图表 | 无法表达时不要硬画图 |
| MCP 失败时 fallback | MCP 正常时不要错误降级 |
| 敏感输出被阻止 | 正常业务分析不要误杀 |
| 模糊请求时澄清 | 清晰请求不要多问 |

关键路由至少覆盖：

- true positive；
- true negative；
- false-positive trap；
- false-negative trap。

对路由类行为优先报告 precision、recall 和 confusion matrix，而不只报告 accuracy。

## 13. Multi-turn Eval Suite

新增：

```text
tests/eval/multiturn_suite.json
```

示例：

```json
{
  "id": "MT01",
  "suite_type": "regression",
  "turns": [
    {
      "user": "比较各地区销售额。",
      "expected_state": {
        "target_metrics": ["sales"],
        "dimensions": ["region"]
      }
    },
    {
      "user": "现在按季度再拆一下。",
      "expected_state": {
        "target_metrics": ["sales"],
        "dimensions": ["region", "quarter"]
      }
    },
    {
      "user": "哪个地区下降最严重？",
      "expected_outcome": {
        "uses_prior_context": true,
        "does_not_invent_metric": true
      }
    }
  ]
}
```

覆盖：

- 应继承的上下文；
- 不应继承的状态；
- 新指令覆盖旧约束；
- clarification 后继续执行；
- 长历史裁剪；
- 不同会话不得串线；
- 历史消息中的 prompt injection 不得覆盖当前系统约束。

## 14. CI/CD 分层

### Pull Request

- Unit tests；
- `validate_eval_suite.py`；
- deterministic smoke eval；
- 小型 regression suite；
- trial isolation tests。

### Main / Nightly

- 完整 dev regression suite；
- capability suite；
- 每 case 3–5 trials；
- paired bootstrap comparison；
- 生成 transcript review queue。

### Release

- 明确授权的 holdout test reveal；
- safety suite；
- MCP integration suite；
- multi-turn suite；
- 人工抽查失败和一部分通过样本。

### Release Report

至少包含：

- regression `pass^k`；
- safety violation count；
- pipeline error rate；
- invalid grader count；
- invalid task count；
- 按任务类别的通过率；
- paired-bootstrap confidence interval；
- cost per successful task；
- latency p50/p95；
- MCP fallback rate；
- human/judge disagreement。

不得只用一个 aggregate score 作为发布判断。

## 15. 分阶段实施计划

### Phase A：Reference Contract 与 Suite Validation

任务：

1. 扩展 `GoldenTestCase` schema；
2. 为现有 100 cases 补充 `suite_type`；
3. 先为 10–20 个高价值 cases 编写完整 reference contract；
4. 实现 `validate_eval_suite.py`；
5. 为 deterministic 和 LLM graders 增加正例、反例、边界例测试。

验收：

- Verified reference 全部通过 required graders；
- Known-bad outputs 全部被对应 grader 拒绝；
- Task、reference 和 grader 不存在隐藏要求；
- Invalid task 会显式失败，不能进入正式分数。

### Phase B：Trial Sandbox

任务：

1. 新增 `TrialContext`；
2. 每个 trial 建独立目录；
3. 输入 fixture 复制到 trial workspace；
4. Viz artifacts 写入独立目录；
5. 隔离或严格约束 RunStore；
6. 审计 Agent/MCP singleton mutable state；
7. 增加顺序和并发 isolation tests。

验收：

- 连续和并发 trial 均无 state、token、文件和缓存泄漏；
- artifact 不覆盖；
- 输入 fixture 不被修改；
- replay 与真实 trial 在报告中明确区分。

### Phase C：Reliability Evaluation

任务：

1. `run_eval.py` 支持 `--trials`、`--suite` 和 `--concurrency`；
2. 实现 case-specific pass policy；
3. 计算 success rate、`pass@k` 和 `pass^k`；
4. capability、regression、safety 分开汇总；
5. 发布比较使用 paired bootstrap。

验收：

- 每次 trial 独立保存；
- 报告展示均值、方差和失败分布；
- Safety failure 不能被 aggregate 掩盖；
- Infra failure 不被统计为 Agent reasoning failure。

### Phase D：Failure Review 与覆盖扩展

任务：

1. 实现 failure taxonomy；
2. 保存标准 transcript；
3. 实现 review queue；
4. 添加 balanced behavioral cases；
5. 建立 multi-turn suite；
6. 将真实用户或生产失败持续转成 regression cases。

验收：

- 每个失败可归因到 Agent、Task、Grader、MCP 或环境；
- 新 failure signature 会进入人工 review；
- 通过样本也接受随机抽查；
- 每个关键路由拥有正反成对测试；
- 多轮上下文继承和跨 session 隔离均有覆盖。

## 16. 建议优先实施的三个改动

如果只安排一个短迭代，优先完成：

1. **Trial 独立目录**：解决当前共享 `data/charts` 的覆盖和并发污染风险；
2. **Reference Contract**：让 task、reference 和 grader 三者可以自动验证；
3. **原生多 Trial 报告**：让常规 eval 能测可靠性，而不只是单次分数。

这三项完成后，MAEDA 将从“有一套评分器”进入“有一套可信 Agent Eval Harness”的阶段。

## 17. 预期涉及文件

| 文件 | 计划修改 |
|---|---|
| `src/eval/runner.py` | 扩展 case schema、pass policy、多 trial 聚合 |
| `src/eval/trial_context.py` | 新增 trial sandbox context |
| `src/state/graph_state.py` | 增加 eval context 与 artifact directory |
| `src/agents/viz_agent.py` | 使用 trial-scoped artifact directory |
| `src/tools/chart_tool.py` | 支持 trial-safe 文件名 |
| `scripts/run_eval.py` | 增加 suite/trials/concurrency 和新报告格式 |
| `scripts/measure_noise.py` | 复用统一 trial runner，明确 replay 与 fresh run |
| `scripts/validate_eval_suite.py` | 新增 suite/reference/grader validation |
| `scripts/build_review_queue.py` | 新增人工审查队列生成 |
| `tests/eval/test_suite.json` | 增加 reference contract 与 suite type |
| `tests/eval/multiturn_suite.json` | 新增多轮评测集 |
| `tests/unit/` | 增加 grader、schema、pass policy 和 isolation tests |
| `tests/integration/` | 增加并发、MCP degradation 和 end-to-end tests |
| `.github/workflows/ci.yml` | 增加分层 eval gates |

## 18. 完成定义

本次改造完成时，团队应能对任何一次失败快速回答：

1. Task 是否明确且可解？
2. Reference 是否经过验证？
3. Grader 是否正确识别了结果？
4. Trial 是否受到前次运行或共享资源影响？
5. 失败是否可稳定复现？
6. 失败属于 Agent、Task、Grader、MCP 还是环境？
7. 修复后是否已进入 regression suite？
8. 改进幅度是否超过已测得的噪声？

只有这些问题能够被证据化回答时，MAEDA 的 Eval 分数才适合用于模型选择、Prompt 更新、架构修改和发布决策。
