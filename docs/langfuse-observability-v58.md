# PaperStorm v5.8 Langfuse 可观测性设计与学习记录

## 1. 为什么接入 Langfuse

PaperStorm 原有 `paperstorm_trace.jsonl` 擅长描述单次调研内部阶段，但 Research、Chat、
Benchmark 使用了不同产物，难以按会话、版本和环境统一比较。v5.8 增加可插拔观测层，
将三类执行统一为 `Trace -> Observation -> Score`，同时保留本地 JSONL 作为审计底座。

Langfuse 不是 Agent runtime，也不替代 LangGraph checkpoint、业务数据库或 Benchmark
脚本。它负责跨运行追踪、指标归集、失败定位和版本比较；PaperStorm runtime 仍负责执行、
重试、状态持久化与恢复。

## 2. 数据模型

| 业务动作 | Trace | 子 Observation | Score |
| --- | --- | --- | --- |
| 论文调研 | `paperstorm.research` | `research_pipeline` | `run_success`、`run_score` |
| 一轮聊天 | `paperstorm.chat` | 实际执行的 LangGraph 节点 | `trajectory_success`、`retrieval_triggered` |
| 公开评测 | `paperstorm.benchmark` | 后续可扩展为逐样本 span | `metrics.json` 数值叶子、`run_success` |

`chat_id` 映射为 Langfuse session，伪匿名用户 ID 映射为 user，版本、运行模式、检索器和
环境写入 metadata/tags。这样可以回答：某版本在哪条路由变慢、哪类问题频繁触发检索、
质量提升是否伴随延迟或成本回归。

## 3. 可靠性与隐私

1. **Fail-open**：Langfuse 未安装、凭据缺失、网络中断或 exporter 抛异常时，不改变
   Agent 的业务返回。
2. **本地镜像**：事件始终尝试写入
   `<service-root>/observability/events.jsonl`；本地写盘失败同样降级，不阻断主链路。
3. **递归脱敏**：密钥、Authorization、Cookie、密码和访问令牌替换为掩码；用户标识
   使用 SHA-256 稳定伪匿名；长文本截断。
4. **诚实状态**：SDK 与凭据就绪显示“已配置”，不把异步客户端创建误报成采集端网络
   已连通。同步异常和本地写盘异常显示“降级”。
5. **进程退出 flush**：FastAPI shutdown 时刷新 SDK 批次，减少短任务尾部 Trace 丢失。

## 4. 如何用 Langfuse 评估 Harness

不要只看“调用是否成功”。建议在 Langfuse 中按 `version`、`environment`、
`run_mode`、`retriever` 分组，建立四类指标：

| 维度 | 代表指标 | 用途 |
| --- | --- | --- |
| 任务正确性 | Answer F1、Evidence Recall、nDCG、run score | 判断答案与证据质量 |
| 轨迹正确性 | trajectory success、错误工具率、无效检索率 | 判断 Agent 是否走对路径 |
| 运行效率 | P50/P95 latency、模型 tokens、检索次数 | 定位慢节点与成本来源 |
| 稳定性 | success rate、retry rate、export failures | 判断是否达到生产运行要求 |

公开 Benchmark 仍由 SciFact、QASPER、LongMemEval 等固定数据与 evaluator 产生权威指标，
Langfuse 负责保存每次运行及版本对比。在线真实问题可增加人工评分或 LLM-as-a-Judge，
但必须与确定性指标分栏，记录 Judge 模型和 Prompt 版本，不能把主观分数伪装成标准答案。

## 5. 面试可以怎么讲

> 我没有把 Langfuse 当作日志 SDK 随手埋点，而是在 Agent runtime 外定义统一的
> Trace/Observation/Score 契约。论文调研、聊天路由和公开 Benchmark 都进入同一观测
> 模型；本地 JSONL 是审计事实源，Langfuse 是可选分析后端。实现采用 fail-open、递归
> 脱敏、用户伪匿名和 shutdown flush，采集系统故障不会扩大为业务故障。评测上把
> Retrieval、Answer、Trajectory、Latency 分开，支持按版本和环境比较。

## 6. 当前边界

- v5.8 已完成 Trace、嵌套 Observation、Score、状态接口、本地镜像与 Langfuse 双写。
- Langfuse SDK 异步发送，当前只能将客户端就绪标记为“已配置”；端到端 collector
  健康检查需要结合部署网络和 Langfuse 服务端指标。
- Benchmark 当前上报聚合指标。若要分析单题失败分布，可在后续将每条公开数据集样本
  映射为 Langfuse Dataset Item/Experiment Run，但不应替代官方 evaluator。
- 默认不安装 Langfuse；使用者需显式安装 `.[observability]` 并配置项目凭据。
