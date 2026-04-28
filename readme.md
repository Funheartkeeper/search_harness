# s13: Search Harness (搜索 Agent)

`s01 > s02 > s03 > s04 > s05 > s06 | s07 > s08 > s09 > s10 > s11 > s12 > [ s13 ]`

> *"先把问题变成可搜索的证据任务"* -- 用户说的是自然语言，harness 要交给模型的是带上下文的意图、Query 和证据包。
>
> **Harness 层**: 搜索编排 -- 多轮意图识别、Query 改写、搜索边界和证据整理。

## 问题

到 s12，Agent 已经能使用工具、管理上下文、拆任务、后台执行、协作和隔离目录。但它面对外部世界时还有一个新问题：用户的问题不是天然适合搜索的 Query。

用户会这样问：

- “那它现在支持 structured output 吗？”
- “第二个方案有没有更新？”
- “这个库和 LangChain 比现在哪个好？”
- “我刚才说的那个政策最新版本是什么？”

这些问题有四类难点：

1. **多轮省略。** “它”“第二个方案”“那个政策”都需要从会话状态里解析。
2. **Query 噪声。** 用户可能有错别字、拼写错误、中英混输。
3. **搜索决策。** 有的问题需要联网，有的问题应该直接基于本地上下文回答。
4. **证据整理。** 搜索结果不能直接丢给模型，需要先去重、标注来源、保留 query trace。

## 解决方案

```
User Message
  -> ConversationState
  -> Intent
  -> QueryPlan
  -> SearchProvider
  -> EvidencePack
  -> Final Answer
```

s13 不把搜索写成一棵固定流程树，而是把每一步变成结构化对象：

- `Intent`：用户到底想问什么，是否需要搜索。
- `QueryPlan`：纠错后的问题、独立问题、搜索 Query 列表。
- `SearchProvider`：可替换的搜索边界，默认 fake，可配置 Tavily。
- `EvidencePack`：统一证据包，包含原问题、改写、查询、结果和备注。
- `ConversationState`：跨轮保存实体、约束和最近搜索。

## 工作原理

1. **维护会话状态。**

```python
state = ConversationState()
state.active_entities.append("Anthropic Messages API")
```

当用户下一轮问“那它现在支持 structured output 吗？”，harness 不直接搜索这句话，而是先把上下文交给意图识别和 Query 改写。

2. **识别意图。**

```python
intent = classify_intent(user_message, state)
# -> need_search=True
# -> intent_type="current_fact"
# -> time_sensitivity="high"
```

模型优先输出结构化 JSON。没有模型或模型失败时，harness 退回到启发式分类，保证示例可以离线测试。

3. **纠错和改写 Query。**

```python
query_plan = rewrite_query(user_message, state, intent)
# corrected_question: "那它现在支持 structured output 吗？"
# standalone_question: "Anthropic Messages API: 那它现在支持 structured output 吗？"
# search_queries: [...]
```

这一步解决的是“用户话语”和“搜索 Query”之间的落差。多轮指代、拼写错误、当前年份、时效窗口都在这里处理。

4. **通过 provider 搜索。**

```python
provider = make_search_provider()
results, notes = search_with_plan(query_plan, provider)
```

第一版提供两个 provider：

- `FakeSearchProvider`：默认 provider，用于测试和离线演示。
- `TavilySearchProvider`：设置 `SEARCH_PROVIDER=tavily` 和 `TAVILY_API_KEY` 后启用真实搜索。

5. **整理证据包。**

```python
evidence = build_evidence_pack(
    original_question=user_message,
    intent=intent,
    query_plan=query_plan,
    results=results,
    notes=notes,
)
```

搜索结果会先经过 URL 规范化、标题去重和同域名数量限制。最终回答不是基于原始搜索响应，而是基于整理后的 evidence pack。

6. **生成答案并更新记忆。**

```python
answer = synthesize_answer(user_message, state, intent, query_plan, evidence)
state.remember_search(query_plan, results)
```

答案生成规则强调：

- 优先回答用户问题。
- 引用来源 URL。
- 区分证据和推断。
- 结果不足时说明不足。

## 相对 s12 的变更

| 组件               | 之前 (s12)                       | 之后 (s13)                                  |
|--------------------|----------------------------------|---------------------------------------------|
| 外部信息           | 主要依赖本地工具和文件           | 可通过搜索 provider 接触网络资源           |
| 用户问题           | 直接进入 agent loop              | 先转成 Intent 和 QueryPlan                  |
| 多轮指代           | 依赖完整 messages                | `ConversationState.active_entities` 显式保存 |
| 工具结果           | 直接作为 tool_result             | 先归一化为 `EvidencePack`                   |
| 可测试性           | 编译和部分管理器测试             | fake provider + 纯函数测试                  |
| 可复盘性           | 任务/事件日志                    | query trace: 原问题 -> 改写 -> 搜索 -> 答案 |

## 试一试

默认离线 fake 搜索：

```sh
python s13_search_harness.py
```

可以试这些 prompt：

1. `Anthropic Messages API 是什么？`
2. `那它现在支持 strutured ouput 吗？`
3. `这个仓库的 agents 目录是做什么的？`
4. `比较一下 LangChain 和 LlamaIndex 现在适合做搜索 agent 吗？`

启用真实 Tavily Search：

```sh
export SEARCH_PROVIDER=tavily
export TAVILY_API_KEY=your_api_key
python s13_search_harness.py
```

如果要查看 intent 和 query plan：

```sh
export SEARCH_DEBUG=1
python s13_search_harness.py
```

## 这一章的核心心智

s13 的重点不是“调用一个搜索 API”。真正新增的 harness 能力是：

- 把多轮对话里的模糊问题变成独立问题。
- 把自然语言问题变成可复盘的 query plan。
- 把搜索供应商封装成可替换边界。
- 把杂乱搜索结果整理成证据包。
- 让最终回答建立在证据之上，而不是建立在模型记忆之上。

搜索 Agent 的稳定性，不来自一个更长的 prompt，而来自 harness 对“意图、查询、证据、来源”的结构化管理。
