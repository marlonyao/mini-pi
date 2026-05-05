# mini-pi 开发路线图

## 当前状态

mini-pi 是一个极简的 Python coding agent，已完成：
- ✅ Agent loop（LLM 调用 → 工具执行 → 循环）
- ✅ 5 个核心工具：bash, read, write, edit, grep
- ✅ JSONL session 持久化
- ✅ 流式输出
- ✅ OpenAI 兼容 API 支持

---

## Phase 1: 上下文压缩（Context Compaction）

> 参考 Pi 的三层压缩策略，移植到 mini-pi

### 1.1 Session Pruning（轻量级，优先实现）

**目标**：裁剪旧的工具调用结果，减少上下文膨胀

**实现思路**：
- 新增 `context.py` 模块，负责上下文组装
- 在 `agent.py` 的 `_stream_llm()` / `_call_llm_sync()` 组装消息前，执行 pruning
- 逻辑：
  1. 遍历 session.messages，找到 role=tool 的旧消息
  2. 超过 N 轮的 tool result：
     - **软裁剪**：保留头尾，中间替换为 `[...truncated...]`
     - **硬清除**：直接替换为 `[tool output removed - older than N turns]`
  3. 对话文本（user/assistant）不动

**配置项**：
```python
@dataclass
class PruningConfig:
    enabled: bool = True
    keep_recent_turns: int = 3    # 保留最近 N 轮的 tool result 完整
    soft_trim_chars: int = 500    # 超过此长度的 tool result 做软裁剪
    max_tool_result_chars: int = 2000  # 单个 tool result 最大字符数
```

### 1.2 Compaction（核心压缩）

**目标**：当对话接近模型上下文窗口限制时，用 LLM 做摘要压缩

**实现思路**：
- 新增 `compactor.py` 模块
- 在 `agent.py` 中，每次调用 LLM 前检查 token 估算
- 如果接近上限 → 触发 compaction

**Compaction 流程**：
```
1. 计算当前消息总 token 数（可以用 tiktoken 或简单估算）
2. 如果超过阈值（如 max_tokens * 0.8）：
   a. 分割消息：旧消息 vs 最近 N 轮
   b. 把旧消息发给 LLM 做摘要（用专门的 compaction prompt）
   c. 在 session 中用一个特殊的 "compaction" entry 替换旧消息
   d. 保留最近消息不动
3. 完整历史保存在 JSONL（新增 type=compaction entry）
4. 后续 get_openai_messages() 只返回摘要 + 最近消息
```

**Compaction Prompt 示例**：
```
Summarize the following conversation history into a concise summary.
Preserve:
- Key decisions and their rationale
- Important code changes made
- File paths that were modified
- Any unresolved issues or open questions
- Current task state and what's left to do

Do NOT preserve:
- Verbose tool outputs
- Exploration steps that didn't lead anywhere
- Repeated attempts at the same thing

Output a structured summary in markdown.
```

**Session 修改**：
```python
# session.py 新增
class Session:
    def compact(self, summary: str, keep_recent: int = 6) -> None:
        """Replace older messages with a compaction summary."""
        # 在 messages 前插入一个特殊的 compaction marker
        # 保留最后 keep_recent 条消息
        # 完整历史通过 type=compaction 记录在 JSONL
        ...

    def get_openai_messages(self) -> list[dict]:
        """返回组装后的消息（已压缩/修剪的版本）"""
        ...
```

**配置项**：
```python
@dataclass
class CompactionConfig:
    enabled: bool = True
    model: str | None = None       # 可指定不同模型做摘要（省钱/快）
    threshold: float = 0.8         # 上下文使用率阈值
    keep_recent_messages: int = 6  # 压缩时保留最近 N 条消息
    max_context_tokens: int = 128000  # 模型上下文窗口大小
```

### 1.3 Token 估算

**实现方式**：
- 优先用 `tiktoken`（如果安装了）
- 降级方案：简单估算 `len(text) / 3.5`（英文）或 `len(text) / 2`（中文）
- 在 config 中加 `max_context_tokens` 配置

### 1.4 手动触发

- REPL 中输入 `/compact` 手动触发压缩
- 可选带提示词：`/compact 重点关注 API 设计决策`

---

## Phase 2: Skill 支持

> 参考 Pi/OpenClaw 的 skill 机制，实现可扩展的工具/知识系统

### 2.1 Skill 目录结构

```
~/.mini-pi/skills/
└── my-skill/
    ├── SKILL.md          # Skill 描述 + 使用说明（agent 读取此文件决定何时使用）
    ├── tools.py          # 可选：skill 提供的工具定义
    └── references/       # 可选：参考文档、示例代码等
        └── example.py
```

### 2.2 Skill 加载机制

**新增 `skills.py` 模块**：

```python
class Skill:
    """A skill is a directory with a SKILL.md and optional tool definitions."""
    name: str
    description: str          # 从 SKILL.md 提取
    skill_dir: Path
    tools: list[ToolDef]      # 可选的工具定义
    system_prompt_addition: str  # SKILL.md 的内容，按需注入

class SkillManager:
    """ discovers and loads skills from skill directories."""
    def __init__(self, skill_dirs: list[str]):
        ...

    def discover(self) -> list[Skill]:
        """Scan skill directories and load all valid skills."""

    def match(self, user_message: str) -> Skill | None:
        """Find the most relevant skill for the current task."""

    def get_tools(self) -> list[dict]:
        """Get OpenAI tool definitions from all active skills."""

    def execute_tool(self, name: str, args: dict) -> str:
        """Execute a skill-provided tool."""
```

### 2.3 Skill 注册方式

**方式一：目录扫描**（推荐）
- 扫描 `~/.mini-pi/skills/` 和项目内 `.mini-pi/skills/`
- 每个含 `SKILL.md` 的子目录就是一个 skill

**方式二：配置注册**
```python
# config.py 或 pyproject.toml
skills:
  - path: ~/.mini-pi/skills/my-skill
  - path: ./skills/code-review
```

### 2.4 Skill 中的工具定义

```python
# skills/my-skill/tools.py
from mini_pi.skills import register_tool
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="Search query")

@register_tool
def search_code(params: SearchParams, **kwargs) -> str:
    """Search code using semantic understanding."""
    ...
```

### 2.5 System Prompt 集成

Skill 的 `SKILL.md` 内容按需注入到 system prompt：

```
When considering which skill to use, scan the available skill descriptions.
If exactly one skill clearly applies: read its SKILL.md, then follow it.
```

### 2.6 内置 Skills 示例

可以随 mini-pi 附带几个示例 skill：
- `code-review`: 代码审查 skill
- `git-workflow`: Git 操作工作流
- `testing`: 测试生成和运行

---

## Phase 3: 增强（可选）

### 3.1 Memory 系统
- 日志文件：`~/.mini-pi/memory/YYYY-MM-DD.md`
- 长期记忆：`~/.mini-pi/MEMORY.md`
- Memory search：简单的关键词/向量搜索

### 3.2 多模型支持
- Compaction 可用更便宜的模型
- 不同 task 用不同模型（编码 vs 摘要 vs 审查）

### 3.3 子 Agent
- 将复杂任务拆分给子 agent
- 参考 Pi 的 subagent spawn 机制

---

## 实现优先级

| 顺序 | 任务 | 复杂度 | 价值 |
|------|------|--------|------|
| 1 | Session Pruning | ⭐ | 🔥🔥🔥 延长会话寿命 |
| 2 | Compaction | ⭐⭐⭐ | 🔥🔥🔥 长对话不崩 |
| 3 | Token 估算 | ⭐ | 🔥🔥 压缩的前提 |
| 4 | Skill 框架 | ⭐⭐ | 🔥🔥🔥 可扩展性 |
| 5 | Skill 工具注册 | ⭐⭐ | 🔥🔥 动态工具 |
| 6 | 内置 Skills | ⭐ | 🔥 示例和开箱即用 |

---

## Pi 上下文压缩机制总结（供参考）

Pi/OpenClaw 的三层防护：
```
Pruning（轻量，每次请求前）
  → 裁剪旧 tool result，不动对话文本
  → 纯内存操作，不改磁盘

Compaction（中等，接近上限时）
  → LLM 摘要旧对话
  → 保存到 session transcript
  → 可指定不同模型做摘要

Memory Flush（压缩前）
  → 先把重要信息写入文件
  → 防止摘要时丢失关键上下文
```

Pi 的扩展机制（Context Engine）：
- 通过 `plugins.slots.contextEngine` 可插拔
- 引擎接口：`ingest() → assemble() → compact() → afterTurn()`
- 默认 `legacy` 引擎用内置摘要
- 可安装插件引擎（如 `lossless-claw`）用 DAG/向量检索等策略
