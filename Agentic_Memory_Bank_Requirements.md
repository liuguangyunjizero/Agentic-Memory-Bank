# Agentic Memory Bank: A Hierarchical Graph-Based Multi-Agent System for Task-Oriented Long-Context Management


## 1. 系统概述

### 1.1 设计目标

Agentic Memory Bank设计为Hierarchical Graph-based的结构，由Multi-Agent进行管理，目标是以任务为导向管理长上下文并提供结构化记忆。该系统提出一种可集成到不同外部Agent框架的记忆管理范式，实现长上下文的高效管理。

特别的，该Memory Agent的设计目标是**解决单次任务的Long-Context问题**，而非打造个性化的、从不断增长的记忆中学习的Long-Term Memory Agent。单次任务完成后，所有记忆将被清理。

### 1.2 应用场景

设计背景面向三种Long-Context场景：

1. **Deep Research**：多源信息检索、验证、综合分析
2. **Long-Document QA**：长文档理解与问答
3. **长对话/具身推理 QA**：多轮交互中的上下文管理

---

## 2. 系统架构

### 2.1 三层存储结构

Agentic Memory Bank采用三层结构来管理Long-Context：

#### 2.1.1 Insight Doc（任务状态层）

以精简的结构化形式管理任务执行状态，作为默认上下文传递给外部框架，随任务执行状态增量更新。

**存储内容**：

1. **任务总目标**：用户原始问题或任务描述

2. **已完成的子任务记录**：四元组列表 `[(t_i, d_i, s_i, c_i)]`
   - `t_i`：任务类型（如"搜索"用于信息检索、"解决"用于推理计算、"回答"用于最终输出）
   - `d_i`：任务详细描述，指导节点的执行目标
   - `s_i`：执行状态，通常为"成功"或"失败"
   - `c_i`：成功执行后的知识上下文（1-2句话的浓缩总结，包括检索到的信息或推理得出的见解）

3. **待办任务列表**：动态调整的任务列表，采用增量式规划策略，每次仅决定下一步要做什么

#### 2.1.2 Query Graph（语义记忆图）

以图结构存储结构化的记忆摘要，支持高效的粗粒度检索。

**节点粒度定义**：一个Query Graph节点对应一个主题簇的上下文经过【结构化Agent】处理后的内容。一个子任务整理一次记忆，由【分类/聚类Agent】决定把一个子任务的上下文拆分组合成几个主题类别。

**节点属性**：
- `id`：唯一标识符
- `summary`：结构化压缩后的详细摘要，包含关键信息、证据等
- `context`：简短的上下文主题描述（一句话概括）
- `keywords`：关键词列表，用于Attribute-Based检索
- `embedding`：语义向量（存储在内存中，用于相似度检索；可扩展到向量数据库存储索引）
- `timestamp`：创建时间

**边与关系状态**：

严格来说，Query Graph只有一种边类型：**related边（相关关系）**，表示两个节点在语义/主题/逻辑上有关联，无向边。

节点之间可能存在特殊的**关系状态**：
- **conflict状态**：两个节点的内容互相矛盾，会触发Cross Validation流程
- **merge状态**：两个节点描述相同或高度重叠的内容，会触发节点合并操作
- **related状态**：两个节点在语义/主题/逻辑上有关联，建立related边

这些关系状态不是边的类型，而是通过【记忆分析Agent】判断得出的节点间关系，用于触发相应的后续操作。

**关系判断优先级**：系统采用优先级机制判断节点间的关系：**conflict > merge > related**。每次都确保没有前一个情况再去处理下一种，具体流程如下：
1. **第一步**：优先检查是否存在conflict关系（内容互相矛盾）
2. **第二步**：如果不存在conflict，再检查是否存在merge关系（内容重叠相同）
3. **第三步**：如果既不存在conflict也不存在merge，才判断是否为related关系

检测到conflict或merge关系后，系统会立即执行相应处理，最终这些节点会被合并或验证。

**检索方式**：
- **Attribute-Based**：通过关键词匹配检索
- **Embedding-Based**：通过语义向量相似度检索
- **混合检索**：`Final_Score = α × Attribute-Based + (1-α) × Embedding-Based`（α为可配置参数，详见3.1.2节检索参数配置）

**检索参数**：
- `k`：top-k检索的k值（可配置参数，详见3.1.2节检索参数配置）
- 检索结果附带一层邻居节点（完整的id, summary, context和timestamp）

#### 2.1.3 Interaction Tree（交互历史层）

以只读的树形结构存储细粒度的原始交互数据，支持多模态信息。忠实记录上下文，不进行修改。

**结构特点**：
- 每个Query Graph节点对应一棵或多颗Interaction Tree
- 父节点（Text）：存储完整的上下文文本，包含推理过程、工具调用、返回结果的摘要等
- 子节点（多模态附件）：存储用户上传的、工具返回的一些原始多模态内容

**多模态附件设计**：

工具（如web_visit、pdf_reader等）返回两部分内容：
1. **摘要**：压缩后的文本描述，直接嵌入父节点
2. **附件**（可选）：重要的原始多模态内容，存储为子节点

附件类型：
- `image`：图片等
- `document`：PDF原文等
- `code`：代码片段、执行结果等

子节点属性：
- `id`：唯一标识符（在父节点文本中引用）
- `type`：多模态类型（image/document/code等）
- `content`：**文件路径**（存储多模态文件的磁盘路径或云存储URI，而非文件内容本身）

**存储策略**：有选择性地保留相关的、重要的多模态内容，而非全部保留。系统通过文件路径索引访问实际文件，避免在节点属性中存储大量二进制数据或文本内容，提高存储效率和系统性能。

---

## 3. 组件设计

### 3.1 硬编码功能模块

这些功能通过程序逻辑直接实现，不涉及LLM调用。

#### 3.1.1 Embedding计算模块

**职责**：为文本内容生成语义向量。

**输入**：文本（summary、context等）

**输出**：Embedding向量

**实现**：调用Embedding API（如OpenAI text-embedding-3）

**存储方案**：
- **MVP阶段**：直接存储在Query Graph节点的embedding属性中（内存）
- **可扩展方案**：存储到向量数据库（如Faiss、Chroma），节点仅保留embedding_id索引

#### 3.1.2 检索模块

**职责**：基于embedding相似度检索候选节点。

**输入**：
- 新节点的embedding
- 检索参数（k值、检索方式等）

**输出**：top-k候选节点列表（包含完整节点信息：id, summary, context, keywords, timestamp）

**实现**：
- 计算余弦相似度
- 混合检索（可选）：结合关键词匹配
- 返回top-k + 每个节点的一层邻居

**检索参数配置**：
- `k`：top-k检索的k值，**可配置参数**（无默认值，需在系统初始化时设置；常见范围参考：5-10）
- `α`：混合检索中的权重系数，**可配置参数**（用于计算混合分数：`Final_Score = α × Attribute-Based + (1-α) × Embedding-Based`；无默认值，需在系统初始化时设置；常见范围参考：0.3-0.7）

#### 3.1.3 图操作模块

**职责**：管理Query Graph的边和节点。

**功能**：
- 创建节点
- 删除节点
- 创建related边
- 删除边
- 查询节点的邻居
- 更新节点属性（context、keywords, embedding等）

**实现**：使用NetworkX或类似图库

#### 3.1.4 Context与Keywords更新模块

**职责**：根据【记忆分析Agent】的建议，更新节点的context和keywords属性。

**输入**：
- 节点id
- 新的context文本
- 新的keywords列表

**处理**：
1. 更新节点的context属性
2. 更新节点的keywords属性
3. 重新计算该节点的embedding（基于summary、context和keywords）
4. 更新存储

---

### 3.2 Agent功能模块

这些功能通过LLM进行智能判断和生成。

#### 3.2.1 【分类/聚类Agent】

**职责**：对长上下文按主题进行分类/聚类，生成每个分类的context和keywords描述。

**目的**：避免单次处理超长上下文，将上下文分块后传给【结构化Agent】。

**输入**：
- 长上下文文本（可能超出Agent窗口长度）
- 可选：当前任务描述（辅助分类决策）

**输出**：
```json
{
  "should_cluster": true,
  "clusters": [
    {
      "cluster_id": 1,
      "context": "量子计算硬件突破（IBM、Google等）",
      "content": "...(属于该主题的原始文本)...",
      "keywords": ["量子比特", "超导芯片", "量子纠错"]
    }
  ]
}
```

**特殊处理：超长上下文**

当输入上下文超过Agent窗口长度时，采用**分次加载**策略：

1. 计算每次加载大小：`chunk_size = agent_window_size × ratio`（ratio为可配置参数，无默认值，需在系统初始化时设置；常见参考：0.9，表示留10%余量）
2. 按段落/章节边界切分上下文为多个块
3. 依次处理每个块，每个块独立生成clusters
4. 所有clusters直接传给【结构化Agent】处理

**分类标准**：一般按主题分类。用户输入的原始问题应单独提取，作为任务总目标存入Insight Doc。

#### 3.2.2 【结构化Agent】

**职责**：对分类后的单个主题内容进行结构化压缩，生成summary摘要。

**输入**：
- 单个cluster的content
- cluster的context和keywords（参考）

**输出**：结构化摘要
```json
{
  "summary": "结构化的详细摘要（包含关键信息、证据、逻辑等）"
}
```

**后续硬编码处理**：
1. 生成节点id（UUID或递增ID）
2. 复制【分类/聚类Agent】提供的context和keywords
3. 读取当前时刻生成timestamp
4. 调用Embedding计算模块，为summary生成embedding
5. 组装完整节点并存入Query Graph

**优势**：每次只处理一个主题分类的上下文，避免窗口溢出。

#### 3.2.3 【记忆分析Agent】

**职责**：判断新节点与现有节点的关系，采用两次独立LLM调用的两阶段判断策略。

**两阶段判断策略说明**：
- **第一阶段**：进行第一次独立LLM调用，优先判断新节点与候选节点是否存在conflict（内容矛盾）或merge（内容重叠）关系
- **第二阶段**：仅当第一阶段判断结果为既无conflict也无merge时，进行第二次独立LLM调用，判断是否存在related（内容相关）关系

这样的设计确保了：(1) 高度矛盾/重叠的关系优先被发现和处理；(2) 两个阶段的判断逻辑相对独立，降低前一阶段判断对后续的影响；(3) 系统资源的合理使用（只在必要时进行第二次调用）。

**输入**：
- 新节点（summary、context、keywords）  ← **不包含embedding**
- 候选节点列表（summary、context、keywords）  ← **检索模块返回的**
- 判断类型：`["conflict", "merge"]` 或 `["related"]`

**输出**：关系列表
```json
[
  {
    "existing_node": "node_B",
    "relationship": "conflict",
    "confidence": 0.95,
    "reasoning": "节点A说CEO是张三，节点B说CEO是李四",
    "conflict_description": "CEO信息矛盾"
  },
  {
    "existing_node": "node_C",
    "relationship": "merge",
    "confidence": 0.90,
    "reasoning": "节点C包含A的所有信息且有更多细节",
    "merge_strategy": "保留C的完整信息，补充A的独特视角"
  },
  {
    "existing_node": "node_D",
    "relationship": "related",
    "confidence": 0.85,
    "reasoning": "两个节点都讨论量子计算，但侧重点不同",
    "context_update_new": "IBM量子芯片技术（属于量子计算硬件突破）",
    "context_update_existing": "量子计算硬件突破（包含IBM芯片技术）",
    "keywords_update_new": ["IBM", "量子芯片", "硬件"],
    "keywords_update_existing": ["量子计算", "硬件突破", "IBM芯片"]
  }
]
```

**关系类型定义**：

1. **unrelated**：无关，两个节点在语义/主题/逻辑上无关联
2. **related**：相关，两个节点在语义/主题/逻辑上有关联，建立related边
3. **conflict**：冲突状态，两个节点的内容互相矛盾，触发Cross Validation流程
4. **merge**：合并状态，两个节点描述相同或高度重叠的内容，触发节点合并操作

**判断标准**：由Agent基于prompt规则自主判断，返回JSON格式的结构化输出。

**后续硬编码处理**（根据Agent输出）：
- **related**：调用图操作模块创建related边；调用Context与Keywords更新模块更新两个节点的context和keywords
- **conflict**：标记conflict状态，传递给【计划Agent】
- **merge**：调用【记忆整合Agent】

#### 3.2.4 【记忆整合Agent】

**职责**：基于多个节点的内容，生成整合后的新节点内容，并为继承的邻居节点提供context和keywords更新建议。

**触发场景**：
1. 【记忆分析Agent】判断为merge关系
2. 【记忆分析Agent】判断为conflict关系，经Cross Validation后需要合并

**输入**：
- 待合并的节点列表（可以是2个或多个节点的id, summary, context, keywords）
- 每个节点的邻居列表（id, context, keywords）
- 触发原因：`merge` 或 `conflict`
- 如果是conflict，还包括外部框架返回的验证结果

**输出**：整合后的新节点内容及邻居更新建议
```json
{
  "summary": "整合后的详细摘要",
  "context": "综合的主题描述",
  "keywords": ["整合后的关键词列表"],
  "neighbor_updates": {
    "node_C": {
      "context": "更新后的context（反映与新节点的关系）",
      "keywords": ["更新后的keywords"]
    },
    "node_D": {
      "context": "更新后的context（反映与新节点的关系）",
      "keywords": ["更新后的keywords"]
    }
  },
  "interaction_tree_description": "综合了节点A、B的内容，基于验证结果修正了..."
}
```

**说明**：
- `neighbor_updates`包含所有继承邻居的新context和keywords
- 【记忆整合Agent】需要理解：
  - 被合并节点的原有邻居关系
  - 新节点与这些邻居的关系如何表述
  - 生成反映新关系的context和keywords描述

**后续硬编码处理**：
1. 生成新节点id和timestamp
2. 计算新节点的embedding（基于summary、context和keywords）
3. 继承所有待合并节点的边（去重：如果多个节点有共同邻居，只保留一条边）
4. 根据【记忆整合Agent】的neighbor_updates，逐一更新邻居节点：
   - 更新邻居节点的context和keywords属性
   - 重新计算邻居节点的embedding（基于summary、context和keywords）
5. 在Interaction Tree中记录合并操作（基于interaction_tree_description）
6. 删除所有待合并节点及其边

**邻居更新逻辑详解**：

合并操作会改变节点的identity（如节点A和B合并成H），因此需要更新所有继承邻居的context和keywords以反映新的关系。

**示例场景**：
```
节点A（新节点）："IBM量子芯片技术"，keywords: ["IBM", "量子芯片", "硬件"]，邻居：无
节点B（旧节点）："2020年量子计算进展"，keywords: ["量子计算", "2020", "进展"]，邻居：C、D
合并 → 节点H："2020-2023年量子计算综合报告（包含IBM芯片技术）"
```

**处理流程**：
1. 【记忆整合Agent】接收输入：
   - 节点A和B的信息
   - B的邻居信息：
     - 节点C：context = "量子纠缠研究（与2020年量子计算进展相关）"，keywords = ["量子纠缠", "研究", "2020"]
     - 节点D：context = "量子算法优化（与2020年量子计算进展相关）"，keywords = ["量子算法", "优化", "2020"]

2. 【记忆整合Agent】输出：
   - 新节点H的summary、context、keywords
   - 邻居更新建议：
     - 节点C：
       - 新context = "量子纠缠研究（与2020-2023年量子计算综合报告相关）"
       - 新keywords = ["量子纠缠", "研究", "2020-2023", "综合报告"]
     - 节点D：
       - 新context = "量子算法优化（与2020-2023年量子计算综合报告相关）"
       - 新keywords = ["量子算法", "优化", "2020-2023", "综合报告"]

3. 硬编码模块执行：
   - 创建节点H，继承B的边（H-C、H-D）
   - 更新节点C的context、keywords和embedding
   - 更新节点D的context、keywords和embedding
   - 删除节点A和B

4. 节点H重新进入检索流程：
   - 检索时**排除C、D**（因为已经是继承的邻居）
   - 只检索新的候选节点
   - 判断H与新候选节点的关系（可能还有conflict/merge）

**合并策略规则**：通过prompt预定义，如：
- 保留更新的信息（基于timestamp）
- 综合所有节点优势，补充缺失信息
- 如果是conflict触发的合并，以验证结果为准
- 为继承的邻居生成合理的context和keywords更新（反映与新节点的关系）

#### 3.2.5 【计划Agent】

**职责**：分析任务目标和记忆，拟定和更新计划。

**输入**：
- Insight Doc（任务状态层）
- 新加入的Query Graph记忆节点（summary、context、keywords）
- 可选：conflict/merge通知

**输出**：更新后的Insight Doc
```json
{
  "task_goal": "研究量子计算在2023年的最新突破...",
  "completed_tasks": [
    {
      "type": "搜索",
      "description": "搜索量子计算硬件突破",
      "status": "成功",
      "context": "找到IBM和Google的量子芯片信息"
    }
  ],
  "pending_tasks": [
    "验证IBM和Google的量子比特数冲突",
    "搜索量子算法进展"
  ]
}
```

**策略**：
- **增量式规划**：每次只决定下一步要做什么，参考完整历史但不重新规划全部
- **动态调整**：根据新记忆和任务执行结果，灵活调整待办任务
- **冲突响应**：当检测到conflict关系时，插入高优先级的Cross Validation任务

**终止判断**：
- 如果待办任务列表为空，且所有子任务状态为"成功"，系统终止
- 否则继续执行循环

**可选优化**：采用SFT、RL后的专用模型。

---

### 3.3 系统接口模块

#### 3.3.1 【适配器Adapter】

**职责**：作为Agentic Memory Bank与外部框架的接口层，负责上下文的拦截、增强和传递。

**设计定位**：Agentic Memory Bank提供一种记忆管理范式，外部Agent框架（如ReAct）通过【适配器Adapter】与该系统交互，实现长上下文的结构化管理。

**功能1：Prompt增强（初始化 + 每轮循环开始）**

增强流程：
1. 读取Insight Doc（获取当前任务和执行状态）
2. 调用检索模块，根据当前任务从Query Graph检索相关记忆：
   - 使用混合检索（`α × Attribute + (1-α) × Embedding`）
   - 返回top-k节点 + 每个节点的一层邻居
3. 组装增强Prompt：
   - Deep Retrieval工具声明和使用说明
   - Insight Doc完整内容
   - 检索到的相关Query Graph记忆（summary + context + keywords）
4. 传递给外部框架

**功能2：上下文拦截（每轮循环结束）**

拦截流程：
1. 外部框架完成当前任务，输出停止token
2. 【适配器】拦截任务执行的上下文
3. 判断任务类型：
   - 如果是Cross Validation任务：直接传给【记忆整合Agent】处理冲突
   - 如果是普通任务：传给【分类/聚类Agent】开始记忆更新流程

#### 3.3.2 【深入检索Deep Retrieval工具】

**职责**：供外部框架调用，读取特定Query Graph记忆的Interaction Tree完整内容。

**输入**：
- Query Graph节点id
- 可选：Interaction Tree子节点id（如果需要特定的多模态内容）

**输出**：
- Interaction Tree父节点的完整文本
- 如果指定子节点id，返回该子节点的完整内容

**使用场景**：
- 外部框架发现Query Graph的summary不够详细
- 需要查看原始推理过程
- 需要访问图片、PDF等多模态子节点

---

## 4. 数据流

### 4.1 初始化阶段

1. **用户输入**：Prompt（包含上下文+问题，上下文可能为空）

2. **【适配器Adapter】拦截**：捕获用户输入

3. **判断是否有上下文需要处理**：
   - 如果有上下文 → 进入步骤4
   - 如果无上下文（仅问题）→ 跳到步骤10

4. **【分类/聚类Agent】分类**：
   - 对上下文进行主题分类
   - 如果超长，采用分次加载（90%窗口）
   - 生成clusters列表，每个cluster包含context和keywords

5. **【结构化Agent】生成summary**：
   - 对每个cluster，生成结构化摘要

6. **硬编码：组装完整节点并存储**：
   - 生成节点id、复制context和keywords、生成timestamp
   - 调用Embedding计算模块，为summary生成embedding
   - 组装完整节点（id、summary、context、keywords、embedding、timestamp）
   - 存入内存

7. **硬编码：检索候选节点**：
   - 调用检索模块，基于新节点的embedding检索top-k候选节点

8. **【记忆分析Agent】判断关系**（两阶段）：

   **第一阶段：优先判断conflict/merge**
   - 输入：新节点（summary、context、keywords）+ 候选节点列表（summary、context、keywords）
   - 批量判断新节点与候选节点是否有conflict或merge关系
   - 处理结果：
     - **如果有conflict**：标记conflict关系，跳到步骤9（【计划Agent】会插入Cross Validation任务）
     - **如果有merge（且无conflict）**：
       - 调用【记忆整合Agent】生成整合后的新节点
       - 硬编码：计算新节点embedding（基于summary、context、keywords），继承边，更新邻居的context、keywords和embedding，删除旧节点
       - 递归回到步骤7（重新检索和判断）
     - **如果既无conflict也无merge**：进入第二阶段

   **第二阶段：判断related关系**
   - 批量判断新节点与候选节点的related关系
   - 硬编码：创建related边，调用Context与Keywords更新模块更新相关节点的context、keywords和embedding

9. **硬编码：创建Interaction Tree**：
   - 父节点：存储完整的原始上下文文本
   - 子节点：如果有多模态内容（图片、PDF等），有选择性地存储为子节点

10. **【计划Agent】分析任务**：
    - 输入：任务总目标 + 新生成的Query Graph节点（summary、context、keywords）或conflict通知
    - 输出：更新Insight Doc
      - 任务总目标
      - 已完成子任务（如果有）
      - 待办任务列表（决定下一步做什么）

11. **【适配器Adapter】增强Prompt**：
    - 读取Insight Doc
    - 硬编码：调用检索模块，检索当前待办任务相关Query Graph记忆（混合检索，top-k + 一层邻居）
    - 组装：Deep Retrieval工具声明 + Insight Doc + 相关记忆
    - 传入外部框架

### 4.2 执行循环

12. **外部框架执行**：
    - 根据Insight Doc的待办任务，执行当前任务
    - 参考提供的Query Graph记忆
    - 必要时调用Deep Retrieval工具查看Interaction Tree完整内容
    - 完成后输出停止token

13. **【适配器Adapter】拦截上下文**：
    - 捕获外部框架的执行上下文

14. **判断任务类型**：

    **分支1：Cross Validation任务（冲突解决）**
    - 外部框架已完成多源信息验证，返回验证结果
    - 直接传给【记忆整合Agent】
    - 【记忆整合Agent】基于conflict触发原因和验证结果生成整合后的新节点
    - 硬编码处理：
      - 计算新节点embedding（基于summary、context、keywords）
      - 继承所有冲突节点的边（去重）
      - 更新所有邻居节点的context、keywords和embedding
      - 在Interaction Tree中记录合并操作
      - 删除所有冲突节点及其边
    - 新节点递归进入步骤7（重新检索和判断）
      - 检索时排除已知邻居（继承的边）
      - 判断与新候选节点的关系
      - 如果还有conflict/merge，递归处理
      - 直到只剩related关系

    **分支2：普通任务**
    - 执行步骤4-8：【分类/聚类Agent】→【结构化Agent】→ 硬编码计算embedding → 硬编码检索 → 【记忆分析Agent】（两阶段）→ 硬编码创建Interaction Tree
    - 更新Query Graph和Interaction Tree

15. **【计划Agent】更新Insight Doc**：
    - 将刚完成的任务加入已完成子任务列表
    - 基于新记忆，决定下一步任务
    - 更新待办任务列表

16. **判断是否终止**：
    - 如果待办任务列表为空 → 任务完成，系统终止
    - 否则 → 返回步骤11（【适配器Adapter】增强下一轮Prompt）

### 4.3 冲突处理机制（提高系统trustworthiness）

**触发条件**：【记忆分析Agent】在第一阶段判断中检测到conflict关系。

**处理流程**：

1. **标记冲突**：
   - 记录conflict关系状态（不建立特殊的边）
   - 将新节点暂时加入Memory Bank（等待验证）

2. **插入验证任务**：
   - 【计划Agent】分析冲突记忆
   - 在待办任务列表最前面插入高优先级任务：
     - 任务类型：Cross Validation
     - 任务描述：验证节点A和节点B的冲突（具体说明冲突点）
     - 执行方式：调用Search等工具检索多源信息进行交叉验证

3. **外部框架执行验证**：
   - 外部框架在下一轮循环中执行Cross Validation任务
   - 搜索权威来源、对比多个信息源
   - 得出验证结论（哪个正确、如何整合等）

4. **解决冲突**：
   - 【适配器】拦截验证结果
   - 识别为Cross Validation任务，直接传给【记忆整合Agent】
   - 【记忆整合Agent】基于conflict触发原因和验证结果生成整合后的新节点
   - 硬编码：删除冲突节点，创建新节点，继承边，更新邻居
   - 生成新的整合节点，继续正常流程
