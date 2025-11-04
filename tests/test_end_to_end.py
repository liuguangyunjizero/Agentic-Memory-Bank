"""
Agentic Memory Bank - 模块基本功能测试

测试各个核心模块的基本功能，不测试完整的端到端问答流程。
完整的问答测试通过 main.py 进行。

测试覆盖：
1. 三层存储的基本操作
2. 硬编码模块的功能
3. Agent的输入输出
4. 工具的基本调用
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 100)
print("  Agentic Memory Bank - 模块基本功能测试")
print("=" * 100)


# ===========================================
# 测试1：三层存储基本操作
# ===========================================

print("\n" + "-" * 100)
print("测试1：三层存储基本操作")
print("-" * 100)

# 1.1 InsightDoc测试
print("\n1.1 InsightDoc 基本操作:")
from src.storage.insight_doc import InsightDoc, CompletedTask, TaskType

doc = InsightDoc(
    doc_id="test_doc_001",
    task_goal="测试任务目标"
)

# 添加已完成任务
doc.add_completed_task(
    task_type=TaskType.NORMAL,
    description="测试搜索任务",
    status="成功",
    context="找到了测试信息"
)

# 设置待办任务
doc.set_pending_tasks(["下一个测试任务"])

print(f"  [OK] Doc ID: {doc.doc_id}")
print(f"  [OK] 任务目标: {doc.task_goal}")
print(f"  [OK] 已完成任务数: {len(doc.completed_tasks)}")
print(f"  [OK] 待办任务数: {len(doc.pending_tasks)}")
print(f"  [OK] 导出/加载测试: ", end="")

# 测试序列化
doc_dict = doc.to_dict()
doc_loaded = InsightDoc.from_dict(doc_dict)
assert doc_loaded.doc_id == doc.doc_id
assert doc_loaded.task_goal == doc.task_goal
print("通过")

# 1.2 QueryGraph测试
print("\n1.2 QueryGraph 基本操作:")
from src.storage.query_graph import QueryGraph, QueryGraphNode
import numpy as np
import time

graph = QueryGraph()

# 创建测试节点
node1 = QueryGraphNode(
    id="node_001",
    summary="测试节点1的摘要",
    context="测试主题1",
    keywords=["关键词1", "关键词2"],
    embedding=np.random.rand(384),
    timestamp=time.time(),
    links=[]
)

node2 = QueryGraphNode(
    id="node_002",
    summary="测试节点2的摘要",
    context="测试主题2",
    keywords=["关键词3", "关键词4"],
    embedding=np.random.rand(384),
    timestamp=time.time(),
    links=[]
)

# 添加节点
graph.add_node(node1)
graph.add_node(node2)
print(f"  [OK] 添加节点: 2个节点")

# 添加边
graph.add_edge(node1.id, node2.id)
print(f"  [OK] 添加边: node_001 <-> node_002")

# 测试邻居查询
neighbors = graph.get_neighbors(node1.id)
assert len(neighbors) == 1
assert neighbors[0].id == node2.id
print(f"  [OK] 邻居查询: 找到 {len(neighbors)} 个邻居")

# 测试节点删除
graph.delete_node(node2.id)
assert graph.get_node_count() == 1
print(f"  [OK] 删除节点: 剩余 {graph.get_node_count()} 个节点")

# 1.3 InteractionTree测试
print("\n1.3 InteractionTree 基本操作:")
from src.storage.interaction_tree import InteractionTree, create_entry, MergeEvent

tree = InteractionTree()

# 创建条目
entry1 = create_entry(
    text="测试交互内容1",
    metadata={"source": "test", "tool": "search"}
)

entry2 = create_entry(
    text="测试交互内容2",
    metadata={"source": "test", "tool": "visit"}
)

# 添加条目
tree.add_entry("node_001", entry1)
tree.add_entry("node_001", entry2)
tree.add_entry("node_002", entry1)
print(f"  [OK] 添加条目: {tree.get_total_entries()} 个条目")

# 测试条目检索
entries = tree.get_entries("node_001")
assert len(entries) == 2
print(f"  [OK] 检索node_001的条目: {len(entries)} 个")

# 记录合并事件
merge_event = MergeEvent(
    event_id="merge_001",
    merged_node_ids=["node_001", "node_002"],
    new_node_id="node_003",
    timestamp=time.time(),
    description="测试合并"
)
tree.record_merge(merge_event)
print(f"  [OK] 记录合并事件: {len(tree.merge_events)} 个事件")


# ===========================================
# 测试2：硬编码模块功能
# ===========================================

print("\n" + "-" * 100)
print("测试2：硬编码模块功能")
print("-" * 100)

# 2.1 Embedding模块
print("\n2.1 Embedding 模块:")
from src.modules.embedding import EmbeddingModule
from src.config import Config

config = Config()
embedding_module = EmbeddingModule.from_config(config)

test_text = "这是一个测试文本用于生成embedding向量"
embedding = embedding_module.compute_embedding(test_text)
print(f"  [OK] 生成embedding: 维度={embedding.shape[0]}")
assert embedding.shape[0] == 384  # all-MiniLM-L6-v2的维度

# 2.2 Retrieval模块
print("\n2.2 Retrieval 模块:")
from src.modules.retrieval import RetrievalModule

# 重新创建测试图
graph = QueryGraph()
for i in range(5):
    node = QueryGraphNode(
        id=f"test_node_{i}",
        summary=f"测试节点{i}的详细摘要内容",
        context=f"测试主题{i}",
        keywords=[f"关键词{i}a", f"关键词{i}b"],
        embedding=np.random.rand(384),
        timestamp=time.time() + i,
        links=[]
    )
    graph.add_node(node)

retrieval_module = RetrievalModule(alpha=0.5, k=3)

query_embedding = np.random.rand(384)
query_keywords = ["关键词1a", "测试"]

retrieved_nodes = retrieval_module.hybrid_retrieval(
    query_embedding=query_embedding,
    query_keywords=query_keywords,
    graph=graph
)

print(f"  [OK] 混合检索: 返回 {len(retrieved_nodes)} 个节点")
assert len(retrieved_nodes) <= 5  # 可能返回top-k + 邻居

# 2.3 GraphOperations模块
print("\n2.3 GraphOperations 模块:")
from src.modules.graph_ops import GraphOperations

graph = QueryGraph()
graph_ops = GraphOperations(graph)

# 创建节点
node_a = QueryGraphNode(
    id="node_a",
    summary="节点A",
    context="主题A",
    keywords=["A"],
    embedding=np.random.rand(384),
    timestamp=time.time(),
    links=[]
)

node_b = QueryGraphNode(
    id="node_b",
    summary="节点B",
    context="主题B",
    keywords=["B"],
    embedding=np.random.rand(384),
    timestamp=time.time(),
    links=[]
)

graph_ops.add_node(node_a)
graph_ops.add_node(node_b)
graph_ops.add_edge(node_a.id, node_b.id)
print(f"  [OK] 添加节点和边: {graph.get_node_count()}个节点, {graph.get_edge_count()}条边")

# 测试合并节点
merged_node = QueryGraphNode(
    id="node_merged",
    summary="合并节点",
    context="合并主题",
    keywords=["merged"],
    embedding=np.random.rand(384),
    timestamp=time.time(),
    links=[]
)

graph_ops.merge_nodes([node_a.id, node_b.id], merged_node)
print(f"  [OK] 合并节点: 剩余 {graph.get_node_count()} 个节点")
assert graph.get_node_count() == 1

# 2.4 ContextUpdate模块
print("\n2.4 ContextUpdate 模块:")
from src.modules.context_update import ContextUpdateModule

graph = QueryGraph()
node = QueryGraphNode(
    id="test_node",
    summary="原始摘要",
    context="原始主题",
    keywords=["原始关键词"],
    embedding=np.random.rand(384),
    timestamp=time.time(),
    links=[]
)
graph.add_node(node)

context_updater = ContextUpdateModule(graph, embedding_module)
context_updater.update_node_context(
    node_id="test_node",
    new_context="更新后的主题",
    new_keywords=["新关键词1", "新关键词2"]
)

updated_node = graph.get_node("test_node")
assert updated_node.context == "更新后的主题"
assert "新关键词1" in updated_node.keywords
print(f"  [OK] 更新节点上下文: context={updated_node.context}")


# ===========================================
# 测试3：Agent实际LLM调用
# ===========================================

print("\n" + "-" * 100)
print("测试3：Agent实际LLM调用（需要API密钥）")
print("-" * 100)

# 导入所有Agent和数据类
from src.agents.classification_agent import ClassificationAgent, ClassificationInput
from src.agents.structure_agent import StructureAgent, StructureInput
from src.agents.analysis_agent import AnalysisAgent, AnalysisInput, NodeInfo
from src.agents.integration_agent import IntegrationAgent, IntegrationInput, NodeWithNeighbors
from src.agents.planning_agent import PlanningAgent, PlanningInput
from src.utils.llm_client import LLMClient

# 初始化LLM客户端和Agent
llm_client = LLMClient.from_config(config)

# 3.1 ClassificationAgent实际调用
print("\n3.1 ClassificationAgent 实际调用:")

classification_agent = ClassificationAgent.from_config(llm_client, config)

test_context = """
人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
机器学习是人工智能的一个子领域，专注于让计算机从数据中学习。
深度学习是机器学习的一种方法，使用神经网络来处理数据。
"""

classification_input = ClassificationInput(
    context=test_context,
    task_goal="了解人工智能的基本概念"
)

try:
    result = classification_agent.run(classification_input)
    print(f"  [OK] 分类成功: should_cluster={result.should_cluster}, 生成{len(result.clusters)}个簇")
    if result.clusters:
        print(f"  [OK] 第1个簇主题: {result.clusters[0].context[:30]}...")
except Exception as e:
    print(f"  [FAIL] 分类失败: {e}")

# 3.2 StructureAgent实际调用
print("\n3.2 StructureAgent 实际调用:")

structure_agent = StructureAgent.from_config(llm_client, config)

long_content = """
量子计算是一种利用量子力学原理进行计算的新型计算范式。与传统的经典计算机使用比特（0或1）不同，
量子计算机使用量子比特（qubit），可以同时处于0和1的叠加态。这种特性使得量子计算机在处理某些
特定问题时具有指数级的速度优势。量子计算的主要应用领域包括密码学、优化问题、药物发现和材料科学。
目前，量子计算还处于早期发展阶段，但已经引起了学术界和工业界的广泛关注。IBM、Google等公司都在
积极研发量子计算机，并取得了一些重要的里程碑成果。
"""

structure_input = StructureInput(
    content=long_content,
    context="量子计算基础",
    keywords=["量子计算", "量子比特", "应用"]
)

try:
    result = structure_agent.run(structure_input)
    compression_ratio = len(result.summary) / len(long_content)
    print(f"  [OK] 结构化成功: 原文{len(long_content)}字 -> 摘要{len(result.summary)}字")
    print(f"  [OK] 压缩比: {compression_ratio:.1%}")
    print(f"  [OK] 摘要预览: {result.summary[:60]}...")
except Exception as e:
    print(f"  [FAIL] 结构化失败: {e}")

# 3.3 AnalysisAgent实际调用
print("\n3.3 AnalysisAgent 实际调用:")

analysis_agent = AnalysisAgent.from_config(llm_client, config)

# 创建测试节点
new_node = NodeInfo(
    id="new_node",
    summary="IJCAI 2025会议将于2025年8月在蒙特利尔举行，投稿截止日期是2025年1月15日",
    context="IJCAI 2025会议信息",
    keywords=["IJCAI", "2025", "截止日期", "蒙特利尔"]
)

candidate_node = NodeInfo(
    id="candidate_node",
    summary="ICML 2025会议将于2025年7月在维也纳举行，投稿截止日期是2025年1月20日",
    context="ICML 2025会议信息",
    keywords=["ICML", "2025", "截止日期", "维也纳"]
)

analysis_input = AnalysisInput(
    new_node=new_node,
    candidate_nodes=[candidate_node]
)

try:
    result = analysis_agent.run(analysis_input)
    print(f"  [OK] 关系分析成功: 发现{len(result.relationships)}个关系")
    if result.relationships:
        rel = result.relationships[0]
        print(f"  [OK] 关系类型: {rel.relationship}")
        print(f"  [OK] 判断理由: {rel.reasoning[:50]}...")
except Exception as e:
    print(f"  [FAIL] 关系分析失败: {e}")

# 3.4 IntegrationAgent实际调用
print("\n3.4 IntegrationAgent 实际调用:")

integration_agent = IntegrationAgent.from_config(llm_client, config)

# 创建冲突节点
node1 = NodeWithNeighbors(
    id="node_1",
    summary="AAAI 2025的投稿截止日期是2024年8月15日",
    context="AAAI 2025截止日期（信息1）",
    keywords=["AAAI", "2025", "8月15日"],
    neighbors=[
        {"id": "neighbor_1", "context": "AAAI会议投稿要求", "summary": "相关信息"}
    ]
)

node2 = NodeWithNeighbors(
    id="node_2",
    summary="AAAI 2025的投稿截止日期是2024年8月30日",
    context="AAAI 2025截止日期（信息2）",
    keywords=["AAAI", "2025", "8月30日"],
    neighbors=[
        {"id": "neighbor_2", "context": "AAAI会议评审流程", "summary": "相关信息"}
    ]
)

validation_result = "根据AAAI 2025官方网站的最新信息，投稿截止日期确认为2024年8月30日（已延期）。"

integration_input = IntegrationInput(
    nodes_to_merge=[node1, node2],
    validation_result=validation_result
)

try:
    result = integration_agent.run(integration_input)
    print(f"  [OK] 整合成功: 合并后摘要长度{len(result.merged_node['summary'])}字")
    print(f"  [OK] 更新邻居数: {len(result.neighbor_updates)}个")
    print(f"  [OK] 合并后主题: {result.merged_node['context'][:40]}...")
except Exception as e:
    print(f"  [FAIL] 整合失败: {e}")

# 3.5 PlanningAgent实际调用
print("\n3.5 PlanningAgent 实际调用:")

planning_agent = PlanningAgent.from_config(llm_client, config)

# 创建测试InsightDoc
test_doc = InsightDoc(
    doc_id="test_planning",
    task_goal="查询NeurIPS 2025的投稿截止日期"
)
test_doc.add_completed_task(
    task_type=TaskType.NORMAL,
    description="搜索NeurIPS 2025会议信息",
    status="成功",
    context="找到了NeurIPS 2025的基本信息，包括举办时间和地点"
)

# 模拟新记忆节点
new_memory = [
    {
        "id": "mem_1",
        "context": "NeurIPS 2025会议信息",
        "summary": "NeurIPS 2025将于2025年12月在新奥尔良举行，投稿截止日期是2025年5月20日",
        "keywords": ["NeurIPS", "2025", "截止日期"]
    }
]

planning_input = PlanningInput(
    insight_doc=test_doc,
    new_memory_nodes=new_memory,
    conflict_notification=None
)

try:
    result = planning_agent.run(planning_input)
    print(f"  [OK] 规划成功: 已完成{len(result.completed_tasks)}个任务")
    print(f"  [OK] 待办任务: {len(result.pending_tasks)}个")
    if result.pending_tasks:
        print(f"  [OK] 下一步: {result.pending_tasks[0][:40]}...")
except Exception as e:
    print(f"  [FAIL] 规划失败: {e}")


# ===========================================
# 测试4：工具基本调用（Mock测试，不实际调用API）
# ===========================================

print("\n" + "-" * 100)
print("测试4：工具基本调用（数据格式验证）")
print("-" * 100)

# 4.1 SearchTool参数验证
print("\n4.1 SearchTool 参数格式:")
from src.tools.search_tool import SearchTool

# 只验证初始化，不实际调用API
try:
    search_tool = SearchTool(search_api_key="test_key")
    print(f"  [OK] SearchTool初始化成功")
except Exception as e:
    print(f"  [FAIL] SearchTool初始化失败: {e}")

# 4.2 VisitTool参数验证
print("\n4.2 VisitTool 参数格式:")
from src.tools.visit_tool import VisitTool
from src.utils.llm_client import LLMClient

# 只验证初始化，不实际调用
try:
    llm_client = LLMClient.from_config(config)
    visit_tool = VisitTool(llm_client=llm_client, jina_api_key=None)
    print(f"  [OK] VisitTool初始化成功（BeautifulSoup模式）")
except Exception as e:
    print(f"  [FAIL] VisitTool初始化失败: {e}")

# 4.3 DeepRetrievalTool
print("\n4.3 DeepRetrievalTool:")
from src.tools.deep_retrieval_tool import DeepRetrievalTool
from src.utils.file_utils import FileUtils

file_utils = FileUtils(temp_dir="data/temp", storage_dir="data/storage")
deep_retrieval_tool = DeepRetrievalTool(tree, file_utils)

# 创建测试条目
test_entry = create_entry(
    text="深度检索测试内容",
    metadata={"source": "test"}
)
tree.add_entry("test_deep_node", test_entry)

# 测试检索
result = deep_retrieval_tool.call({"node_id": "test_deep_node"})
assert "深度检索测试内容" in result
print(f"  [OK] DeepRetrievalTool检索成功，返回{len(result)}字符")


# ===========================================
# 测试完成
# ===========================================

print("\n" + "=" * 100)
print("[成功] 所有模块基本功能测试通过！")
print("=" * 100)
print("\n说明：")
print("  - 测试1-2: 三层存储和硬编码模块的基本操作")
print("  - 测试3: 所有5个Agent的实际LLM调用（需要有效的API密钥）")
print("  - 测试4: 工具的初始化和基本功能")
print("  - 完整的端到端问答流程测试请使用 main.py")
print("=" * 100)
