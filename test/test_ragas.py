"""
RAGAS 评估测试文件 - 测试 app.py RAG 系统的检索与生成质量

用法:
    python test/test_ragas.py                    # 运行完整评估
    python test/test_ragas.py --verbose          # 打印每个问题的详细结果
    python test/test_ragas.py --output report.json  # 保存 JSON 报告

评估指标:
    - Faithfulness (忠实度):     回答是否基于检索到的上下文，有无编造
    - AnswerRelevancy (答案相关性): 回答是否切题
    - ContextPrecision (上下文精度): 检索到的上下文是否相关
    - ContextRecall (上下文召回):   检索上下文是否覆盖参考答案
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# ========== DeepSeek 云模型配置 ==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "") #<-- 填写你的 DeepSeek API Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"   # DeepSeek V4 模型

# 设置 HuggingFace 镜像（必须在其他 import 之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'

# 添加父目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========== 检查依赖 ==========
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_ollama import OllamaEmbeddings
    from datasets import Dataset
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: pip install ragas datasets langchain-openai langchain-ollama")
    sys.exit(1)

# ========== 测试数据集 ==========
# 基于 docs/ 两份文档的 QA 对（5 道 ML + 5 道 DL）
# ground_truth 为文档中可直接找到的参考答案
TEST_DATA = [
    # ====== 机器学习部分（机器学习原理与算法.docx）====== #
    {
        "question": "什么是监督学习和无监督学习？它们的核心区别是什么？",
        "ground_truth": "监督学习的训练数据包含输入特征与真实标签，模型学习从特征到标签的映射关系，用于预测未知样本的输出结果，核心分为分类任务（输出离散类别）和回归任务（输出连续数值）。无监督学习的训练数据仅包含输入特征，无任何标签信息，模型自主挖掘数据内部结构、分布规律与关联关系，核心任务包含聚类、降维、关联规则挖掘、密度估计等。两者的核心区别在于训练数据是否包含标签信息。"
    },
    {
        "question": "岭回归和Lasso回归有什么区别？各适用于什么场景？",
        "ground_truth": "岭回归（Ridge）在线性回归损失函数中增加L2正则惩罚项，约束权重参数大小，避免参数过大导致过拟合，同时解决矩阵不可逆的共线性问题，所有权重参数平滑缩小、无参数归零、保留全部特征。Lasso回归引入L1绝对值惩罚项，可使大量无效特征权重直接归零，实现自动特征筛选，适用于高维稀疏特征数据集。缺点是损失函数不可导，无解析解，仅能通过迭代优化求解。弹性网（Elastic Net）融合L1与L2正则优势，同时具备特征筛选与权重平滑能力，是工业界最优的正则化回归算法。"
    },
    {
        "question": "SVM中硬间隔和软间隔有什么区别？惩罚系数C有什么作用？",
        "ground_truth": "硬间隔SVM仅适用于完全线性可分的数据，要求所有样本被正确分类且满足最大间隔约束。软间隔SVM针对真实数据中普遍存在的噪声与重叠样本，通过引入松弛变量与惩罚系数C，允许少量样本错分或进入间隔内部，平衡间隔最大化与分类准确率。惩罚系数C为超参数：C越大，对错分样本的惩罚越重，模型越倾向于拟合训练集，间隔越小，容易过拟合；C越小，容错性越高，间隔越大，泛化能力越强，但可能欠拟合。"
    },
    {
        "question": "决策树算法的优缺点分别是什么？如何缓解过拟合？",
        "ground_truth": "决策树优点包括：易于理解和解释、可视化能力强、不需要特征标准化、能处理数值型和类别型数据、可处理多输出问题。缺点包括：容易过拟合、对训练数据微小变化敏感（高方差）、可能产生有偏树（某些类别占主导时）、贪心算法不保证全局最优解。缓解过拟合的方法包括：预剪枝（提前停止分裂）、后剪枝（生成后剪掉冗余分支）、限制树的深度、设置节点最小样本数、设置分裂增益阈值。"
    },
    {
        "question": "集成学习中Bagging和Boosting的核心区别是什么？代表算法有哪些？",
        "ground_truth": "Bagging（Bootstrap Aggregating）通过自助采样并行独立训练多个基学习器，通过投票或平均得到最终结果，核心目标是降低方差、防止过拟合，代表算法为随机森林（Random Forest）。Boosting串行构建基学习器，每个新学习器重点修正前一个学习器的错误，通过加权组合所有基学习器，核心目标是降低偏差，代表算法包括AdaBoost、GBDT（梯度提升树）、XGBoost、LightGBM。两者的关键区别：Bagging并行训练、降低方差；Boosting串行训练、降低偏差。"
    },
    # ====== 深度学习部分（深度学习原理与应用.docx）====== #
    {
        "question": "深度学习相比传统机器学习有哪些核心优势？",
        "ground_truth": "深度学习有四大核心优势：第一，自动特征学习，无需领域专家人工设计特征（如SIFT、HOG特征），通过多层网络自动挖掘数据深层特征；第二，大数据适配性，模型参数规模庞大，数据量越大模型泛化能力越强，性能持续提升；第三，端到端建模，无需拆分特征提取、特征筛选、模型训练等步骤，直接从原始输入映射到任务输出，简化工程流程；第四，强非线性拟合能力，深层神经网络理论上可以拟合任意复杂的连续非线性函数，能够解决传统算法无法处理的复杂场景任务。"
    },
    {
        "question": "ReLU激活函数相比Sigmoid有什么优点和缺点？",
        "ground_truth": "ReLU公式为f(x)=max(0,x)，是当前深度学习最主流的隐藏层激活函数。相比Sigmoid的优点：输入正数时梯度恒为1，彻底缓解梯度消失问题；无指数运算，计算速度极快；稀疏激活特性，输入负数时输出为0，简化网络计算，提升特征稀疏性。缺点：存在神经元死亡问题，若参数更新不当，大量神经元输入恒为负数，梯度永久为0，参数不再更新；输出非零中心化。改进型包括Leaky ReLU（负数区域保留微小梯度）和GELU（广泛应用于Transformer大模型）。Sigmoid目前仅用于二分类输出层，极少用于隐藏层。"
    },
    {
        "question": "什么是反向传播算法（BP算法）？其核心原理是什么？",
        "ground_truth": "反向传播算法（BP算法）是训练深层神经网络的核心算法，1986年由Rumelhart和Hinton等人提出，解决了多层神经网络的参数更新问题。核心流程分为三步：第一步，前向传播计算所有层的输出值；第二步，计算输出层的误差梯度；第三步，通过链式法则反向逐层传递误差，计算每一层权重和偏置的梯度。参数更新规则为参数 = 参数 - 学习率 × 梯度。其数学基础是链式法则：对于多层复合函数，通过链式法则逐层求解损失函数对每一层参数的偏导数，实现误差从输出层向输入层的反向传递。"
    },
    {
        "question": "什么是Dropout正则化？为什么能防止过拟合？",
        "ground_truth": "Dropout是一种经典且高效的深度学习正则化技术。训练阶段，每次前向传播时随机丢弃（置零）一定比例的神经元（常见比例为0.5），不参与本次前向传播与反向传播，相当于每次训练都在随机子网络上进行。测试阶段，所有神经元参与计算但输出按丢弃比例缩小。Dropout能防止过拟合的原因有三：一是强迫网络学习冗余表示，每层神经元不依赖于特定神经元的存在；二是每次迭代等效于训练不同的子网络，最终输出是所有子网络的平均集成；三是打破了神经元之间的复杂共适应关系，显著提升模型的泛化能力。"
    },
    {
        "question": "什么是卷积神经网络（CNN）？其核心组成结构有哪些？",
        "ground_truth": "卷积神经网络（CNN）是专门为处理网格化结构数据（如图像）设计的深度学习模型，是计算机视觉领域最核心的模型架构。CNN的核心组成结构包括：卷积层（Convolutional Layer），通过卷积核在输入数据上滑动，提取局部特征，具有参数共享和局部连接特性，大幅减少参数量；池化层（Pooling Layer），通过下采样操作降低特征图的空间尺寸，减少计算量、防止过拟合，常见的有最大池化和平均池化；全连接层（Fully Connected Layer），位于网络末端，将提取的特征映射到任务输出（分类标签或回归值）。CNN的经典模型包括LeNet-5（手写识别）、AlexNet（2012年ImageNet冠军）、VGG、ResNet等。"
    },
]


def get_ollama_session():
    """创建 Ollama HTTP Session"""
    import requests
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=5,
        pool_maxsize=10,
        max_retries=1,
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def generate_answer(question, docs, prompt, config, session):
    """调用 DeepSeek 云模型生成回答（不带来源标注），返回纯文本答案"""
    try:
        response = session.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "stream": False,
            },
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            timeout=120,
        )
        if response.status_code != 200:
            print(f"    ⚠️ LLM 调用失败: {response.status_code} - {response.text[:200]}")
            return ""
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"    ⚠️ 生成回答异常: {e}")
        return ""


def run_evaluation(verbose=False):
    """运行 RAGAS 评估，返回结果字典"""
    import app as rag_app
    import requests

    print("=" * 60)
    print("🔬 RAGAS 评估 - app.py RAG 系统")
    print("=" * 60)

    # Step 1: 初始化 RAG 系统
    print("\n[1/4] 初始化检索器...")
    rag_app.qa_cache.clear()
    if rag_app.retriever is None:
        if not rag_app.init_retriever():
            print("❌ 检索器初始化失败，请检查 docs 文件夹和 Ollama 服务")
            return None
    print("✅ 检索器就绪")

    # Step 2: 配置 RAGAS 评估 LLM（DeepSeek 云模型）+ Embedding（本地 Ollama）
    print("\n[2/4] 配置 RAGAS 评估 LLM 和 Embedding...")
    if not DEEPSEEK_API_KEY:
        print("⚠️ 请先设置 DEEPSEEK_API_KEY（在 test_ragas.py 顶部）")
        return None

    evaluator_llm_lc = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=DEEPSEEK_API_KEY,
        temperature=0,
    )
    evaluator_llm = LangchainLLMWrapper(evaluator_llm_lc, bypass_n=True)

    ollama_embeddings = OllamaEmbeddings(
        model=rag_app.CONFIG["embedding_model"],
        base_url=rag_app.CONFIG["ollama_url"],
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]
    print("✅ 评估 LLM + Embedding 配置完成")

    # Step 3: 逐条运行 RAG pipeline
    print(f"\n[3/4] 运行 {len(TEST_DATA)} 条测试数据...")
    session = get_ollama_session()
    eval_samples = []
    per_question = []

    for i, item in enumerate(TEST_DATA):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n  [{i + 1}/{len(TEST_DATA)}] Q: {question}")

        t0 = time.time()
        docs, prompt, sources = rag_app.retrieve(question)
        retrieve_time = time.time() - t0

        if docs is None:
            print(f"    ⚠️ 未检索到相关内容，跳过")
            per_question.append({
                "question": question,
                "ground_truth": ground_truth,
                "error": "未检索到相关内容",
            })
            continue

        t0 = time.time()
        answer = generate_answer(
            question, docs, prompt, rag_app.CONFIG, session
        )
        gen_time = time.time() - t0

        contexts = [doc.page_content for doc in docs]

        print(f"    检索: {len(docs)}父块 / {len(sources)}来源 ({retrieve_time:.1f}s)")
        print(f"    生成: {len(answer)}字 ({gen_time:.1f}s)")

        if verbose:
            print(f"    答案预览: {answer[:200]}...")

        eval_samples.append({
            "user_input": question,
            "reference": ground_truth,
            "retrieved_contexts": contexts,
            "response": answer,
        })

        per_question.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer_preview": answer[:300] + ("..." if len(answer) > 300 else ""),
            "num_contexts": len(contexts),
            "sources": sources,
            "retrieve_time_s": round(retrieve_time, 2),
            "gen_time_s": round(gen_time, 2),
        })

    if not eval_samples:
        print("\n❌ 没有可评估的样本")
        return {"error": "没有可评估的样本", "per_question": per_question}

    # Step 4: 计算 RAGAS 指标
    print(f"\n[4/4] 计算 RAGAS 指标...")
    eval_dataset = Dataset.from_list(eval_samples)
    result = ragas_evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)

    result_df = result.to_pandas()
    metrics_summary = {}

    for col in result_df.columns:
        if col not in ("user_input", "reference", "retrieved_contexts", "response"):
            try:
                series = result_df[col].dropna()
                if len(series) > 0:
                    metrics_summary[col] = round(float(series.mean()), 4)
            except Exception:
                pass

    # 总体均分
    if metrics_summary:
        valid_scores = [v for v in metrics_summary.values() if v > 0]
        metrics_summary["overall_avg"] = (
            round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0
        )

    # 打印结果
    print("\n" + "=" * 60)
    print("📊 RAGAS 评估结果")
    print("=" * 60)
    metric_labels = {
        "faithfulness": "忠实度 (Faithfulness)",
        "answer_relevancy": "答案相关性 (AnswerRelevancy)",
        "context_precision": "上下文精度 (ContextPrecision)",
        "context_recall": "上下文召回 (ContextRecall)",
        "overall_avg": "综合均分",
    }
    for key, score in metrics_summary.items():
        label = metric_labels.get(key, key)
        bar = "█" * int(score * 40) + "░" * (40 - int(score * 40))
        print(f"  {label:30s}  {score:.4f}  {bar}")
    print("=" * 60)

    return {
        "metrics": metrics_summary,
        "num_samples": len(eval_samples),
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser(
        description="RAGAS 评估 - 测试 app.py RAG 系统质量"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="打印每个问题的详细评估结果"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="保存 JSON 报告到指定路径"
    )
    parser.add_argument(
        "--questions", "-q", type=int, default=None,
        help="只测试前 N 个问题（调试用）"
    )
    args = parser.parse_args()

    # 切换到项目根目录（app.py 使用相对路径 ./docs、./chroma_db）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    global TEST_DATA
    if args.questions:
        TEST_DATA = TEST_DATA[:args.questions]

    # 初始化 jieba（app.py 只在 __main__ 中调用）
    import jieba
    jieba.initialize()

    start_time = time.time()
    result = run_evaluation(verbose=args.verbose)
    elapsed = time.time() - start_time

    if result is None:
        sys.exit(1)

    if "error" in result:
        print(f"\n❌ 评估失败: {result['error']}")
        sys.exit(1)

    print(f"\n⏱ 总耗时: {elapsed:.1f}s")

    # 保存 JSON 报告
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"📄 报告已保存: {output_path.resolve()}")

    print("\n✅ 评估完成")


if __name__ == "__main__":
    main()
