from pymilvus import MilvusClient, DataType
import numpy as np

# ========================================
#   颜色辅助函数
# ========================================
def blue(t): return f"\033[94m{t}\033[0m"
def green(t): return f"\033[92m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def red(t): return f"\033[91m{t}\033[0m"
def bold(t): return f"\033[1m{t}\033[0m"
def cyan(t): return f"\033[96m{t}\033[0m"

URI = "http://localhost:19530"
TOKEN = "root:Milvus"
COLLECTION_NAME = "phrase_match_demo_ultimate"

# ========================================
#   1. 建表
# ========================================
def setup_collection(client: MilvusClient):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # English
    schema.add_field(
        field_name="text_en",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "english"},
    )

    # Chinese (使用默认 chinese 分词)
    schema.add_field(
        field_name="text_zh",
        datatype=DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "chinese"},
    )

    EMB_DIM = 8
    schema.add_field(field_name="embeddings", datatype=DataType.FLOAT_VECTOR, dim=EMB_DIM)

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embeddings", index_type="HNSW", metric_type="IP", params={"M": 8, "efConstruction": 64})
    client.create_index(COLLECTION_NAME, index_params)
    return EMB_DIM

# ========================================
#   2. 插入数据 (实验室级纯净数据)
# ========================================
def insert_data(client: MilvusClient, emb_dim: int):
    # ---------------------------------------------------------
    # 英文策略：Target = "data privacy"
    # 去除 stop words (of, the, is) 避免位置计算干扰
    # ---------------------------------------------------------
    english_data = [
        # Slop 0: Adjacent
        "Ensure data privacy now.",
        
        # Slop 1: Insert 1 word 'user' -> "data user privacy"
        "Ensure data user privacy now.",
        
        # Slop 2: Reverse -> "privacy data"
        "Ensure privacy data now.",
        
        # Slop 3: Reverse + Insert 'user' -> "privacy user data"
        # 逻辑：privacy(0), user(1), data(2) -> Target: data, privacy
        # Move data to 0 (2 moves), privacy is at 0 (needs 1).
        # Total edits approx 3.
        "Ensure privacy user data now."
    ]

    # ---------------------------------------------------------
    # 中文策略：Target = "逻辑 芯片"
    # 核心修正：
    # 1. 移除所有停用词（的、了、该）。
    # 2. 只有实义词。
    # 3. 强制空格辅助分词，确保 Token 独立。
    # ---------------------------------------------------------
    chinese_data = [
        # # [Slop 0] 连续
        # # 结构：前缀 + 逻辑 + 芯片 + 后缀
        # "测试 逻辑 芯片 性能",
        #
        # # [Slop 1] 插入 1 个实义词 '核心'
        # # 结构：逻辑 + 核心 + 芯片
        # # 距离：1
        # "测试 逻辑 核心 芯片 性能",
        #
        # # [Slop 2] 反转
        # # 结构：芯片 + 逻辑
        # # 距离：通常为 2
        # "测试 芯片 逻辑 性能",
        #
        # # [Slop 3] 反转 + 插入 '核心'
        # # 结构：芯片 + 核心 + 逻辑
        # # 距离：通常为 3 (或4，取决于具体实现，Slop 3 通常能覆盖)
        # "测试 芯片 核心 逻辑 性能",

        "测试 的 中文",
    ]

    total = len(english_data)
    rng = np.random.default_rng(42)
    vectors = rng.random((total, emb_dim)).astype("float32")

    rows = []
    for en, zh, v in zip(english_data, chinese_data, vectors):
        rows.append({"text_en": en, "text_zh": zh, "embeddings": v.tolist()})

    client.insert(collection_name=COLLECTION_NAME, data=rows)
    client.load_collection(COLLECTION_NAME)
    print(f"已插入 {total} 条无干扰的纯净测试数据。\n")

# ========================================
#   3. 查询逻辑 (显示优化)
# ========================================
def phrase_query(client, field, target_phrase, slop_val):
    if slop_val is None:
        expr = f"PHRASE_MATCH({field}, '{target_phrase}')"
        label = "slop=0 (精确)"
    else:
        expr = f"PHRASE_MATCH({field}, '{target_phrase}', {slop_val})"
        label = f"slop={slop_val}"

    print(bold(cyan(f"▶ {expr}")))
    print(blue(f"  测试目标: {label}"))

    results = client.query(
        collection_name=COLLECTION_NAME,
        filter=expr,
        output_fields=["id", field]
    )

    count = len(results)
    
    # 构建高亮显示
    hl_results = []
    if count > 0:
        results.sort(key=lambda x: x['id'])
        for r in results:
            text = r[field]
            # 简单粗暴的高亮，仅供演示
            for term in target_phrase.split():
                text = text.replace(term, bold(red(term)))
            hl_results.append(f"  ✓ [ID:{r['id']}] {text}")

    print(f"  共命中: {count} 条")
    if not hl_results:
        print(red("  (无匹配)"))
    else:
        for line in hl_results:
            print(green(line))

    print("-" * 50 + "\n")

# ========================================
#   Main Demo
# ========================================
def main():
    client = MilvusClient(uri=URI, token=TOKEN)
    emb_dim = setup_collection(client)
    insert_data(client, emb_dim)

    # ---------- 英文演示 ----------
    print(bold("========================================"))
    print(bold("🌍 英文 Phrase Match (No Stopwords)"))
    print(bold("Target: 'data privacy'"))
    print(bold("========================================"))

    # 预期：1条
    phrase_query(client, "text_en", "data privacy", None)
    # 预期：2条 (新增 data user privacy)
    phrase_query(client, "text_en", "data privacy", 1)
    # 预期：3条 (新增 privacy data)
    phrase_query(client, "text_en", "data privacy", 2)
    # 预期：4条 (新增 privacy user data)
    phrase_query(client, "text_en", "data privacy", 3)

    # ---------- 中文演示 ----------
    print(bold("========================================"))
    print(bold("🈺 中文 Phrase Match (无停用词干扰)"))
    print(bold("Target: '逻辑 芯片'"))
    print(bold("========================================"))

    # 预期：1条 (测试 逻辑 芯片 性能)
    phrase_query(client, "text_zh", "中文 测试", 8)
    
    # # 预期：2条 (新增 测试 逻辑 核心 芯片 性能)
    # # 解释：逻辑(0) 核心(1) 芯片(2) -> 距离1
    # phrase_query(client, "text_zh", "逻辑 芯片", 1)
    #
    # # 预期：3条 (新增 测试 芯片 逻辑 性能)
    # # 解释：反转 -> 距离2
    # phrase_query(client, "text_zh", "逻辑 芯片", 2)
    #
    # # 预期：4条 (新增 测试 芯片 核心 逻辑 性能)
    # # 解释：反转+插入 -> 距离3
    # phrase_query(client, "text_zh", "逻辑 芯片", 6)

if __name__ == "__main__":
    main()