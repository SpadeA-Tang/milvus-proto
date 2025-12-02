# compute_phrase_match_slop.py demonstrates how to use compute_phrase_match_slop API
# to calculate the slop (edit distance) between a query phrase and data texts.
#
# The slop value represents the minimum number of position moves needed to make
# the query phrase match the data text.

from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "phrase_match_slop_demo"


def print_results(data_texts, is_match_list, slop_list):
    for text, is_match, slop in zip(data_texts, is_match_list, slop_list):
        match_str = "Matched,  " if is_match else "Unmatched,"
        slop_str = str(slop) if is_match else "N/A"
        print(f"  [{match_str} slop={slop_str:>3}] {text}")


def main():
    client = MilvusClient("http://localhost:19530")

    # ==================== English ====================
    print("=" * 80)

    query_text = "data privacy"
    data_texts = [
        "Ensure data privacy now.",           # Slop 0: Adjacent match
        "Ensure data user privacy now.",      # Slop 1: One word inserted
        "Ensure privacy data now.",           # Slop 2: Reversed order
        "Ensure privacy user data now.",      # Slop 3: Reversed + inserted
        "Data is important.",                 # No match: only one term
    ]

    print(f"Query: '{query_text}'\n")

    is_match_list, slop_list = client.compute_phrase_match_slop(
        query_text=query_text,
        data_text=data_texts,
        analyzer_params={"tokenizer": "standard"},
    )
    print_results(data_texts, is_match_list, slop_list)

    # ==================== Chinese ====================
    print()
    print("=" * 80)

    query_text_zh = "数据隐私"
    data_texts_zh = [
        "确保数据隐私安全",
        "确保数据用户隐私安全",
        "确保隐私数据安全",
        "确保隐私用户数据安全",
        "数据很重要",
    ]

    print(f"Query: '{query_text_zh}'\n")

    is_match_list_zh, slop_list_zh = client.compute_phrase_match_slop(
        query_text=query_text_zh,
        data_text=data_texts_zh,
        analyzer_params={"tokenizer": "jieba"},
    )
    print_results(data_texts_zh, is_match_list_zh, slop_list_zh)

if __name__ == "__main__":
    main()
