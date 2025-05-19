import json
import argparse

PROMPT1 = """
请你阅读以下新闻内容，并撰写一句话的简洁、准确的中文新闻摘要。
{content}
"""

PROMPT2 = """
请你仔细阅读以下新闻内容，先分析新闻的核心信息和重要细节，然后用一句话简洁、准确地总结新闻的主要内容，避免冗长。

新闻内容：
{content}

请先简要说明新闻的关键点，再写出不超过20个字的一句话摘要。按照如下格式生成：
<think>
...(简要说明新闻的关键点)
</think>

<Answer>
(摘要内容)
</Answer>
"""

def preprocess_json(input_file, output_file, CoT=False):
    prompt = PROMPT2 if CoT else PROMPT1

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    for idx, item in enumerate(data):
        new_item = {
            "id": str(idx),
            "instruction": prompt.format(content=item["content"]),
            "answer": item.get("title", ""),
        }
        processed_data.append(new_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess JSON data with or without CoT prompt.")
    parser.add_argument("--input_file", type=str, help="Input JSON file path")
    parser.add_argument("--output_file", type=str, help="Output JSON file path")
    parser.add_argument("--cot", action="store_true", help="Use CoT prompt if specified")

    args = parser.parse_args()

    preprocess_json(args.input_file, args.output_file, CoT=args.cot)
