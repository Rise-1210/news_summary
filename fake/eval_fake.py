import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm
import json
import os
import pickle

# 设置API配置
base_url = "https://api.siliconflow.cn/v1"
apikey = ""

# 初始化OpenAI客户端
client = OpenAI(
    base_url=base_url,
    api_key=apikey
)

def ensure_directory_exists(file_path):
    """确保目录存在，如果不存在则创建"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def load_data(file_path):
    """加载pickle数据文件"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)

        print(f"数据形状: {df.shape}")
        print("\n数据前5行:")
        print(df.head())
        return df
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        raise

def detect_fake_news(text):
    """使用Qwen模型检测假新闻，并返回标签和原始LLM输出"""
    prompt = f"""请判断以下新闻是否为假新闻。请只回答"是"或"否"。

新闻内容：
{text}

请判断："""

    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": "你是一个专业的假新闻检测助手。请仔细分析新闻内容，判断其真实性。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )

        result = response.choices[0].message.content.strip()
        print(f"预测结果: {result}")

        if result == "是":
            return 1, result
        elif result == "否":
            return 0, result
        else:
            return None, result
    except Exception as e:
        print(f"Error processing text: {e}")
        return None, f"ERROR: {e}"

def evaluate_predictions(true_labels, pred_labels):
    """评估预测结果"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print("\n评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

def main():
    # 设置文件路径
    data_file = r"C:\Users\zhr\Desktop\summary\fake\data\test.pkl"
    results_file = "fake/detection_results_test.csv"

    # 确保保存目录存在
    ensure_directory_exists(results_file)

    # 加载数据
    df = load_data(data_file)

    # 存储结果
    predictions = []
    raw_outputs = []

    print("\n开始预测...")
    for text in tqdm(df['content']):
        pred, raw = detect_fake_news(text)
        predictions.append(pred)
        raw_outputs.append(raw)
        time.sleep(1)  # 避免API限制

    # 保存所有原始与预测结果（不管有没有错误）
    results = pd.DataFrame({
        'content': df['content'],
        'true_label': df['label'],
        'predicted_label': predictions,
        'llm_output': raw_outputs
    })
    results.to_csv(results_file, index=False)
    print(f"\n结果已保存到 '{results_file}'")

    # 仅使用有效预测进行评估
    filtered = results[results['predicted_label'].isin([0, 1])]
    if not filtered.empty:
        evaluate_predictions(
            filtered['true_label'].astype(int).tolist(),
            filtered['predicted_label'].tolist()
        )
    else:
        print("\n没有可用于评估的有效预测结果。")

if __name__ == "__main__":
    main()
