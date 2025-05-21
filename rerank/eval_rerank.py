import pandas as pd
import requests
import json
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import re
import time

class NewsReranker:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def load_data(self, behaviors_path: str, news_path: str):
        self.behaviors = pd.read_csv(behaviors_path, sep='\t', header=None, 
                                     names=['impression_id', 'user_id', 'timestamp', 'history', 'candidates'])
        self.news = pd.read_csv(news_path, sep='\t', header=None,
                                names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'entities', 'keywords'])
        self.news.set_index('news_id', inplace=True)

    def get_news_content(self, news_id: str) -> str:
        try:
            news_item = self.news.loc[news_id]
            return f"ID: {news_id}\nTitle: {news_item['title']}\nAbstract: {news_item['abstract']}"
        except KeyError:
            return f"ID: {news_id}\nTitle: [Missing]\nAbstract: [Missing]"

    def get_history_content(self, history_ids: List[str]) -> str:
        return "\n\n".join(self.get_news_content(nid) for nid in history_ids)

    def parse_ranking_from_response(self, response_text: str, candidate_ids: List[str]) -> List[str]:
        found_ids = re.findall(r'N\d+', response_text)
        seen = set()
        ranked_ids = []
        for nid in found_ids:
            if nid in candidate_ids and nid not in seen:
                ranked_ids.append(nid)
                seen.add(nid)
        return ranked_ids if ranked_ids else candidate_ids

    def rerank(self, history_ids: List[str], candidate_ids: List[str]) -> tuple:
        history_content = self.get_history_content(history_ids)
        prompt = f"""Given the user's reading history:\n{history_content}\n\nPlease rank the following news articles from most relevant to least relevant:\n"""
        for news_id in candidate_ids:
            prompt += f"\n{self.get_news_content(news_id)}\n"

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=30  # 避免无限等待
        )

        if response.status_code != 200:
            raise Exception(f"API call failed: {response.text}")

        response_text = response.json()['choices'][0]['message']['content']
        ranked_ids = self.parse_ranking_from_response(response_text, candidate_ids)
        return ranked_ids, response_text

    def calculate_ngcd(self, predicted_ranking: List[str], ground_truth: List[str], k: int = 10) -> float:
        if not predicted_ranking or not ground_truth:
            return 0.0
        pred_top_k = predicted_ranking[:k]
        true_top_k = ground_truth[:k]
        return len(set(pred_top_k) & set(true_top_k)) / k


# 重试机制函数
def safe_rerank(reranker, history_ids, candidate_ids, retries=3):
    for attempt in range(retries):
        try:
            return reranker.rerank(history_ids, candidate_ids)
        except Exception as e:
            print(f"[Retry {attempt + 1}/{retries}] rerank failed: {e}")
            time.sleep(5)
    raise Exception("All retries failed.")


def main():
    base_url = "https://api.siliconflow.cn/v1"
    api_key = ""
    reranker = NewsReranker(base_url, api_key)

    reranker.load_data(
        r"C:\Users\zhr\Desktop\summary\rerank\MINDlarge_test\behaviors.tsv",
        r"C:\Users\zhr\Desktop\summary\rerank\MINDlarge_test\news.tsv"
    )

    results = []

    for idx, row in tqdm(reranker.behaviors.iterrows(), total=len(reranker.behaviors)):
        try:
            if pd.isna(row['history']) or pd.isna(row['candidates']):
                print(f"[Skip] impression_id {row['impression_id']} has missing history or candidates.")
                continue

            history_ids = row['history'].split()
            candidate_ids = row['candidates'].split()

            reranked_ids, llm_response = safe_rerank(reranker, history_ids, candidate_ids)
            ngcd = reranker.calculate_ngcd(reranked_ids, candidate_ids)

            results.append({
                'impression_id': row['impression_id'],
                'user_id': row['user_id'],
                'timestamp': row['timestamp'],
                'history': row['history'],
                'candidates': row['candidates'],
                'predicted_ranking': ' '.join(reranked_ids),
                'llm_response': llm_response,
                'ngcd_score': ngcd
            })
        except Exception as e:
            print(f"[Error] impression_id {row.get('impression_id', 'unknown')} failed: {e}")
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv("rerank_results_all.csv", index=False)
    print("\nResults saved to rerank_results_all.csv")
    print(f"Average NGCD@10: {results_df['ngcd_score'].mean():.4f}")


if __name__ == "__main__":
    main()
