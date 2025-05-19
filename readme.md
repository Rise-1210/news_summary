<h1 align="center">Data Science in Practice: Summary Part </h1>

## 🔧 Installation

```
cd summary

conda create -n summary python=3.10
conda activate summary

pip install -r requirements.txt
```

## 📋 Preparation
For Models:
Please place your model inside the `model` folder, for example: `model/Qwen2.5-7B-Instruct` (that is the model we used).

You can use the following command to download this model from Hugging Face.
```
huggingface-cli download --resume-download Qwen/Qwen2.5-7B-Instruct --local-dir model/Qwen2.5-7B-Instruct
```

## 🏃 Quick Start
You can use our bash script to generate news summaries. 

* For the LLM, use `LLM.sh`

## 📊 Output Statistics
In our final report, we present the **Precision**, **Recall**, and **F1 scores** for **ROUGE-1**, **ROUGE-2**, **ROUGE-L**, and **BERTScore**.





