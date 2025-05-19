#For LLM
python LLM/preprocess.py \
    --input_file data/summary_test_data.json \
    --output_file data/summary_test_LLM.json\


python LLM/vllm_infer_text.py  \
    --model_name_or_path LLM/model/Qwen2.5-7B-Instruct \
    --dataset_path data/summary_test_LLM.json \
    --template qwen\
    --prediction_key llm_predict_qwen_raw\
    --tensor_parallel_size 4

python evaluate.py \
    --file data/summary_test_LLM.json \
    --key llm_predict_qwen_raw\
    --ref answer
