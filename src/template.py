IGNORE_INDEX = -100

inst_s_list = ['Q:', 'Question:', '', 'Input:']

answer_s_list = ['A:', 'Answer:', 'Response:']

user_tokens=[1786, 4194, 95388]  # <用户>

assistant_tokens=[1786, 10850, 95388] # <AI>

pythia_user_tokens=[29, 12335, 46136, 31]  # <用户>

pythia_assistant_tokens=[29, 18128, 31] # <AI>

RESPONSE_START_TOKEN_IDS = [128006, 78191, 128007]  # <|start_header_id|>assistant<|end_header_id|>
RESPONSE_START_TOKEN_IDS_pre = [27, 91, 318, 5011, 91, 29, 78191] # <|im_start|>assistant

PROMPT_DICT = {
    "Open-domain-QA": (
        "Background:\n{context}\n\n<inst s>\n{question}\n\n<answer s>\n"
    ),
    "Reading-Comprehension": (
        "Background:\n{context}\n\n<inst s>\n{question}\n\n<answer s>\n"
    ),
    # 这个数据集很特殊，没有passage
    "quarel": (
        "<inst s>\n{question}\n\n<answer s>\n"
    ),
    "Summarization": (
        "Background:\n{context}\n\nSummarize this article:\n<answer s>\n"
    ),
    # 我对这个任务有点问题了，cot已经包含答案了
    "Chain-of-thought-Reasoning": (
        "Background:\n{context}\n\n<inst s>\n{question}\n\n<answer s>\n"
    ),
}

dataset_type_dict = {
    "aqua_rat" : "Chain-of-thought-Reasoning",
    "cnn_dailymail" : "Summarization",
    "commonsense_qa" : "Open-domain-QA",
    "coqa" : "Reading-Comprehension",
    "drop" : "Reading-Comprehension",
    "ecqa" : "Chain-of-thought-Reasoning",
    "gsm8k" : "Chain-of-thought-Reasoning",
    "math_qa" : "Open-domain-QA",
    "newsqa" : "Reading-Comprehension",
    "quail" : "Reading-Comprehension",
    "quarel" : "quarel",
    "squad_v2" : "Reading-Comprehension",
    "strategyqa" : "Chain-of-thought-Reasoning",
    "web_questions" : "Open-domain-QA",
    "wiki_qa" : "Open-domain-QA",
    "yahoo_answers_qa" : "Open-domain-QA",
    "marcoqa" : "Open-domain-QA",
}

llama_dataset_type_dict = {
    "aqua_rat" : "Open-domain-QA",
    "cnn_dailymail" : "Summarization",
    "commonsense_qa" : "Open-domain-QA",
    "coqa" : "Reading-Comprehension",
    "drop" : "Reading-Comprehension",
    "ecqa" : "Open-domain-QA",
    "gsm8k" : "Chain-of-thought-Reasoning",
    "math_qa" : "Open-domain-QA",
    "newsqa" : "Reading-Comprehension",
    "quail" : "Reading-Comprehension",
    "quarel" : "quarel",
    "squad_v2" : "Reading-Comprehension",
    "strategyqa" : "Chain-of-thought-Reasoning",
    "web_questions" : "Open-domain-QA",
    "wiki_qa" : "Open-domain-QA",
    "yahoo_answers_qa" : "Open-domain-QA",
    "marcoqa" : "Open-domain-QA",
}



prompt_template = "<用户>{}<AI>"
augment_template = 'Background:\n{}\n\n{}'


minicpm_multi_choice = 'The following is multiple choice question. Please choose the best answer choice which can answer the following question.\n{}\nAnswer:'
QA_template = 'Q: {}\nA:'

minicpm_multi_COT_template = "Please answer multiple choice question and choose the best answer choice first. Then give your explanation between [<COT] and [COT>]."
minicpm_QA_COT_template = "Please answer the question. Then give your explanation between [<COT] and [COT>]."

COT_question = 'question: {}\nAnswer:'

llama_multi_choice = 'Please answer the multiple choice questions below and output only the choice.\n{}\nAnswer:'
llama_multi_COT_template = "Please answer multiple choice question and choose the best answer choice first. Then give your explanation."
llama_QA_COT_templeta = "Please answer the question and only output the answer. Then give your explanation between [<COT] and [COT>]."

