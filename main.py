import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


### Cosine similarity ###

encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def compute_cosine_similarity(sent1, sent2):
    emb1 = encoder.encode(sent1, convert_to_tensor=True)
    emb2 = encoder.encode(sent2, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()



### Initialize Model ###

class FlanT5Evaluator:
    def __init__(self, model_name="google/flan-t5-small"): # base, large
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded: {model_name} on {self.device}")

    def get_token_probability(self, prompt, target_token):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False
            )
            logits = outputs.scores[0][0]
            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs[target_id].item()

    def experiment_a_metalinguistic(self, anchor, cand1, cand2, prompt_type="simple"):
        prompts = {
                  "simple": "Can you conclude from '{}' that '{}'? Respond with either Yes or No as your answer.",
                  "instruct": "You are a helpful writing assistant. Tell me if you can conclude from '{}' that '{}'. Respond with either Yes or No as your answer.",
                  "complex": "Here is a sentence: '{}' Can you conclude this from '{}'? Respond with either Yes or No as your answer. Answer: ",
        }

        if prompt_type == "complex":
            prompt1 = prompts[prompt_type].format(cand1, anchor)
            prompt2 = prompts[prompt_type].format(cand2, anchor)
        else:
            prompt1 = prompts[prompt_type].format(anchor, cand1)
            prompt2 = prompts[prompt_type].format(anchor, cand2)
            prompt1 = prompts[prompt_type].format(anchor, cand1)
            prompt2 = prompts[prompt_type].format(anchor, cand2)

        prob_yes_1 = self.get_token_probability(prompt1, "Yes")
        prob_yes_2 = self.get_token_probability(prompt2, "Yes")

        return {
            "prob_yes_cand1": prob_yes_1,
            "prob_yes_cand2": prob_yes_2,
            "yes_differential": prob_yes_1 - prob_yes_2,
            "correct": prob_yes_1 > prob_yes_2
        }

    def experiment_b_metalinguistic(self, anchor, cand1, cand2, prompt_type="simple"):
        prompts = {
                  "simple": "Which sentence can you conclude from '{}'?: 1) {} 2) {} Respond with either 1 or 2 as your answer.",
                  "instruct": "You are a helpful writing assistant. Tell me which sentence you can conclude from '{}': 1) {} 2) {} Respond with either 1 or 2 as your answer.",
                  "complex": "Here are two sentences: 1) {} 2) {} Which sentence can you conclude from '{}'? Respond with 1 or 2. Answer:",
        }

        if prompt_type == "complex":
            prompt = prompts[prompt_type].format(cand1, cand2, anchor)
        else:
            prompt = prompts[prompt_type].format(anchor, cand1, cand2)

        prob_1 = self.get_token_probability(prompt, "1")
        prob_2 = self.get_token_probability(prompt, "2")

        return {
            "prob_1": prob_1,
            "prob_2": prob_2,
            "difference": prob_1 - prob_2,
            "correct": prob_1 > prob_2
        }



### Run Experiments ###

def run_experiments(data_path, model_name):
    evaluator = FlanT5Evaluator(model_name)
    df = pd.read_csv(data_path)

    results_a, results_b = [], []
    prompt_types = ["simple", "instruct", "complex"]

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Progress"):
        anchor, cand1, cand2, answer = row["sentence"], row["Yes"], row["No"], row["answer"]
        sim1 = compute_cosine_similarity(anchor, cand1)
        sim2 = compute_cosine_similarity(anchor, cand2)
        correct_direct = (sim1 > sim2 and answer == "Yes") or (sim2 > sim1 and answer == "No")

        for prompt_type in prompt_types:
            meta_a = evaluator.experiment_a_metalinguistic(anchor, cand1, cand2, prompt_type)
            results_a.append({
                "prompt_type": prompt_type,
                "direct_difference": sim1 - sim2,
                "direct_correct": correct_direct,
                "meta_difference": meta_a["yes_differential"],
                "meta_correct": meta_a["correct"]
            })

            meta_b = evaluator.experiment_b_metalinguistic(anchor, cand1, cand2, prompt_type)
            results_b.append({
                "prompt_type": prompt_type,
                "direct_difference": sim1 - sim2,
                "direct_correct": correct_direct,
                "meta_difference": meta_b["difference"],
                "meta_correct": meta_b["correct"]
            })

    return results_a, results_b



### Save Results ###

def analyze_results(results_a, results_b):
    df_a = pd.DataFrame(results_a)
    df_b = pd.DataFrame(results_b)

    print("=== EXPERIMENT A ===")
    for pt in df_a["prompt_type"].unique():
        d = df_a[df_a["prompt_type"] == pt]
        r, p = pearsonr(d["direct_difference"], d["meta_difference"])
        print(f"{pt} - Meta Accuracy: {d['meta_correct'].mean():.3f}, r = {r:.3f}, p = {p:.3f}")

    print("\n=== EXPERIMENT B ===")
    for pt in df_b["prompt_type"].unique():
        d = df_b[df_b["prompt_type"] == pt]
        r, p = pearsonr(d["direct_difference"], d["meta_difference"])
        print(f"{pt} - Meta Accuracy: {d['meta_correct'].mean():.3f}, r = {r:.3f}, p = {p:.3f}")

    df_a.to_csv("result_a.csv", index=False)
    df_b.to_csv("result_b.csv", index=False)
    print("✅ saved.")



### Run / Execute ###

results_a, results_b = run_experiments("data.csv", "google/flan-t5-small") #base, large 
analyze_results(results_a, results_b)
