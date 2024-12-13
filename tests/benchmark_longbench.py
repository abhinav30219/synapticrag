"""Benchmark script for evaluating SynapticRAG on LongBench"""
import sys
import os
import json
import torch
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import datasets
from rouge import Rouge

from synaptic.pipeline import Pipeline
from synaptic.config import Config

# Import LongBench constants
DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
}

DATASET2MAXNEWTOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512
}

DATASET2CATEGORY = {
    "narrativeqa": "EN Single-Doc QA",
    "qasper": "EN Single-Doc QA", 
    "multifieldqa_en": "EN Single-Doc QA",
    "hotpotqa": "EN Multi-Doc QA",
    "2wikimqa": "EN Multi-Doc QA",
    "musique": "EN Multi-Doc QA",
    "gov_report": "EN Summarization",
    "qmsum": "EN Summarization",
    "multi_news": "EN Summarization"
}

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Calculate word-level F1 score"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def rouge_score(prediction, ground_truth):
    """Calculate ROUGE-L score"""
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
        return scores["rouge-l"]["f"]
    except:
        return 0.0

class LongBenchEvaluator:
    """Evaluator for LongBench datasets"""
    
    def __init__(
        self,
        config: Config,
        data_dir: str = "data/longbench",
        output_dir: str = "results/longbench",
        dataset_names: Optional[List[str]] = None
    ):
        self.pipeline = Pipeline(config=config)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if dataset_names is None:
            self.dataset_names = [
                "narrativeqa", "qasper", "multifieldqa_en",
                "hotpotqa", "2wikimqa", "musique",
                "gov_report", "qmsum", "multi_news"
            ]
        else:
            self.dataset_names = dataset_names
    
    def evaluate_dataset(self, dataset_name: str) -> Dict:
        """Evaluate model on a specific dataset"""
        # Load dataset
        dataset_path = self.data_dir / f"{dataset_name}.json"
        if not dataset_path.exists():
            print(f"Dataset {dataset_name} not found at {dataset_path}")
            return {}
            
        with open(dataset_path) as f:
            dataset = json.load(f)
            
        results = []
        scores = []
        
        for item in tqdm(dataset, desc=f"Evaluating {dataset_name}"):
            # Get prediction
            prompt = DATASET2PROMPT[dataset_name].format(
                context=item["context"],
                input=item.get("question", "")
            )
            
            prediction = self.pipeline(
                query=prompt,
                context=item["context"],
                max_new_tokens=DATASET2MAXNEWTOKENS[dataset_name]
            )
            
            # Calculate score
            if dataset_name in ["gov_report", "qmsum", "multi_news"]:
                score = rouge_score(prediction, item["answer"])
            else:
                score = max(f1_score(prediction, ans) for ans in item["answers"])
                
            scores.append(score)
            results.append({
                "question": item.get("question", ""),
                "context": item["context"],
                "prediction": prediction,
                "answer": item["answer"] if "answer" in item else item["answers"],
                "score": score
            })
            
        # Save results
        output_path = self.output_dir / f"{dataset_name}_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        return {
            "dataset": dataset_name,
            "category": DATASET2CATEGORY[dataset_name],
            "average_score": sum(scores) / len(scores) if scores else 0,
            "num_examples": len(dataset)
        }
    
    def evaluate_all(self) -> Dict:
        """Evaluate on all datasets"""
        all_results = {}
        category_scores = defaultdict(list)
        
        for dataset_name in self.dataset_names:
            result = self.evaluate_dataset(dataset_name)
            if result:
                all_results[dataset_name] = result
                category_scores[result["category"]].append(result["average_score"])
                
        # Calculate category averages
        category_averages = {
            category: sum(scores) / len(scores)
            for category, scores in category_scores.items()
        }
        
        # Calculate overall average
        all_scores = [r["average_score"] for r in all_results.values()]
        overall_average = sum(all_scores) / len(all_scores) if all_scores else 0
        
        summary = {
            "results": all_results,
            "category_averages": category_averages,
            "overall_average": overall_average
        }
        
        # Save summary
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        return summary

if __name__ == "__main__":
    # Initialize evaluator
    config = Config()
    evaluator = LongBenchEvaluator(config=config)
    
    # Run evaluation
    results = evaluator.evaluate_all()
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"\nOverall Average: {results['overall_average']:.2f}")
    print("\nCategory Averages:")
    for category, score in results["category_averages"].items():
        print(f"{category}: {score:.2f}")
    print("\nDataset Results:")
    for dataset, result in results["results"].items():
        print(f"{dataset}: {result['average_score']:.2f} ({result['num_examples']} examples)")
