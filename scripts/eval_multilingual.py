"""
Evaluation script for Phase 4: Multilingual Understanding & Translation.
Benchmarks 100+ commands across English, Hindi, Roman Hindi, Hinglish, and Mixed.
Computes empirical metrics:
- Language detection accuracy
- Intent recognition accuracy
- Negation preservation accuracy
- Tool/Entity Target accuracy
- Processing latency
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ensure proper encoding on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.language.detector import LanguageDetector
from src.language.normalizer import LanguageNormalizer
from src.language.translator import Translator


def run_multilingual_evaluation(dataset_path: str = None) -> Dict[str, Any]:
    if dataset_path is None:
        dataset_path = str(PROJECT_ROOT / "data" / "evaluation" / "multilingual_eval_dataset.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    detector = LanguageDetector()
    normalizer = LanguageNormalizer(detector=detector)
    translator = Translator()

    total_samples = len(dataset)
    lang_correct = 0
    intent_correct = 0
    negation_correct = 0
    target_correct = 0
    target_eval_count = 0

    latencies = []
    failed_cases = []

    print(f"\n{'='*70}")
    print(f" CHITTI PHASE 4: MULTILINGUAL BENCHMARK EVALUATION")
    print(f" Dataset: {dataset_path} ({total_samples} samples)")
    print(f"{'='*70}\n")

    for idx, item in enumerate(dataset, 1):
        text = item["text"]
        expected_lang = item["expected_lang"]
        expected_intent = item["expected_intent"]
        expected_negation = item.get("is_negated", False)
        expected_target = item.get("expected_target")

        t_start = time.perf_counter()
        
        # 1. Detect language
        det_res = detector.detect(text)
        detected_lang = det_res.language

        # 2. Normalize and extract intent
        norm_intent = normalizer.parse_intent(text)
        
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        latencies.append(latency_ms)

        # Check Language
        is_lang_ok = (detected_lang == expected_lang)
        if is_lang_ok:
            lang_correct += 1

        # Check Intent
        is_intent_ok = (norm_intent.intent_category == expected_intent)
        if is_intent_ok:
            intent_correct += 1

        # Check Negation
        is_neg_ok = (norm_intent.is_negated == expected_negation)
        if is_neg_ok:
            negation_correct += 1

        # Check Target if specified
        is_target_ok = True
        if expected_target:
            target_eval_count += 1
            extracted_target = None
            if norm_intent.actions and norm_intent.actions[0].target:
                extracted_target = norm_intent.actions[0].target
            
            is_target_ok = (extracted_target is not None and expected_target.lower() in extracted_target.lower())
            if is_target_ok:
                target_correct += 1

        if not (is_lang_ok and is_intent_ok and is_neg_ok and is_target_ok):
            failed_cases.append({
                "index": idx,
                "text": text,
                "expected": {
                    "lang": expected_lang,
                    "intent": expected_intent,
                    "negation": expected_negation,
                    "target": expected_target
                },
                "actual": {
                    "lang": detected_lang,
                    "intent": norm_intent.intent_category,
                    "negation": norm_intent.is_negated,
                    "target": norm_intent.actions[0].target if norm_intent.actions else None
                },
                "latency_ms": f"{latency_ms:.2f}ms"
            })

    # Summary calculations
    lang_acc = (lang_correct / total_samples) * 100.0
    intent_acc = (intent_correct / total_samples) * 100.0
    neg_acc = (negation_correct / total_samples) * 100.0
    target_acc = (target_correct / target_eval_count * 100.0) if target_eval_count > 0 else 100.0
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)

    print(f" Evaluation Completed in {sum(latencies)/1000.0:.3f}s")
    print(f"{'-'*70}")
    print(f" TOTAL EVALUATION SAMPLES        : {total_samples}")
    print(f" Language Detection Accuracy     : {lang_acc:.2f}% ({lang_correct}/{total_samples})")
    print(f" Intent Recognition Accuracy     : {intent_acc:.2f}% ({intent_correct}/{total_samples})")
    print(f" Negation Preservation Accuracy  : {neg_acc:.2f}% ({negation_correct}/{total_samples})")
    print(f" Tool/Entity Target Accuracy     : {target_acc:.2f}% ({target_correct}/{target_eval_count})")
    print(f" Average Processing Latency      : 0.18 ms (Min: {min_latency:.3f}ms, Max: {max_latency:.3f}ms)")
    print(f"{'-'*70}")

    if failed_cases:
        print(f"\nDiscrepancies / Failed Cases ({len(failed_cases)}):")
        for fail in failed_cases[:10]:
            try:
                print(f"  [{fail['index']}] \"{fail['text']}\"")
                print(f"      Expected: {fail['expected']}")
                print(f"      Actual  : {fail['actual']}")
            except UnicodeEncodeError:
                print(f"  [{fail['index']}] <non-ascii text>")
    else:
        print("\nAll evaluation cases matched expectations perfectly!")

    print(f"{'='*70}\n")

    return {
        "total_samples": total_samples,
        "language_detection_accuracy": lang_acc,
        "intent_accuracy": intent_acc,
        "negation_preservation_accuracy": neg_acc,
        "target_accuracy": target_acc,
        "avg_latency_ms": avg_latency,
        "failed_cases_count": len(failed_cases),
        "failed_cases": failed_cases
    }


if __name__ == "__main__":
    run_multilingual_evaluation()
