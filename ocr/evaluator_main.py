
import argparse
import os
import json
from datetime import datetime
from typing import List, Optional

import Credentials
from OCRModels import (
    OCRModel,
    GoogleVisionOCR,
    TesseractOCR,
)
from Metrics import (
    SimilarityMetric,
    SequenceMatcherMetric,
    Jaccard,
    Levenshtein
)
from Evaluator import OCREvaluator


def setup_ocr_models(
        use_google_vision: bool = True,
        use_tesseract: bool = True,
        use_trocr: bool = False,
        google_credentials: Optional[str] = Credentials.GOOGLE_APPLICATION_CREDENTIALS
) -> List[OCRModel]:
    models = []

    if use_google_vision:
        try:
            models.append(GoogleVisionOCR(credentials_path=google_credentials))
            print("Added Google Vision OCR model")
        except Exception as e:
            print(f"Failed to initialize Google Vision OCR: {e}")

    if use_tesseract:
        try:
            models.append(TesseractOCR())
            print("Added Tesseract OCR model")
        except Exception as e:
            print(f"Failed to initialize Tesseract OCR: {e}")
    #commented out due to conda pytorch issues
    #if use_trocr:
    #    try:
    #        models.append(TROCR())
    #        print("Added TrOCR model")
    #    except Exception as e:
    #        print(f"Failed to initialize TrOCR: {e}")

    return models


def setup_similarity_metrics() -> List[SimilarityMetric]:
    metrics = [
        SequenceMatcherMetric(),
        Jaccard(),
        Levenshtein()
    ]

    print(f"Set up {len(metrics)} similarity metrics: {', '.join(m.name() for m in metrics)}")
    return metrics


def create_results_directory():
    base_dir = "evaluation_results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f"evaluation_results_{timestamp}")
    os.makedirs(results_dir)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir)
    return results_dir


def main():
    parser = argparse.ArgumentParser(description="Book Spine OCR Evaluation")
    parser.add_argument("--ground_truth", type=str, required=True,
                        help="Path to ground truth JSON file")
    parser.add_argument("--google_credentials", type=str, default=None,
                        help="Path to Google Cloud credentials JSON file")
    parser.add_argument("--skip_google", action="store_true",
                        help="Skip Google Vision OCR")
    parser.add_argument("--skip_tesseract", action="store_true",
                        help="Skip Tesseract OCR")
    parser.add_argument("--skip_trocr", action="store_true",
                        help="Skip TrOCR")

    args = parser.parse_args()
    results_dir = create_results_directory()
    
    ocr_models = setup_ocr_models(
        use_google_vision=not args.skip_google,
        use_tesseract=not args.skip_tesseract,
        use_trocr=not args.skip_trocr,
        google_credentials=args.google_credentials
    )

    if not ocr_models:
        print("Error: No OCR models could be initialized. Exiting.")
        return 1
        
    similarity_metrics = setup_similarity_metrics()
    evaluator = OCREvaluator(ocr_models, similarity_metrics)
    print(f"Evaluating OCR models using ground truth from {args.ground_truth}")
    results_file = os.path.join(results_dir, "evaluation_results.json")
    plots_dir = os.path.join(results_dir, "plots")
    evaluation_results = evaluator.evaluate_all(args.ground_truth)
    evaluator.save_results(results_file)
    dataframes, summary = evaluator.analyze_results()
    original_plot_results = evaluator.plot_results
    def plot_results_wrapper(dataframes):
        current_dir = os.getcwd()
        os.chdir(plots_dir)
        original_plot_results(dataframes)
        os.chdir(current_dir)
    
    plot_results_wrapper(dataframes)
    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Evaluation completed successfully!")
    print(f"Results saved to: {results_dir}")
    return 0


if __name__ == "__main__":
    exit(main())