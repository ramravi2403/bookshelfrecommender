
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from OCRModels import OCRModel
from Metrics import SimilarityMetric


class OCREvaluator:

    def __init__(
            self,
            ocr_models: List[OCRModel],
            similarity_metrics: List[SimilarityMetric]
    ):
        self.ocr_models = ocr_models
        print(f"Evaluating {len(ocr_models)} models: {', '.join([model.name() for model in ocr_models])}")
        self.similarity_metrics = similarity_metrics
        self.results = {}

    def load_ground_truth(self, json_path: str) -> Dict[str, Dict[str, str]]:
        with open(json_path, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        return ground_truth_data

    def evaluate_directory(
            self,
            directory_path: str,
            ground_truth_dict: Dict[str, str]
    ) -> Dict[str, Any]:
        directory_results = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        image_files = [f for f in os.listdir(directory_path)
                       if os.path.splitext(f)[1].lower() in image_extensions]

        print(f"Processing {len(image_files)} images in {directory_path}")

        for filename in tqdm(image_files, desc="Processing book spines"):
            file_path = os.path.join(directory_path, filename)
            ground_truth = ground_truth_dict.get(filename, "")

            file_results = {
                "ground_truth": ground_truth,
                "models": {}
            }
            for model in self.ocr_models:
                model_name = model.name()
                print(f"Processing {filename} with {model_name}")  # Debug print

                try:
                    ocr_text = model.detect_text(file_path)
                    print(f"{model_name} result: {ocr_text[:30]}...")  # Truncated preview

                    model_results = {
                        "ocr_text": ocr_text,
                        "metrics": {}
                    }

                    for metric in self.similarity_metrics:
                        metric_name = metric.name()
                        similarity = metric.similarity(ocr_text, ground_truth) if ground_truth else 0
                        model_results["metrics"][metric_name] = similarity
                    file_results["models"][model_name] = model_results
                except Exception as e:
                    print(f"Error processing {filename} with {model_name}: {e}")
                    file_results["models"][model_name] = {
                        "ocr_text": f"ERROR: {str(e)}",
                        "metrics": {metric.name(): 0 for metric in self.similarity_metrics}
                    }

            directory_results[filename] = file_results

        return directory_results

    def evaluate_all(self, ground_truth_json: str) -> Dict[str, Any]:
        ground_truth_data = self.load_ground_truth(ground_truth_json)
        all_results = {}

        # Process each directory in the ground truth data
        for directory_path, gt_dict in ground_truth_data.items():
            print(f"\nProcessing directory: {directory_path}")

            # Check if directory exists
            if not os.path.exists(directory_path):
                print(f"Directory {directory_path} does not exist. Skipping...")
                continue

            # Process all images in the directory with ground truth comparison
            directory_results = self.evaluate_directory(directory_path, gt_dict)

            # Store results for this directory
            all_results[directory_path] = directory_results

        self.results = all_results

        # Debug: Print summary of collected results to verify all models are included
        self._print_result_summary()

        return all_results

    def _print_result_summary(self):
        """Debug method to print a summary of collected results"""
        print("\n=== Results Summary ===")

        # Check if results are empty
        if not self.results:
            print("No results collected!")
            return

        # Count directories, files, and models in results
        directories = len(self.results)
        files = sum(len(dir_results) for dir_results in self.results.values())

        # Get list of models found in results
        models_in_results = set()
        for dir_results in self.results.values():
            for file_results in dir_results.values():
                models_in_results.update(file_results.get("models", {}).keys())

        print(f"Collected results for {directories} directories and {files} files")
        print(f"Models found in results: {', '.join(sorted(models_in_results))}")
        print(f"Expected models: {', '.join([model.name() for model in self.ocr_models])}")

        # Check if any models are missing
        missing_models = set([model.name() for model in self.ocr_models]) - models_in_results
        if missing_models:
            print(f"WARNING: The following models are missing from results: {', '.join(missing_models)}")

        print("=====================")

    def save_results(self, output_path: str = 'book_spines_evaluation.json') -> str:
        # Debug: Print model names right before saving
        models_in_results = set()
        for dir_results in self.results.values():
            for file_results in dir_results.values():
                models_in_results.update(file_results.get("models", {}).keys())
        print(f"Saving results with models: {', '.join(sorted(models_in_results))}")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")
        return output_path

    def _create_dataframe(self) -> pd.DataFrame:
        rows = []

        for directory, files in self.results.items():
            for filename, file_data in files.items():
                ground_truth = file_data["ground_truth"]

                # Make sure we process all models in the results
                for model_name, model_data in file_data.get("models", {}).items():
                    # Skip if model data is malformed
                    if not isinstance(model_data, dict):
                        print(f"Warning: Invalid model data for {model_name} in {filename}")
                        continue

                    ocr_text = model_data.get("ocr_text", "")

                    # Process all metrics for this model
                    for metric_name, similarity in model_data.get("metrics", {}).items():
                        rows.append({
                            "directory": directory,
                            "filename": filename,
                            "model": model_name,
                            "metric": metric_name,
                            "ocr_text": ocr_text,
                            "ground_truth": ground_truth,
                            "similarity": similarity,
                            "ocr_word_count": len(ocr_text.split()) if ocr_text else 0,
                            "gt_word_count": len(ground_truth.split()) if ground_truth else 0,
                            "word_count_diff": len(ocr_text.split()) - len(
                                ground_truth.split()) if ocr_text and ground_truth else 0
                        })

        # If no rows were created, something is wrong
        if not rows:
            print("WARNING: No data could be extracted from results. Check the structure of self.results.")
            # Print a sample of self.results for debugging
            print("Sample of results data:")
            for directory, files in list(self.results.items())[:1]:
                for filename, file_data in list(files.items())[:1]:
                    print(f"Directory: {directory}, File: {filename}")
                    print(json.dumps(file_data, indent=2)[:500] + "...")  # Print truncated sample

        return pd.DataFrame(rows)

    def generate_dataframes(self) -> Dict[str, pd.DataFrame]:
        all_data_df = self._create_dataframe()

        # Verify we have data for all models
        models_in_df = all_data_df["model"].unique() if not all_data_df.empty else []
        print(f"Models in dataframe: {', '.join(models_in_df)}")

        dataframes = {
            "all_data": all_data_df
        }

        if not all_data_df.empty:
            model_comparison = all_data_df.groupby(['model', 'metric'])['similarity'].agg(
                ['mean', 'std', 'min', 'max']).reset_index()
            dataframes["model_comparison"] = model_comparison

            metric_comparison = all_data_df.groupby(['metric', 'model'])['similarity'].agg(
                ['mean', 'std', 'min', 'max']).reset_index()
            dataframes["metric_comparison"] = metric_comparison

            directory_comparison = all_data_df.groupby(['directory', 'model', 'metric'])['similarity'].agg(
                ['mean', 'std', 'min', 'max']).reset_index()
            dataframes["directory_comparison"] = directory_comparison
        else:
            print("WARNING: Empty dataframe, cannot create comparison dataframes")
            # Add empty dataframes to maintain structure
            dataframes["model_comparison"] = pd.DataFrame()
            dataframes["metric_comparison"] = pd.DataFrame()
            dataframes["directory_comparison"] = pd.DataFrame()

        return dataframes

    def analyze_results(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        dataframes = self.generate_dataframes()

        all_data_df = dataframes["all_data"]
        model_comparison = dataframes["model_comparison"]
        summary = {}

        print("\nEvaluation Summary:")

        if all_data_df.empty:
            print("No data available for analysis")
            return dataframes, summary

        for model in all_data_df["model"].unique():
            model_data = all_data_df[all_data_df["model"] == model]

            print(f"\nModel: {model}")
            mean_similarity = model_data["similarity"].mean()
            print(f"Average Similarity: {mean_similarity:.4f}")

            model_summary = {
                "mean_similarity": mean_similarity,
                "min_similarity": model_data["similarity"].min(),
                "max_similarity": model_data["similarity"].max(),
                "std_similarity": model_data["similarity"].std(),
                "total_images": model_data["filename"].nunique(),
                "similarity_ranges": {}
            }

            # Group by similarity ranges
            ranges = [(0, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)]
            for low, high in ranges:
                count = ((model_data["similarity"] >= low) & (model_data["similarity"] < high)).sum()
                percentage = 100 * count / len(model_data) if len(model_data) > 0 else 0
                print(f"Similarity {low:.1f}-{high:.1f}: {count} books ({percentage:.1f}%)")

                model_summary["similarity_ranges"][f"{low:.1f}-{high:.1f}"] = {
                    "count": int(count),
                    "percentage": float(percentage)
                }

            summary[model] = model_summary

        return dataframes, summary

    def plot_results(self, dataframes: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        if dataframes is None:
            dataframes = self.generate_dataframes()

        all_data_df = dataframes["all_data"]
        model_comparison = dataframes["model_comparison"]

        if all_data_df.empty:
            print("No data available for plotting")
            return

        # Set style
        sns.set(style="whitegrid")

        # 1. Model performance comparison across all metrics
        plt.figure(figsize=(12, 8))
        sns.barplot(x="model", y="mean", hue="metric", data=model_comparison, palette="viridis")
        plt.title("OCR Model Performance Comparison", fontsize=16)
        plt.xlabel("OCR Model", fontsize=14)
        plt.ylabel("Average Similarity Score", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Similarity Metric")
        plt.tight_layout()
        plt.savefig("model_comparison.png")
        plt.close()

        plt.figure(figsize=(15, 10))
        for i, model in enumerate(all_data_df["model"].unique()):
            plt.subplot(len(all_data_df["model"].unique()), 1, i + 1)
            model_data = all_data_df[all_data_df["model"] == model]

            for metric in all_data_df["metric"].unique():
                metric_data = model_data[model_data["metric"] == metric]
                sns.kdeplot(metric_data["similarity"], label=f"{metric}", fill=True, alpha=0.3)

            plt.title(f"Distribution of Similarity Scores - {model}", fontsize=14)
            plt.xlabel("Similarity Score", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend()

        plt.tight_layout()
        plt.savefig("similarity_distributions.png")
        plt.close()

        # 3. Word count difference analysis
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="model", y="word_count_diff", data=all_data_df)
        plt.title("Word Count Difference (OCR - Ground Truth)", fontsize=16)
        plt.xlabel("OCR Model", fontsize=14)
        plt.ylabel("Word Count Difference", fontsize=14)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig("word_count_difference.png")
        plt.close()

        # 4. Heat map of model performance by directory
        directory_pivot = all_data_df.pivot_table(
            index=["directory", "model"],
            columns="metric",
            values="similarity",
            aggfunc="mean"
        ).reset_index()

        # Create individual heatmaps for each metric
        for metric in all_data_df["metric"].unique():
            plt.figure(figsize=(10, 8))
            pivot_data = directory_pivot.pivot(index="directory", columns="model", values=metric)
            sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': f'{metric} Similarity'})
            plt.title(f"OCR Performance by Directory - {metric}", fontsize=16)
            plt.ylabel("Directory", fontsize=14)
            plt.xlabel("OCR Model", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"directory_comparison_{metric}.png")
            plt.close()

        print("Plots saved to current directory.")