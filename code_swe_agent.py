#!/usr/bin/env python3
"""
SWE-bench agent capable of using Claude Code or Codex backends.
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import jsonlines

from utils.claude_interface import ClaudeCodeInterface
from utils.codex_interface import CodexCodeInterface
from utils.gemini_interface import GeminiCodeInterface
from utils.prompt_formatter import PromptFormatter
from utils.patch_extractor import PatchExtractor
from utils.run_artifacts import run_timestamp, save_instance_artifacts


DEFAULT_BACKEND = os.environ.get("CODE_SWE_BACKEND", "claude")


def _process_instance_worker(
    instance: Dict,
    prompt_template: Optional[str],
    model: Optional[str],
    backend: str,
    base_dir: str,
    output_dir: Optional[str],
) -> Dict:
    """Run one instance in a worker process (each gets its own CLI session)."""
    instance_id = instance["instance_id"]
    print(f"[pid {os.getpid()}] starting {instance_id}", flush=True)
    os.chdir(base_dir)
    agent = CodeSWEAgent(
        prompt_template,
        model,
        backend,
        output_dir=Path(output_dir) if output_dir else None,
    )
    return agent.process_instance(instance)


class CodeSWEAgent:
    """Main agent for running SWE-bench using different code models."""

    def __init__(self, prompt_template: Optional[str] = None,
                 model: Optional[str] = None,
                 backend: str = DEFAULT_BACKEND,
                 output_dir: Optional[Path] = None):
        self.backend = (backend or DEFAULT_BACKEND).lower()
        if self.backend == "codex":
            self.interface = CodexCodeInterface()
        elif self.backend == "gemini":
            self.interface = GeminiCodeInterface()
        else:
            self.backend = "claude"
            self.interface = ClaudeCodeInterface()

        self.prompt_formatter = PromptFormatter(prompt_template)
        self.prompt_template = prompt_template
        self.patch_extractor = PatchExtractor()
        self.base_dir = Path.cwd()
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.output_dir if self.output_dir else self.base_dir / "predictions"

        # Resolve model name from alias
        self.model = get_model_name(model, self.backend) if model else None
        self.model_alias = model  # Keep original alias for logging

        # Create directories if they don't exist
        self.predictions_dir.mkdir(exist_ok=True)
        self.pred_timestamp: Optional[str] = None
        self.pred_file: Optional[Path] = None
        self.run_timestamp: Optional[str] = None

    def setup_repository(self, instance: Dict) -> Optional[str]:
        """Set up a repository for testing."""
        instance_id = instance["instance_id"]
        repo_name = instance["repo"]
        base_commit = instance["base_commit"]

        # Create temporary directory for this instance (cross-platform)
        temp_dir = Path(tempfile.gettempdir()) / f"swe_bench_{instance_id}"

        try:
            # Remove if exists
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            # Save current directory
            original_dir = Path.cwd()
            
            # Clone repository
            print(f"Cloning {repo_name} to {temp_dir}")
            clone_url = f"https://github.com/{repo_name}.git"
            
            result = subprocess.run(
                ["git", "clone", clone_url, str(temp_dir)],
                capture_output=True,
                text=True,
                cwd=str(original_dir)  # Ensure we're in a valid directory
            )
            
            if result.returncode != 0:
                print(f"Failed to clone repository: {result.stderr}")
                return None
                
            # Checkout base commit
            os.chdir(temp_dir)
            result = subprocess.run(
                ["git", "checkout", base_commit],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Failed to checkout commit: {result.stderr}")
                os.chdir(str(original_dir))  # Return to original directory
                return None

            os.chdir(str(original_dir))  # Return to original directory
            return str(temp_dir)
            
        except Exception as e:
            print(f"Error setting up repository: {e}")
            # Try to return to original directory if possible
            try:
                os.chdir(str(original_dir))
            except Exception as chdir_error:
                print(f"Warning: Failed to return to original directory: {chdir_error}")
            return None
            
    def process_instance(self, instance: Dict) -> Dict:
        """Process a single SWE-bench instance."""
        instance_id = instance["instance_id"]
        print(f"\nProcessing {instance_id}")

        original_dir = os.getcwd()

        repo_path = self.setup_repository(instance)
        if not repo_path:
            prediction = {
                "instance_id": instance_id,
                "model": f"{self.backend}-code",
                "prediction": "",
                "error": "Failed to set up repository",
            }
            self._save_instance_artifacts(instance_id, {}, "", repo_path or "")
            return prediction

        cli_result: Dict = {}
        patch = ""
        try:
            prompt = self.prompt_formatter.format_for_cli(instance)

            os.chdir(repo_path)
            subprocess.run(["git", "add", "-A"], capture_output=True)
            subprocess.run(["git", "stash"], capture_output=True)

            model_info = f" with model {self.model_alias}" if self.model else ""
            print(f"Running {self.backend.title()} Code{model_info}...")
            cli_result = self.interface.execute_code_cli(prompt, repo_path, self.model)

            if not cli_result["success"]:
                print(f"{self.backend.title()} Code execution failed: {cli_result['stderr']}")
                prediction = {
                    "instance_id": instance_id,
                    "model": self.model_alias or f"{self.backend}-code",
                    "prediction": "",
                    "error": f"Execution failed: {cli_result['stderr']}",
                }
                self._save_instance_artifacts(instance_id, cli_result, "", repo_path)
                return prediction

            patch = self.patch_extractor.extract_from_cli_output(cli_result["stdout"], repo_path)

            is_valid, error = self.patch_extractor.validate_patch(patch)
            if not is_valid:
                print(f"Invalid patch: {error}")
                patch = ""

            prediction = self.patch_extractor.format_for_swebench(
                patch, instance_id, self.model_alias or f"{self.backend}-code"
            )
            self._save_instance_artifacts(instance_id, cli_result, patch, repo_path)
            return prediction

        except Exception as e:
            import traceback
            print(f"Error processing instance: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            prediction = {
                "instance_id": instance_id,
                "model": self.model_alias or f"{self.backend}-code",
                "prediction": "",
                "error": str(e),
            }
            self._save_instance_artifacts(instance_id, cli_result, patch, repo_path)
            return prediction
        finally:
            try:
                os.chdir(original_dir)
            except Exception as e:
                print(f"Warning: Could not restore directory: {e}")

            if repo_path and os.path.exists(repo_path):
                shutil.rmtree(repo_path)

    def _save_instance_artifacts(
        self,
        instance_id: str,
        cli_result: Dict,
        patch: str,
        repo_path: str,
    ):
        """Write session/stream into output-dir if set."""
        if self.output_dir:
            save_instance_artifacts(
                self.output_dir,
                instance_id,
                cli_output=cli_result,
                backend=self.backend,
            )
            
    def run_on_dataset(self, dataset_name: str, split: str = "test",
                      limit: Optional[int] = None,
                      workers: int = 1) -> List[Dict]:
        """Run on a full dataset."""
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        instances = [dict(row) for row in dataset]
        workers = max(1, workers)
            
        self.pred_timestamp = run_timestamp()
        self.run_timestamp = self.pred_timestamp
        self.pred_file = self.predictions_dir / (
            "predictions.jsonl" if self.output_dir else f"predictions_{self.pred_timestamp}.jsonl"
        )
        if self.pred_file.exists():
            self.pred_file.unlink()
        json_name = "predictions.json" if self.output_dir else f"predictions_{self.pred_timestamp}.json"
        json_file = self.predictions_dir / json_name
        if json_file.exists():
            json_file.unlink()

        predictions: List[Dict] = []
        base_dir = str(self.base_dir.resolve())
        output_dir = str(self.output_dir) if self.output_dir else None

        if workers == 1:
            for instance in tqdm(instances, desc="Processing instances"):
                prediction = self.process_instance(instance)
                predictions.append(prediction)
                self._save_predictions(prediction)
        else:
            print(f"Running {len(instances)} instances with {workers} parallel workers...")
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _process_instance_worker,
                        instance,
                        self.prompt_template,
                        self.model_alias,
                        self.backend,
                        base_dir,
                        output_dir,
                    ): instance["instance_id"]
                    for instance in instances
                }
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="Processing instances"):
                    instance_id = futures[future]
                    try:
                        prediction = future.result()
                    except Exception as exc:
                        prediction = {
                            "instance_id": instance_id,
                            "model": self.model_alias or f"{self.backend}-code",
                            "prediction": "",
                            "error": str(exc),
                        }
                    predictions.append(prediction)
                    self._save_predictions(prediction)

        with open(json_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"Saved predictions to {self.pred_file}")
        if self.output_dir:
            print(f"Saved per-instance artifacts under {self.output_dir}")
        return predictions
    
    def run_on_instance(self, instance_id: str, dataset_name: str = "princeton-nlp/SWE-bench_Lite") -> Dict:
        """Run on a single instance by ID."""
        self.run_timestamp = run_timestamp()
        dataset = load_dataset(dataset_name, split="test")
        
        # Find the instance
        instance = None
        for item in dataset:
            if item["instance_id"] == instance_id:
                instance = item
                break
                
        if not instance:
            raise ValueError(f"Instance {instance_id} not found in dataset")
            
        return self.process_instance(instance)
    
    def _save_predictions(self, prediction: Dict):
        """Append a single prediction to the jsonl file."""
        if not self.pred_file:
            raise ValueError("Prediction timestamp not initialized. Call run_on_dataset first.")

        with jsonlines.open(self.pred_file, mode='a') as writer:
            writer.write(prediction)


def main():
    parser = argparse.ArgumentParser(description="Run code models on SWE-bench")
    parser.add_argument("--dataset_name", type=str,
                       default="princeton-nlp/SWE-bench_Lite",
                       help="Dataset to use")
    parser.add_argument("--instance_id", type=str,
                       help="Run on a specific instance ID")
    parser.add_argument("--limit", type=int,
                       help="Limit number of instances to process")
    parser.add_argument("--prompt_template", type=str,
                       help="Path to custom prompt template")
    parser.add_argument("--model", type=str,
                       help="Model to use (e.g., opus-4.1, codex-4.2, or any name)")
    parser.add_argument("--backend", type=str, choices=["claude", "codex", "gemini"],
                       help="Code model backend to use")
    parser.add_argument("--workers", type=int, default=1,
                       help="Parallel workers for patch generation (default: 1)")
    parser.add_argument("-o", "--output-dir", type=str,
                       help="Run output directory; writes per-instance *.session/stream.json")
    
    args = parser.parse_args()
    
    backend = args.backend or DEFAULT_BACKEND

    # Check if selected CLI is available
    if backend == "codex":
        cli_cmd = "codex"
    elif backend == "gemini":
        cli_cmd = "gemini"
    else:
        cli_cmd = "claude"
    try:
        result = subprocess.run([cli_cmd, "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {cli_cmd} CLI not found. Please ensure '{cli_cmd}' is installed and in PATH")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {cli_cmd} CLI not found. Please ensure '{cli_cmd}' is installed and in PATH")
        sys.exit(1)

    agent = CodeSWEAgent(
        args.prompt_template,
        args.model,
        backend,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    if not agent.output_dir:
        ds_tag = args.dataset_name.split("/")[-1].replace("_", "-").lower()
        auto_dir = Path("runs") / f"{backend}-{ds_tag}-{run_timestamp()}"
        agent.output_dir = auto_dir.resolve()
        agent.output_dir.mkdir(parents=True, exist_ok=True)
        agent.predictions_dir = agent.output_dir
        print(f"Auto output directory: {auto_dir}")
    
    # Run on specific instance or dataset
    if args.instance_id:
        print(f"Running on instance: {args.instance_id}")
        prediction = agent.run_on_instance(args.instance_id, args.dataset_name)
        print(f"Prediction saved: {prediction}")
    else:
        print(f"Running on dataset: {args.dataset_name}")
        predictions = agent.run_on_dataset(
            args.dataset_name,
            limit=args.limit,
            workers=args.workers,
        )
        print(f"Processed {len(predictions)} instances")


if __name__ == "__main__":
    main()
