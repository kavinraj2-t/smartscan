import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.rf_scan_env import RFScanEnv
from evaluation.evaluate_rl import evaluate_rl_model
from evaluation.evaluate import evaluate_scheduler
from baselines.random_scan import RandomScheduler
from baselines.round_robin import RoundRobinScheduler

def main():
    print("==================================================")
    print("PHASE 11: EVALUATING GENERALIZED MODEL ON UNSEEN DATA")
    print("==================================================")
    
    base_dir = "data/raw/datasets--alan-turing-institute--turing-synthetic-radar-dataset/snapshots/68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51/scan/train_scan"
    test_file = os.path.join(base_dir, "config_6.h5")
    
    print("\n--- GENERALIZED RL AGENT ---")
    evaluate_rl_model(
        filepath=test_file,
        model_path="models/Model_Generalized_5Files_final.zip",
        name="Generalized RL Agent (Stochastic Policy)",
        enable_exploration=False,
        enable_repeat_penalty=False,
        norm_stats_path="environment/norm_generalized.json",
        deterministic=False
    )
    
    print("\n--- BASELINES ---")
    # For baselines, normalization stats don't technically matter as they ignore the observation,
    # but we must initialize the environment anyway.
    env_test = RFScanEnv(
        filepaths=test_file,
        enable_exploration=False,
        enable_repeat_penalty=False,
        norm_stats_path="environment/norm_generalized.json"
    )
    
    evaluate_scheduler(env_test, RoundRobinScheduler(10), "Round Robin Baseline")
    evaluate_scheduler(env_test, RandomScheduler(10), "Random Scan Baseline")
    
    print("==================================================")

if __name__ == "__main__":
    main()
