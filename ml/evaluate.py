# ml/evaluate.py
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from simulator.config import RANDOM_SEED
from simulator.emitters import ContinuousEmitter, PeriodicEmitter, AgileEmitter, BurstEmitter
from simulator.environment import RFEnvironment
from receiver.receiver import Receiver
from receiver.round_robin import RoundRobinScheduler
from receiver.random_scan import RandomScheduler
from receiver.heuristic_scheduler import AdaptiveHeuristicScheduler
from ml.rl_scheduler import RLScheduler
from metrics.metrics import MetricsCalculator

def create_environment(seed, scenario_name="Mixed"):
    random.seed(seed)
    
    if scenario_name == "Mostly Periodic":
        emitters = [
            PeriodicEmitter(name="Comm-Periodic-1", band_index=0, period=4, active_duration=2),
            PeriodicEmitter(name="Comm-Periodic-2", band_index=2, period=7, active_duration=3),
            PeriodicEmitter(name="Comm-Periodic-3", band_index=4, period=10, active_duration=2)
        ]
    elif scenario_name == "Random Burst":
        emitters = [
            BurstEmitter(name="Data-Burst-1", band_index=1, burst_probability=0.3, max_burst_duration=2),
            BurstEmitter(name="Data-Burst-2", band_index=3, burst_probability=0.1, max_burst_duration=5),
            BurstEmitter(name="Data-Burst-3", band_index=4, burst_probability=0.2, max_burst_duration=3)
        ]
    elif scenario_name == "Frequency Agile":
        emitters = [
            AgileEmitter(name="Hopping-Agile-1", band_indices=[0, 1, 2], hop_interval=4),
            AgileEmitter(name="Hopping-Agile-2", band_indices=[2, 3, 4], hop_interval=2)
        ]
    else: # Mixed
        emitters = [
            ContinuousEmitter(name="Radar-Continuous", band_index=0),
            PeriodicEmitter(name="Comm-Periodic", band_index=2, period=5, active_duration=2),
            AgileEmitter(name="Hopping-Agile", band_indices=[1, 3, 4], hop_interval=3),
            BurstEmitter(name="Data-Burst", band_index=1, burst_probability=0.2, max_burst_duration=3)
        ]
        
    return RFEnvironment(emitters=emitters)

def run_simulation(scheduler, num_steps, seed, scenario):
    env = create_environment(seed, scenario)
    receiver = Receiver()
    ground_truth_history = []
    
    for _ in range(num_steps):
        state = env.step()
        ground_truth_history.append(state)
        
        t = state["time"]
        selected_band = scheduler.get_next_band(t)
        record = receiver.scan(t, selected_band, state)
        
        if hasattr(scheduler, 'update_feedback'):
            scheduler.update_feedback(t, selected_band, record['result'])
            
    metrics_calc = MetricsCalculator(ground_truth_history, receiver.scan_history)
    metrics = metrics_calc.calculate_metrics()
    return metrics, receiver.scan_history

def evaluate_models():
    print("--- Phase 4: Evaluating Schedulers Across Multiple Scenarios ---")
    num_steps = 1000
    num_episodes = 5
    
    strategies = {
        "Round Robin": lambda: RoundRobinScheduler(),
        "Random": lambda: RandomScheduler(),
        "Adaptive Heuristic": lambda: AdaptiveHeuristicScheduler(),
        "RL Agent (PPO)": lambda: RLScheduler()
    }
    
    scenarios = ["Mixed", "Mostly Periodic", "Random Burst", "Frequency Agile"]
    results = []
    
    for scenario in scenarios:
        print(f"\nEvaluating Scenario: {scenario}")
        for name, scheduler_factory in strategies.items():
            print(f"  -> Running {name}...")
            for ep in range(num_episodes):
                seed = RANDOM_SEED + 200 + ep # Unseen test seeds
                scheduler = scheduler_factory()
                metrics, history = run_simulation(scheduler, num_steps, seed, scenario)
                metrics["Scenario"] = scenario
                metrics["Strategy"] = name
                metrics["Episode"] = ep
                results.append(metrics)
            
    df = pd.DataFrame(results)
    
    # Save raw CSV
    csv_path = "results/rl_comparison.csv"
    df.to_csv(csv_path, index=False)
    
    # Print comparison table for each scenario
    for scenario in scenarios:
        summary = df[df["Scenario"] == scenario].groupby("Strategy").mean(numeric_only=True).reset_index()
        print(f"\n==========================================================================================")
        print(f"SCENARIO: {scenario} (Mean over 1000 steps, {num_episodes} episodes)")
        print(f"{'Metric':<20} {'Round Robin':<13} {'Random':<13} {'Heuristic':<13} {'RL Agent (PPO)'}")
        print("-" * 90)
        
        def get_val(df, strategy):
            return df[df["Strategy"] == strategy].iloc[0]
            
        rr = get_val(summary, "Round Robin")
        rand = get_val(summary, "Random")
        ah = get_val(summary, "Adaptive Heuristic")
        rl = get_val(summary, "RL Agent (PPO)")
        
        print(f"{'Hit Rate':<20} {rr['hit_rate']:<12.2f}% {rand['hit_rate']:<12.2f}% {ah['hit_rate']:<12.2f}% {rl['hit_rate']:.2f}%")
        print(f"{'Interception Rate':<20} {rr['interception_rate']:<12.2f}% {rand['interception_rate']:<12.2f}% {ah['interception_rate']:<12.2f}% {rl['interception_rate']:.2f}%")
        print(f"{'Average Delay':<20} {rr['average_intercept_delay']:<13.2f} {rand['average_intercept_delay']:<13.2f} {ah['average_intercept_delay']:<13.2f} {rl['average_intercept_delay']:.2f}")
        print(f"{'Miss Percentage':<20} {rr['percentage_of_missed_transmissions']:<12.2f}% {rand['percentage_of_missed_transmissions']:<12.2f}% {ah['percentage_of_missed_transmissions']:<12.2f}% {rl['percentage_of_missed_transmissions']:.2f}%")
        print("==========================================================================================\n")
    
    generate_charts(df, scenarios)

def generate_charts(df, scenarios):
    charts_dir = "results/charts"
    os.makedirs(charts_dir, exist_ok=True)
    
    # 1. Hit Rate Comparison across scenarios
    plt.figure(figsize=(12, 6))
    
    # Calculate means
    means = df.groupby(["Scenario", "Strategy"])["hit_rate"].mean().unstack()
    
    ax = means.plot(kind='bar', figsize=(12, 6), colormap='viridis')
    plt.title("Hit Rate Comparison Across Scenarios")
    plt.ylabel("Hit Rate (%)")
    plt.xlabel("Simulation Scenario")
    plt.xticks(rotation=0)
    plt.legend(title="Strategy")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "rl_scenarios_hit_rate.png"))
    plt.close()
    
    # 2. Average Intercept Delay
    plt.figure(figsize=(12, 6))
    means_delay = df.groupby(["Scenario", "Strategy"])["average_intercept_delay"].mean().unstack()
    ax = means_delay.plot(kind='bar', figsize=(12, 6), colormap='viridis')
    plt.title("Average Intercept Delay Across Scenarios (Lower is Better)")
    plt.ylabel("Delay (time steps)")
    plt.xlabel("Simulation Scenario")
    plt.xticks(rotation=0)
    plt.legend(title="Strategy")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "rl_scenarios_avg_delay.png"))
    plt.close()
    
    print(f"Scenario comparison charts saved to: {charts_dir}")

if __name__ == "__main__":
    evaluate_models()
