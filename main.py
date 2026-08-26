# main.py
"""
Entry point for Phase 2 of the Smart Scan project.
Evaluates Round-Robin and Random scanning baseline strategies.
"""
import random
from simulator.config import RANDOM_SEED
from simulator.emitters import ContinuousEmitter, PeriodicEmitter, AgileEmitter, BurstEmitter
from simulator.environment import RFEnvironment
from receiver.receiver import Receiver
from receiver.round_robin import RoundRobinScheduler
from receiver.random_scan import RandomScheduler
from metrics.metrics import MetricsCalculator

def create_environment():
    """Creates a fresh, reproducible RF environment."""
    random.seed(RANDOM_SEED)
    emitters = [
        ContinuousEmitter(name="Radar-Continuous", band_index=0), # B1
        PeriodicEmitter(name="Comm-Periodic", band_index=2, period=5, active_duration=2), # B3
        AgileEmitter(name="Hopping-Agile", band_indices=[1, 3, 4], hop_interval=3), # B2, B4, B5
        BurstEmitter(name="Data-Burst", band_index=1, burst_probability=0.2, max_burst_duration=3) # B2
    ]
    return RFEnvironment(emitters=emitters)

def run_simulation(scheduler_class, num_steps=100):
    """Runs the simulation with a given scheduler and returns metrics and history."""
    env = create_environment()
    scheduler = scheduler_class()
    receiver = Receiver()
    
    ground_truth_history = []
    
    for _ in range(num_steps):
        state = env.step()
        ground_truth_history.append(state)
        
        t = state["time"]
        selected_band = scheduler.get_next_band(t)
        
        # Receiver performs scan by requesting observation of ONLY the selected band
        receiver.scan(t, selected_band, state)
        
    metrics_calc = MetricsCalculator(ground_truth_history, receiver.scan_history)
    metrics = metrics_calc.calculate_metrics()
    
    return receiver.scan_history, metrics

def print_scan_history(history, num_records=20):
    """Prints a short scan history."""
    print(f"{'Time':<5} | {'Scanned Band':<15} | {'Emitter':<25} | {'Result'}")
    print("-" * 60)
    for record in history[:num_records]:
        t = record['time_step']
        band = f"Scanned {record['selected_band']}"
        emitter = record['emitter_id'] if record['emitter_id'] else "No Signal"
        result = record['result']
        print(f"T={t:<3} | {band:<15} | {emitter:<25} | {result}")
    print("-" * 60)

def main():
    print("--- Smart Scan Phase 2: Baseline Scanning Strategies ---")
    
    num_steps = 100
    
    print("\nRunning Round-Robin Scheduler...")
    rr_history, rr_metrics = run_simulation(RoundRobinScheduler, num_steps)
    print_scan_history(rr_history, num_records=20)
    
    print("\nRunning Random Scheduler...")
    rand_history, rand_metrics = run_simulation(RandomScheduler, num_steps)
    print_scan_history(rand_history, num_records=20)
    
    # Print Comparison Table
    print("\n================================================")
    print("SCAN STRATEGY COMPARISON")
    print(f"{'Metric':<25} {'Round Robin':<13} {'Random'}")
    print("-" * 48)
    print(f"{'Total Scans':<25} {rr_metrics['total_scans']:<13} {rand_metrics['total_scans']}")
    print(f"{'Hits':<25} {rr_metrics['total_hits']:<13} {rand_metrics['total_hits']}")
    print(f"{'Misses':<25} {rr_metrics['total_misses']:<13} {rand_metrics['total_misses']}")
    print(f"{'Hit Rate':<25} {rr_metrics['hit_rate']:<12.2f}% {rand_metrics['hit_rate']:.2f}%")
    print(f"{'Interception Rate':<25} {rr_metrics['interception_rate']:<12.2f}% {rand_metrics['interception_rate']:.2f}%")
    print(f"{'Average Intercept Delay':<25} {rr_metrics['average_intercept_delay']:<13.2f} {rand_metrics['average_intercept_delay']:.2f}")
    print(f"{'Missed Transmissions':<25} {rr_metrics['missed_transmissions']:<13} {rand_metrics['missed_transmissions']}")
    print("================================================\n")

if __name__ == "__main__":
    main()
