#!/usr/bin/env python3
"""
Terminal-based trajectory viewer for ULD distillation training
Shows loss trajectory as ASCII plot in terminal
"""

import json
import os
from typing import List, Tuple
from pathlib import Path

class TerminalPlotter:
    """ASCII plotting in terminal"""
    
    def __init__(self, width: int = 80, height: int = 20):
        self.width = width
        self.height = height
    
    def plot_line(self, data: List[float], title: str = "Training Loss"):
        """Create ASCII line plot"""
        if not data or len(data) < 2:
            print("❌ Not enough data points to plot")
            return
        
        # Normalize data to fit plot dimensions
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Create plot grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Plot data points
        for i, value in enumerate(data):
            if i >= self.width:
                break
            
            # Normalize to grid coordinates
            y = int(self.height - 1 - ((value - min_val) / range_val) * (self.height - 1))
            y = max(0, min(self.height - 1, y))
            
            # Mark the point
            if i == 0:
                grid[y][i] = '●'
            else:
                # Connect to previous point with line
                prev_value = data[i-1]
                prev_y = int(self.height - 1 - ((prev_value - min_val) / range_val) * (self.height - 1))
                prev_y = max(0, min(self.height - 1, prev_y))
                
                # Draw line between points
                start_y, end_y = (prev_y, y) if prev_y <= y else (y, prev_y)
                for line_y in range(start_y, end_y + 1):
                    if line_y == y:
                        grid[line_y][i] = '●'
                    else:
                        grid[line_y][i] = '│' if grid[line_y][i] == ' ' else grid[line_y][i]
        
        # Print the plot
        print(f"\n📊 {title}")
        print(f"{'─' * (self.width + 10)}")
        print(f"Max: {max_val:.4f}")
        
        for row in grid:
            print('│', ''.join(row), '│')
        
        print(f"{'─' * (self.width + 10)}")
        print(f"Min: {min_val:.4f}")
        print(f"Steps: {len(data)} | Range: {range_val:.4f}")
    
    def plot_histogram(self, data: List[float], bins: int = 20, title: str = "Loss Distribution"):
        """Create ASCII histogram"""
        if not data:
            print("❌ No data to plot")
            return
        
        # Create bins
        min_val, max_val = min(data), max(data)
        range_val = max_val - min_val if max_val != min_val else 1
        bin_width = range_val / bins
        
        # Count values in each bin
        counts = [0] * bins
        for value in data:
            bin_idx = min(bins - 1, int((value - min_val) / bin_width))
            counts[bin_idx] += 1
        
        # Normalize for display
        max_count = max(counts) if counts else 1
        bar_height = 10
        
        print(f"\n📊 {title}")
        print(f"{'─' * 50}")
        
        for i in range(bar_height, 0, -1):
            line = ""
            for count in counts:
                if count >= (i * max_count / bar_height):
                    line += "█"
                else:
                    line += " "
            print(f"{i:2d} │{line}│")
        
        print(f"   └{'─' * bins}┘")
        print(f"   {min_val:.3f}{'':>{bins-8}}{max_val:.3f}")

def load_training_data(checkpoint_dir: str = "./distillation_checkpoints") -> dict:
    """Load training data from checkpoint"""
    state_file = Path(checkpoint_dir) / "training_state.json"
    
    if not state_file.exists():
        print(f"❌ No training data found at {state_file}")
        return {}
    
    with open(state_file, 'r') as f:
        return json.load(f)

def analyze_trajectory():
    """Analyze and display training trajectory"""
    print("🔍 Loading training trajectory...")
    
    data = load_training_data()
    if not data:
        return
    
    # Extract trajectory data
    history = data.get("training_history", [])
    if not history:
        print("❌ No training history found")
        return
    
    # Extract losses
    losses = [entry["loss"] for entry in history if entry.get("loss") is not None]
    phases = [entry["phase"] for entry in history]
    examples = [entry["example"] for entry in history]
    modes = [entry["mode"] for entry in history]
    
    # Print summary
    print(f"\n📈 Training Summary:")
    print(f"   Total Steps: {len(history)}")
    print(f"   Current Phase: {data.get('current_phase', 'N/A')}")
    print(f"   Examples Processed: {data.get('total_examples_processed', 0)}")
    print(f"   API Calls: {data.get('api_calls_made', 0)}")
    print(f"   Est. Cost: ${data.get('estimated_cost', 0):.2f}")
    
    if losses:
        avg_loss = sum(losses) / len(losses)
        recent_losses = losses[-10:] if len(losses) >= 10 else losses
        recent_avg = sum(recent_losses) / len(recent_losses)
        
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Recent Loss (last 10): {recent_avg:.4f}")
        print(f"   Loss Range: {min(losses):.4f} - {max(losses):.4f}")
    
    # Create plots
    plotter = TerminalPlotter(width=60, height=15)
    
    if losses:
        # Full trajectory
        plotter.plot_line(losses, "Loss Trajectory (All Steps)")
        
        # Recent trajectory (last 30 points)
        if len(losses) > 30:
            recent = losses[-30:]
            plotter.plot_line(recent, "Recent Loss Trajectory (Last 30 Steps)")
        
        # Loss distribution
        plotter.plot_histogram(losses, bins=15, title="Loss Distribution")
    
    # Phase breakdown
    if phases:
        phase_counts = {}
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        print(f"\n📊 Phase Breakdown:")
        for phase, count in sorted(phase_counts.items()):
            bar = "█" * min(20, count // 2)
            print(f"   Phase {phase}: {count:3d} steps {bar}")
    
    # Mode breakdown
    if modes:
        mode_counts = {}
        for mode in modes:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        print(f"\n🎯 Training Mode Breakdown:")
        for mode, count in sorted(mode_counts.items()):
            percentage = (count / len(modes)) * 100
            bar = "█" * min(20, int(percentage // 2))
            print(f"   {mode:8s}: {count:3d} ({percentage:5.1f}%) {bar}")
    
    # Validation results
    validation_entries = [entry for entry in history if entry.get("validation")]
    if validation_entries:
        print(f"\n🔍 Latest Validation Results:")
        latest_val = validation_entries[-1]["validation"]
        total_tests = sum(latest_val.values())
        if total_tests > 0:
            coherent_pct = (latest_val.get("coherent", 0) / total_tests) * 100
            repetitive_pct = (latest_val.get("repetitive", 0) / total_tests) * 100
            failed_pct = (latest_val.get("failed", 0) / total_tests) * 100
            
            print(f"   Coherent:   {coherent_pct:5.1f}% {'█' * int(coherent_pct // 5)}")
            print(f"   Repetitive: {repetitive_pct:5.1f}% {'█' * int(repetitive_pct // 5)}")
            print(f"   Failed:     {failed_pct:5.1f}% {'█' * int(failed_pct // 5)}")

def real_time_monitor(refresh_seconds: int = 5):
    """Real-time trajectory monitoring"""
    import time
    import os
    
    print("🚀 Real-Time ULD Training Monitor")
    print("=" * 50)
    print(f"🔄 Refreshing every {refresh_seconds} seconds...")
    print("Press Ctrl+C to stop")
    print()
    
    last_step_count = 0
    
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🚀 Real-Time ULD Training Monitor")
            print("=" * 50)
            print(f"🔄 Auto-refresh: {refresh_seconds}s | Press Ctrl+C to stop")
            print()
            
            # Load and analyze current data
            data = load_training_data()
            if not data:
                print("❌ No training data found - waiting for training to start...")
                time.sleep(refresh_seconds)
                continue
            
            history = data.get("training_history", [])
            current_step_count = len(history)
            
            # Show if new steps were added
            if current_step_count > last_step_count:
                new_steps = current_step_count - last_step_count
                print(f"🆕 {new_steps} new training steps detected!")
            elif current_step_count == last_step_count and last_step_count > 0:
                print("⏸️  No new training steps (training may be paused)")
            
            last_step_count = current_step_count
            
            # Run analysis
            analyze_trajectory_compact()
            
            print(f"\n⏰ Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🔄 Next refresh in {refresh_seconds} seconds...")
            
            time.sleep(refresh_seconds)
            
    except KeyboardInterrupt:
        print("\n👋 Real-time monitoring stopped")

def analyze_trajectory_compact():
    """Compact version for real-time monitoring"""
    data = load_training_data()
    if not data:
        return
    
    history = data.get("training_history", [])
    if not history:
        print("❌ No training history found")
        return
    
    # Extract recent data
    losses = [entry["loss"] for entry in history if entry.get("loss") is not None]
    
    # Current status
    current_phase = data.get('current_phase', 'N/A')
    current_example = data.get('current_example', 0)
    total_processed = data.get('total_examples_processed', 0)
    api_calls = data.get('api_calls_made', 0)
    cost = data.get('estimated_cost', 0)
    
    print(f"📊 Status: Phase {current_phase}, Example {current_example}")
    print(f"📈 Processed: {total_processed} | API Calls: {api_calls} | Cost: ${cost:.2f}")
    
    if losses:
        recent_10 = losses[-10:]
        recent_avg = sum(recent_10) / len(recent_10)
        latest_loss = losses[-1]
        
        print(f"💹 Latest Loss: {latest_loss:.4f}")
        print(f"📊 Recent Avg (10): {recent_avg:.4f}")
        
        # Mini trajectory (last 20 points)
        if len(losses) >= 20:
            mini_losses = losses[-20:]
            plotter = TerminalPlotter(width=40, height=8)
            plotter.plot_line(mini_losses, "Recent Loss Trend (Last 20)")
    
    # Latest validation if available
    validation_entries = [entry for entry in history if entry.get("validation")]
    if validation_entries:
        latest_val = validation_entries[-1]["validation"]
        coherent = latest_val.get("coherent", 0)
        repetitive = latest_val.get("repetitive", 0)
        failed = latest_val.get("failed", 0)
        total = coherent + repetitive + failed
        
        if total > 0:
            coherent_pct = (coherent / total) * 100
            print(f"🔍 Validation: {coherent_pct:.0f}% coherent, {repetitive} repetitive, {failed} failed")

def main():
    """Main function with options"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ['--real-time', '-r', '--monitor', '-m']:
        # Real-time monitoring mode
        refresh_rate = 5
        if len(sys.argv) > 2:
            try:
                refresh_rate = int(sys.argv[2])
            except ValueError:
                pass
        
        real_time_monitor(refresh_rate)
    else:
        # Static analysis mode
        print("🚀 ULD Training Trajectory Analyzer")
        print("=" * 50)
        print("💡 Tip: Use --real-time or -r for live monitoring")
        print()
        
        try:
            analyze_trajectory()
        except KeyboardInterrupt:
            print("\n👋 Analysis interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            
        print(f"\n💡 For real-time monitoring, run:")
        print(f"   python trajectory.py --real-time")
        print(f"   python trajectory.py -r 3  # refresh every 3 seconds")

if __name__ == "__main__":
    main()
