#!/usr/bin/env python3
"""
Real-time training monitor for 3D Fast-DDPM BraTS
Monitor training progress and provide real-time recommendations
"""

import time
import os
import re
import argparse
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class TrainingMonitor:
    """Monitor training progress in real-time"""
    
    def __init__(self, log_file=None, window_size=50):
        self.log_file = log_file
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.steps = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        self.target_indices = deque(maxlen=window_size)
        
    def parse_log_line(self, line):
        """Parse training log line to extract metrics"""
        # Pattern for your training logs
        pattern = r'Step (\d+) - Loss: ([\d.]+), Target: (\d+), LR: ([\d.e-]+)'
        match = re.search(pattern, line)
        
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            target_idx = int(match.group(3))
            lr = float(match.group(4))
            return step, loss, target_idx, lr
        return None
    
    def update_metrics(self, step, loss, target_idx, lr):
        """Update metrics with new values"""
        self.steps.append(step)
        self.losses.append(loss)
        self.target_indices.append(target_idx)
        self.learning_rates.append(lr)
    
    def analyze_current_state(self):
        """Analyze current training state"""
        if len(self.losses) < 5:
            return "Not enough data for analysis"
        
        recent_losses = list(self.losses)[-10:]
        recent_steps = list(self.steps)[-10:]
        
        # Calculate trend
        if len(recent_losses) >= 2:
            trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        else:
            trend = 0
        
        # Calculate statistics
        current_loss = recent_losses[-1]
        loss_std = np.std(recent_losses)
        avg_loss = np.mean(recent_losses)
        
        # Generate analysis
        analysis = []
        analysis.append(f"📊 CURRENT METRICS:")
        analysis.append(f"  Current Loss: {current_loss:.4f}")
        analysis.append(f"  Average (last 10): {avg_loss:.4f}")
        analysis.append(f"  Volatility: {loss_std:.4f}")
        analysis.append(f"  Trend: {'📉 Decreasing' if trend < 0 else '📈 Increasing'} ({trend:.6f})")
        
        # Status assessment
        if current_loss > 0.8:
            status = "🔴 HIGH - Early training"
        elif current_loss > 0.5:
            status = "🟡 MEDIUM - Making progress"
        elif current_loss > 0.3:
            status = "🟢 GOOD - Converging well"
        else:
            status = "✅ EXCELLENT - Near convergence"
        
        analysis.append(f"  Status: {status}")
        
        # Recommendations
        analysis.append(f"\n💡 RECOMMENDATIONS:")
        if loss_std > 0.1:
            analysis.append("  ⚠️  High volatility - consider reducing LR")
        if trend > 0 and len(recent_losses) > 5:
            analysis.append("  ❌ Loss increasing - check for overfitting or high LR")
        if current_loss > 0.7 and recent_steps[-1] > 1000:
            analysis.append("  🔧 Slow convergence - verify normalization and model")
        if trend < -0.001:
            analysis.append("  ✅ Good convergence rate - continue training")
        
        return "\n".join(analysis)
    
    def predict_convergence(self):
        """Predict when training will converge"""
        if len(self.losses) < 10:
            return "Not enough data for prediction"
        
        recent_losses = list(self.losses)[-20:]
        recent_steps = list(self.steps)[-20:]
        
        if len(recent_losses) < 5:
            return "Need more data points"
        
        # Fit exponential decay or linear trend
        try:
            trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            current_loss = recent_losses[-1]
            
            predictions = []
            targets = [0.5, 0.3, 0.1]
            
            for target in targets:
                if trend < 0 and current_loss > target:
                    steps_needed = (current_loss - target) / abs(trend)
                    current_step = recent_steps[-1]
                    target_step = current_step + steps_needed
                    predictions.append(f"  Loss {target}: ~Step {target_step:.0f}")
                elif current_loss <= target:
                    predictions.append(f"  Loss {target}: Already achieved! ✅")
                else:
                    predictions.append(f"  Loss {target}: Cannot predict (trend not favorable)")
            
            return "🔮 CONVERGENCE PREDICTIONS:\n" + "\n".join(predictions)
        
        except Exception as e:
            return f"Prediction error: {e}"
    
    def monitor_file(self, follow=True):
        """Monitor log file for new entries"""
        if not self.log_file or not os.path.exists(self.log_file):
            print("No log file specified or file doesn't exist")
            return
        
        print(f"Monitoring: {self.log_file}")
        print("Press Ctrl+C to stop monitoring\n")
        
        with open(self.log_file, 'r') as f:
            # Go to end of file
            f.seek(0, 2)
            
            while follow:
                line = f.readline()
                if line:
                    parsed = self.parse_log_line(line)
                    if parsed:
                        step, loss, target_idx, lr = parsed
                        self.update_metrics(step, loss, target_idx, lr)
                        
                        # Print analysis every 10 steps
                        if step % 50 == 0:
                            print(f"\n{'-'*60}")
                            print(f"UPDATE AT STEP {step}")
                            print(f"{'-'*60}")
                            print(self.analyze_current_state())
                            print(f"\n{self.predict_convergence()}")
                            print(f"{'-'*60}\n")
                
                else:
                    time.sleep(1)
    
    def generate_plots(self, save_path="training_plots.png"):
        """Generate training progress plots"""
        if len(self.losses) < 5:
            print("Not enough data for plotting")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        steps_list = list(self.steps)
        losses_list = list(self.losses)
        lrs_list = list(self.learning_rates)
        targets_list = list(self.target_indices)
        
        # Loss curve
        ax1.plot(steps_list, losses_list, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Progress')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Learning rate
        ax2.plot(steps_list, lrs_list, 'r-', linewidth=2, label='Learning Rate')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_yscale('log')
        
        # Loss distribution
        ax3.hist(losses_list, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Loss Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Target modality distribution
        target_counts = np.bincount(targets_list)
        modalities = ['T1n', 'T1c', 'T2w', 'T2f']
        ax4.bar(range(len(target_counts)), target_counts, 
                tick_label=[modalities[i] for i in range(len(target_counts))])
        ax4.set_xlabel('Target Modality')
        ax4.set_ylabel('Count')
        ax4.set_title('Target Modality Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Monitor 3D Fast-DDPM training')
    parser.add_argument('--log_file', type=str, help='Path to training log file')
    parser.add_argument('--follow', action='store_true', help='Follow log file in real-time')
    parser.add_argument('--analysis_only', action='store_true', help='Only run analysis on existing data')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_file)
    
    if args.analysis_only and args.log_file:
        # Analyze existing log file
        with open(args.log_file, 'r') as f:
            for line in f:
                parsed = monitor.parse_log_line(line)
                if parsed:
                    step, loss, target_idx, lr = parsed
                    monitor.update_metrics(step, loss, target_idx, lr)
        
        print("ANALYSIS OF EXISTING LOG:")
        print("="*50)
        print(monitor.analyze_current_state())
        print(f"\n{monitor.predict_convergence()}")
        monitor.generate_plots()
    
    elif args.follow and args.log_file:
        monitor.monitor_file(follow=True)
    
    else:
        # Manual analysis mode
        print("=== 3D FAST-DDPM TRAINING MONITOR ===")
        print("\nBased on your current progress:")
        print("- Loss: 0.8809 → 0.5902 (33% improvement)")
        print("- Trend: Decreasing (-0.000407/step)")
        print("- Predicted steps to loss 0.3: ~712")
        print("- Status: ✅ Training progressing normally")
        
        print("\n🎯 IMMEDIATE ACTIONS:")
        print("1. Continue training with current settings")
        print("2. Monitor for next 500-1000 steps")
        print("3. Look for loss to reach ~0.4 range")
        print("4. Check sample quality after 2000 steps")
        
        print("\n💡 OPTIMIZATION TIPS:")
        print("- Your training speed (1.54 it/s) is reasonable")
        print("- Consider increasing batch size if GPU memory allows")
        print("- Monitor GPU utilization to ensure full usage")


if __name__ == '__main__':
    main()
