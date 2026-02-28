import argparse
import os
import json
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot Scaling Laws")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing log JSON files")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory '{args.log_dir}' not found. Run training first.")
        return

    plt.figure(figsize=(10, 6))
    
    files_found = False
    for filename in os.listdir(args.log_dir):
        if filename.startswith("scaling_") and filename.endswith(".json"):
            scale_name = filename.replace("scaling_", "").replace(".json", "")
            filepath = os.path.join(args.log_dir, filename)
            
            try:
                with open(filepath, "r") as f:
                    logs = json.load(f)
                
                steps = [entry["step"] for entry in logs]
                losses = [entry["loss"] for entry in logs]
                
                plt.plot(steps, losses, label=f"Scale: {scale_name}")
                files_found = True
            except Exception as e:
                print(f"Failed to read {filepath}: {e}")

    if not files_found:
        print(f"No valid log files found in {args.log_dir}.")
        return

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Progressive Scaling: Loss vs Steps")
    plt.legend()
    plt.grid(True)
    
    output_path = "scaling_plot.png"
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")

if __name__ == "__main__":
    main()
