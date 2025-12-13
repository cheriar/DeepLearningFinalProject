"""
Quick script to check if training is done and send results
Run this manually when you think training might be complete
"""
import os
import sys
from monitor_and_email import (
    check_training_complete, 
    get_training_stats, 
    analyze_predictor, 
    generate_analysis_report
)

def main():
    print("Checking training status...")
    
    if not check_training_complete():
        print("✗ Training not complete yet.")
        print("  Final checkpoint not found.")
        print("  Training may still be running.")
        return
    
    print("✓ Training completed!")
    print("\nGenerating analysis report...")
    
    # Get stats
    stats = get_training_stats()
    print(f"  Best accuracy: {stats.get('best_accuracy', 'N/A')}%")
    print(f"  Final accuracy: {stats.get('final_accuracy', 'N/A')}%")
    
    # Analyze predictor
    predictor_analysis = None
    if stats.get('best_checkpoint'):
        print("Analyzing predictor...")
        predictor_analysis = analyze_predictor(stats['best_checkpoint'])
        if predictor_analysis:
            print(f"  Sigma range: {predictor_analysis['sigma_range']}")
    
    # Generate report
    report = generate_analysis_report(stats, predictor_analysis)
    
    # Save report
    report_path = "training_analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\n✓ Report saved to: {report_path}")
    print("\nTo send email, run:")
    print("  python send_email_manual.py")
    print("\nOr to view the report:")
    print(f"  cat {report_path}")

if __name__ == "__main__":
    main()

