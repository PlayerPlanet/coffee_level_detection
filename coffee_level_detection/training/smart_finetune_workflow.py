"""
Integrated workflow: Smart annotation + Fine-tuning    # Fine-tuning parameters
    parser.add_argument('--pretrained', type=str, default='coffeeCNN.pth',
                       help='Path to pretrained model')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Fine-tuning epochs (conservative default)')
    parser.add_argument('--lr', type=float, default=1e-6,
                       help='Fine-tuning learning rate (very conservative)')
    parser.add_argument('--early-stopping', type=int, default=3,
                       help='Early stopping patience')es smart labeling with automatic model fine-tuning.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def count_annotations(manual_levels_dir='manual_levels'):
    """Count existing annotations."""
    if not os.path.exists(manual_levels_dir):
        return 0
    
    count = 0
    for filename in os.listdir(manual_levels_dir):
        if filename.startswith('level_') and filename.endswith('.json'):
            count += 1
    return count


def main():
    """Main workflow function."""
    parser = argparse.ArgumentParser(description='Smart Annotation + Fine-tuning Workflow')
    
    # Annotation phase
    parser.add_argument('--annotate-samples', type=int, default=20,
                       help='Number of samples to annotate')
    parser.add_argument('--skip-annotation', action='store_true',
                       help='Skip annotation phase')
    
    # Fine-tuning phase  
    parser.add_argument('--skip-finetuning', action='store_true',
                       help='Skip fine-tuning phase')
    parser.add_argument('--min-new-annotations', type=int, default=10,
                       help='Minimum new annotations required for fine-tuning')
    
    # Fine-tuning parameters
    parser.add_argument('--pretrained', type=str, default='coffee_level_detection\inference\coffeeCNN.pth',
                       help='Path to pretrained model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Fine-tuning learning rate')
    parser.add_argument('--early-stopping', default=2)
    args = parser.parse_args()
    
    print("🎯 Coffee Level Detection - Integrated Workflow")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 1: Smart Annotation
    if not args.skip_annotation:
        print("\n📋 PHASE 1: SMART ANNOTATION")
        print("=" * 40)
        
        initial_count = count_annotations()
        print(f"📊 Initial annotations: {initial_count}")
        
        # Run smart labeler
        annotation_cmd = [
            'poetry', 'run', 'python', '-m', 
            'coffee_level_detection.dataset_collection.smart_labeler',
            '--samples', str(args.annotate_samples)
        ]
        
        if not run_command(annotation_cmd, "Smart annotation"):
            print("❌ Annotation failed - stopping workflow")
            return 1
        
        final_count = count_annotations()
        new_annotations = final_count - initial_count
        print(f"📊 New annotations: {new_annotations}")
        print(f"📊 Total annotations: {final_count}")
        
    else:
        print("\n⏭️ Skipping annotation phase")
        new_annotations = count_annotations()
    
    # Phase 2: Fine-tuning
    if not args.skip_finetuning:
        print(f"\n🔧 PHASE 2: MODEL FINE-TUNING")
        print("=" * 40)
        
        total_annotations = count_annotations()
        print(f"📊 Available annotations: {total_annotations}")
        
        if total_annotations < args.min_new_annotations:
            print(f"⚠️ Insufficient annotations for fine-tuning")
            print(f"   Need at least {args.min_new_annotations}, have {total_annotations}")
            print("   Consider running more annotation rounds first")
            return 1
        
        # Check if pretrained model exists
        if not os.path.exists(args.pretrained):
            print(f"⚠️ Pretrained model {args.pretrained} not found!")
            print("   Fine-tuning will start from random weights")
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_model = f"coffeeCNN_finetuned_{timestamp}.pth"
        
        # Run fine-tuning
        finetune_cmd = [
            'poetry', 'run', 'python', '-m',
            'coffee_level_detection.training.finetune_coffeeCNN',
            '--pretrained', args.pretrained,
            '--output', output_model,
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--early-stopping', str(args.early_stopping)
        ]
        
        if not run_command(finetune_cmd, "Model fine-tuning"):
            print("❌ Fine-tuning failed")
            return 1
        
        print(f"\n🎉 Fine-tuned model saved as: {output_model}")
        
        # Optionally replace the original model
        print(f"\n💾 Model Management:")
        print(f"   Original: {args.pretrained}")
        print(f"   Fine-tuned: {output_model}")
        
        if args.pretrained == 'coffeeCNN.pth':
            response = input("Replace original model with fine-tuned version? (y/N): ")
            if response.lower() in ['y', 'yes']:
                import shutil
                backup_name = f"coffeeCNN_backup_{timestamp}.pth"
                shutil.copy2(args.pretrained, backup_name)
                shutil.copy2(output_model, args.pretrained)
                print(f"✅ Original backed up to: {backup_name}")
                print(f"✅ Fine-tuned model is now the main model")
    
    else:
        print("\n⏭️ Skipping fine-tuning phase")
    
    # Summary
    print(f"\n🎉 WORKFLOW COMPLETE")
    print("=" * 30)
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not args.skip_annotation:
        print(f"📋 Annotations added: {new_annotations if 'new_annotations' in locals() else 'N/A'}")
    
    if not args.skip_finetuning:
        print(f"🔧 Model fine-tuned: ✅")
        print(f"💾 Output model: {output_model if 'output_model' in locals() else 'N/A'}")
    
    print("\n📋 Next steps:")
    print("   - Test the fine-tuned model on validation data")
    print("   - Run inference to generate new predictions")
    print("   - Repeat the cycle for continuous improvement")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())