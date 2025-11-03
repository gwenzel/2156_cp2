"""
Submission utilities for final selection and file generation
"""
import numpy as np
import os
from datetime import datetime
import json


class SubmissionBuilder:
    """Handle final submission creation and file management"""
    
    def __init__(self, output_dir='data/submissions'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_final_selection(self, selected_grids, selected_predictions, 
                              div_score, method_name="diversity_optimized"):
        """Prepare final selection data for submission"""
        
        selected_min_scores = np.min(selected_predictions, axis=1)
        
        # Calculate performance metrics
        metrics = {
            'count': len(selected_grids),
            'min_score': float(np.min(selected_min_scores)),
            'max_score': float(np.max(selected_min_scores)),
            'mean_score': float(np.mean(selected_min_scores)),
            'std_score': float(np.std(selected_min_scores)),
            'diversity_score': float(div_score),
            'method': method_name,
            'valid_grids': int(np.sum(selected_min_scores >= 0.75)),
            'high_quality': int(np.sum(selected_min_scores >= 0.80))
        }
        
        # Per-advisor statistics
        advisor_names = ['Wellness', 'Tax', 'Transportation', 'Business']
        advisor_stats = {}
        
        for i, advisor in enumerate(advisor_names):
            advisor_scores = selected_predictions[:, i]
            advisor_stats[advisor] = {
                'min': float(np.min(advisor_scores)),
                'max': float(np.max(advisor_scores)),
                'mean': float(np.mean(advisor_scores)),
                'std': float(np.std(advisor_scores))
            }
        
        metrics['advisor_stats'] = advisor_stats
        
        return metrics
    
    def save_submission(self, selected_grids, selected_predictions, metrics, 
                       include_analysis=True):
        """Save final submission with all supporting files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"final_submission_{timestamp}"
        
        # Main submission file (grids only as required)
        submission_path = os.path.join(self.output_dir, f"{base_filename}.npy")
        np.save(submission_path, selected_grids)
        
        print(f"💾 MAIN SUBMISSION:")
        print(f"   Grids: {submission_path}")
        print(f"   Count: {len(selected_grids):,}")
        
        saved_files = [submission_path]
        
        if include_analysis:
            # Predictions file
            predictions_path = os.path.join(self.output_dir, f"{base_filename}_predictions.npy")
            np.save(predictions_path, selected_predictions)
            saved_files.append(predictions_path)
            
            # Metrics file
            metrics_path = os.path.join(self.output_dir, f"{base_filename}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            saved_files.append(metrics_path)
            
            # Summary report
            summary_path = os.path.join(self.output_dir, f"{base_filename}_summary.txt")
            self._create_summary_report(metrics, summary_path)
            saved_files.append(summary_path)
            
            print(f"\n💾 SUPPORTING FILES:")
            print(f"   Predictions: {predictions_path}")
            print(f"   Metrics: {metrics_path}")
            print(f"   Summary: {summary_path}")
        
        return submission_path, saved_files
    
    def _create_summary_report(self, metrics, summary_path):
        """Create human-readable summary report"""
        
        lines = [
            f"FINAL SUBMISSION SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "📊 SELECTION PERFORMANCE:",
            f"  • Total grids: {metrics['count']:,}",
            f"  • Method: {metrics['method']}",
            f"  • Diversity score: {metrics['diversity_score']:.6f}",
            "",
            "🎯 SCORE STATISTICS:",
            f"  • Min score: {metrics['min_score']:.4f}",
            f"  • Max score: {metrics['max_score']:.4f}",
            f"  • Mean score: {metrics['mean_score']:.4f}",
            f"  • Std deviation: {metrics['std_score']:.4f}",
            "",
            "✅ QUALITY METRICS:",
            f"  • Valid grids (≥0.75): {metrics['valid_grids']:,} ({metrics['valid_grids']/metrics['count']*100:.1f}%)",
            f"  • High-quality (≥0.80): {metrics['high_quality']:,} ({metrics['high_quality']/metrics['count']*100:.1f}%)",
            "",
            "🏆 ADVISOR PERFORMANCE:",
        ]
        
        for advisor, stats in metrics['advisor_stats'].items():
            lines.extend([
                f"  {advisor}:",
                f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]",
                f"    Mean: {stats['mean']:.3f} (±{stats['std']:.3f})",
            ])
        
        lines.extend([
            "",
            "🔧 TECHNICAL DETAILS:",
            f"  • Grid dimensions: 7×7",
            f"  • Total positions: 49 per grid",
            f"  • District types: 5 (Residential, Industrial, Commercial, Parks, Office)",
            f"  • Optimization: Diversity maximization with score constraints",
            "",
            "📁 FILES GENERATED:",
            f"  • Main submission: final_submission_*.npy",
            f"  • Predictions: final_submission_*_predictions.npy", 
            f"  • Metrics: final_submission_*_metrics.json",
            f"  • Summary: final_submission_*_summary.txt",
        ])
        
        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def validate_submission(self, selected_grids, selected_predictions, min_threshold=0.78):
        """Validate submission meets requirements"""
        
        issues = []
        
        # Check grid format
        if selected_grids.ndim != 3 or selected_grids.shape[1:] != (7, 7):
            issues.append(f"Invalid grid shape: {selected_grids.shape} (expected: (N, 7, 7))")
        else:
            print("Shape ok")
        
        # Check district values
        if not np.all((selected_grids >= 0) & (selected_grids <= 4)):
            issues.append("Grid values outside valid range [0, 4]")
        else:
            print("District values ok")
        
        # Check predictions format
        if selected_predictions.ndim != 2 or selected_predictions.shape[1] != 4:
            issues.append(f"Invalid predictions shape: {selected_predictions.shape} (expected: (N, 4))")
        else:
            print("Predictions format ok")
        
        # Check score threshold
        min_scores = np.min(selected_predictions, axis=1)
        valid_count = np.sum(min_scores >= min_threshold)
        if valid_count < len(selected_grids) * 0.95:  # At least 95% should be valid
            issues.append(f"Too many invalid grids: {len(selected_grids) - valid_count}/{len(selected_grids)} below threshold {min_threshold}")
        else:
            print("Scores ok")
        
        # Check for exact duplicates
        flat_grids = selected_grids.reshape(len(selected_grids), -1)
        unique_grids = np.unique(flat_grids, axis=0)
        if len(unique_grids) < len(selected_grids):
            duplicates = len(selected_grids) - len(unique_grids)
            issues.append(f"Found {duplicates} duplicate grids")
        else:
            print("Duplicates ok")
        
        if issues:
            print("❌ VALIDATION ISSUES FOUND:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ SUBMISSION VALIDATION PASSED")
            return True
    
    def compare_methods(self, method_results):
        """Compare results from different optimization methods"""
        
        print(f"\n📊 METHOD COMPARISON:")
        print(f"{'Method':<20} {'Count':<8} {'Diversity':<12} {'Mean Score':<12} {'Valid %':<10}")
        print("-" * 70)
        
        for method_name, result in method_results.items():
            metrics = result['metrics']
            diversity = metrics['diversity_score']
            mean_score = metrics['mean_score']
            valid_pct = metrics['valid_grids'] / metrics['count'] * 100
            
            print(f"{method_name:<20} {metrics['count']:<8,} {diversity:<12.6f} {mean_score:<12.4f} {valid_pct:<10.1f}%")
        
        # Recommend best method
        best_method = max(method_results.keys(), 
                         key=lambda k: method_results[k]['metrics']['diversity_score'])
        
        print(f"\n🏆 RECOMMENDED METHOD: {best_method}")
        print(f"   Highest diversity score: {method_results[best_method]['metrics']['diversity_score']:.6f}")
        
        return best_method