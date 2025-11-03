"""
Analysis and visualization utilities for data processing pipeline
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class AnalysisPlots:
    """Handle analysis plots for data processing pipeline"""
    
    @staticmethod
    def plot_data_filtering_analysis(min_scores, source_labels, valid_grids_by_threshold, 
                                   optimal_threshold, candidate_sources, candidate_predictions):
        """
        Create comprehensive visualization of data filtering results
        
        Args:
            min_scores: Array of minimum scores across advisors
            source_labels: Labels indicating data source ('original', 'generated')
            valid_grids_by_threshold: Dict with threshold -> grid info
            optimal_threshold: Selected optimal safety threshold
            candidate_sources: Final candidate source labels
            candidate_predictions: Final candidate predictions array
        """
        
        plt.figure(figsize=(15, 8))
        
        # Min score distribution by source
        plt.subplot(2, 3, 1)
        source_counts = Counter(source_labels)
        
        if 'original' in source_counts:
            original_mask = source_labels == 'original'
            plt.hist(min_scores[original_mask], bins=30, alpha=0.6, 
                    label='Original', color='blue', density=True)
        
        if 'generated' in source_counts:
            generated_mask = source_labels == 'generated'
            plt.hist(min_scores[generated_mask], bins=30, alpha=0.6, 
                    label='Generated', color='orange', density=True)
        
        # Add threshold lines
        safety_thresholds = [0.75, 0.80, 0.85, 0.90]
        for threshold in safety_thresholds:
            color = 'red' if threshold == optimal_threshold else 'gray'
            alpha = 0.8 if threshold == optimal_threshold else 0.3
            plt.axvline(threshold, color=color, linestyle='--', alpha=alpha)
        
        plt.xlabel('Minimum Score')
        plt.ylabel('Density')
        plt.title('Min Score Distribution by Source')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Candidate counts by threshold
        plt.subplot(2, 3, 2)
        thresholds = list(valid_grids_by_threshold.keys())
        counts = [valid_grids_by_threshold[t]['count'] for t in thresholds]
        colors = ['red' if t == optimal_threshold else 'lightblue' for t in thresholds]
        bars = plt.bar(thresholds, counts, color=colors, edgecolor='black')
        plt.xlabel('Safety Threshold')
        plt.ylabel('Valid Grids Count')
        plt.title('Valid Grids by Threshold')
        plt.grid(alpha=0.3)
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{count:,}', ha='center', va='bottom', 
                     rotation=90 if count > 1000 else 0)
        
        # Source composition at optimal threshold
        plt.subplot(2, 3, 3)
        final_source_counts = Counter(candidate_sources)
        if len(final_source_counts) > 0:
            sources = list(final_source_counts.keys())
            counts = list(final_source_counts.values())
            colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'][:len(sources)]
            
            plt.pie(counts, labels=sources, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            plt.title(f'Final Candidate Composition\n(Threshold {optimal_threshold:.2f})')
        
        # Score distributions by advisor
        advisor_names = ['Wellness', 'Tax', 'Transportation', 'Business']
        for i, advisor in enumerate(advisor_names[:3]):
            plt.subplot(2, 3, 4 + i)
            
            if 'original' in Counter(candidate_sources):
                original_mask = candidate_sources == 'original'
                if np.any(original_mask):
                    plt.hist(candidate_predictions[original_mask, i], bins=20, alpha=0.6, 
                            label='Original', color='blue', density=True)
            
            if 'generated' in Counter(candidate_sources):
                generated_mask = candidate_sources == 'generated'
                if np.any(generated_mask):
                    plt.hist(candidate_predictions[generated_mask, i], bins=20, alpha=0.6, 
                            label='Generated', color='orange', density=True)
            
            plt.xlabel(f'{advisor} Score')
            plt.ylabel('Density')
            plt.title(f'{advisor} Score Distribution')
            if i == 0:
                plt.legend()
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_augmentation_analysis(candidate_grids, candidate_predictions, candidate_count,
                                 augmented_grids, unique_grids, unique_predictions):
        """
        Create visualization comparing before/after augmentation results
        
        Args:
            candidate_grids: Original candidate grids
            candidate_predictions: Original candidate predictions
            candidate_count: Number of original candidates
            augmented_grids: Grids after augmentation
            unique_grids: Grids after deduplication
            unique_predictions: Predictions after deduplication
        """
        
        plt.figure(figsize=(15, 5))
        
        # Score distribution comparison
        plt.subplot(1, 3, 1)
        candidate_min_scores = np.min(candidate_predictions, axis=1)
        final_min_scores = np.min(unique_predictions, axis=1)
        
        plt.hist(candidate_min_scores, bins=30, alpha=0.6, 
                label='Original Candidates', color='blue', density=True)
        plt.hist(final_min_scores, bins=30, alpha=0.6, 
                label='After Augmentation', color='orange', density=True)
        plt.xlabel('Minimum Score')
        plt.ylabel('Density')
        plt.title('Score Distribution: Before vs After Augmentation')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Pool size comparison
        plt.subplot(1, 3, 2)
        categories = ['Original\nCandidates', 'After\nAugmentation', 'After\nDeduplication']
        counts = [candidate_count, len(augmented_grids), len(unique_grids)]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        bars = plt.bar(categories, counts, color=colors, edgecolor='black')
        plt.ylabel('Grid Count')
        plt.title('Augmentation Pool Size')
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{count:,}', ha='center', va='bottom')
        
        # Diversity preview (sample variance in flattened grids)
        plt.subplot(1, 3, 3)
        original_flat = candidate_grids.reshape(len(candidate_grids), -1)
        augmented_flat = unique_grids.reshape(len(unique_grids), -1)
        
        original_diversity = np.mean(np.var(original_flat, axis=0))
        augmented_diversity = np.mean(np.var(augmented_flat, axis=0))
        
        diversity_types = ['Original', 'Augmented']
        diversity_values = [original_diversity, augmented_diversity]
        colors = ['lightblue', 'lightgreen']
        
        bars = plt.bar(diversity_types, diversity_values, color=colors, edgecolor='black')
        plt.ylabel('Mean Position Variance')
        plt.title('Grid Diversity Preview')
        
        for bar, value in zip(bars, diversity_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_pipeline_summary(data_stats):
        """
        Create summary visualization of entire pipeline
        
        Args:
            data_stats: Dict containing pipeline statistics
        """
        
        plt.figure(figsize=(12, 8))
        
        # Pipeline flow chart
        plt.subplot(2, 2, 1)
        stages = ['Original\nData', 'Generated\nGrids', 'Safety\nFiltered', 'Augmented\n& Unique']
        counts = [
            data_stats.get('original_count', 0),
            data_stats.get('generated_count', 0), 
            data_stats.get('filtered_count', 0),
            data_stats.get('final_count', 0)
        ]
        
        plt.bar(stages, counts, color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
        plt.ylabel('Grid Count')
        plt.title('Pipeline Processing Flow')
        plt.xticks(rotation=45)
        
        for i, count in enumerate(counts):
            if count > 0:
                plt.text(i, count, f'{count:,}', ha='center', va='bottom')
        
        # Quality progression
        plt.subplot(2, 2, 2)
        quality_stages = ['Input', 'Filtered', 'Final']
        min_scores = [
            data_stats.get('input_min_score', 0),
            data_stats.get('filtered_min_score', 0),
            data_stats.get('final_min_score', 0)
        ]
        mean_scores = [
            data_stats.get('input_mean_score', 0),
            data_stats.get('filtered_mean_score', 0),
            data_stats.get('final_mean_score', 0)
        ]
        
        x = np.arange(len(quality_stages))
        width = 0.35
        
        plt.bar(x - width/2, min_scores, width, label='Min Score', alpha=0.8)
        plt.bar(x + width/2, mean_scores, width, label='Mean Score', alpha=0.8)
        
        plt.xlabel('Pipeline Stage')
        plt.ylabel('Score')
        plt.title('Quality Progression')
        plt.xticks(x, quality_stages)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Processing efficiency
        plt.subplot(2, 2, 3)
        efficiency_metrics = ['Retention\nRate', 'Quality\nGain', 'Diversity\nGain']
        values = [
            data_stats.get('retention_rate', 0) * 100,
            data_stats.get('quality_improvement', 0) * 100,
            data_stats.get('diversity_improvement', 0) * 100
        ]
        colors = ['lightgreen' if v > 0 else 'lightcoral' for v in values]
        
        bars = plt.bar(efficiency_metrics, values, color=colors)
        plt.ylabel('Percentage')
        plt.title('Processing Efficiency')
        plt.xticks(rotation=45)
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{value:.1f}%', ha='center', va='bottom')
        
        # Source composition
        plt.subplot(2, 2, 4)
        source_names = list(data_stats.get('source_composition', {}).keys())
        source_counts = list(data_stats.get('source_composition', {}).values())
        
        if source_names:
            plt.pie(source_counts, labels=source_names, autopct='%1.1f%%', startangle=90)
            plt.title('Final Data Source Composition')
        else:
            plt.text(0.5, 0.5, 'No source data available', ha='center', va='center')
            plt.title('Source Composition')
        
        plt.tight_layout()
        plt.show()


def create_pipeline_summary_stats(original_grids=None, generated_grids=None, 
                                filtered_grids=None, final_grids=None,
                                original_preds=None, filtered_preds=None, final_preds=None,
                                source_labels=None):
    """
    Helper function to create statistics dictionary for pipeline summary
    
    Returns:
        dict: Statistics for use with plot_pipeline_summary
    """
    
    stats = {}
    
    # Counts
    stats['original_count'] = len(original_grids) if original_grids is not None else 0
    stats['generated_count'] = len(generated_grids) if generated_grids is not None else 0
    stats['filtered_count'] = len(filtered_grids) if filtered_grids is not None else 0
    stats['final_count'] = len(final_grids) if final_grids is not None else 0
    
    # Quality scores
    if original_preds is not None:
        original_min_scores = np.min(original_preds, axis=1)
        stats['input_min_score'] = np.min(original_min_scores)
        stats['input_mean_score'] = np.mean(original_min_scores)
    
    if filtered_preds is not None:
        filtered_min_scores = np.min(filtered_preds, axis=1)
        stats['filtered_min_score'] = np.min(filtered_min_scores)
        stats['filtered_mean_score'] = np.mean(filtered_min_scores)
    
    if final_preds is not None:
        final_min_scores = np.min(final_preds, axis=1)
        stats['final_min_score'] = np.min(final_min_scores)
        stats['final_mean_score'] = np.mean(final_min_scores)
    
    # Efficiency metrics
    if stats['original_count'] > 0:
        stats['retention_rate'] = stats['final_count'] / stats['original_count']
    
    if 'input_mean_score' in stats and 'final_mean_score' in stats:
        if stats['input_mean_score'] > 0:
            stats['quality_improvement'] = (stats['final_mean_score'] - stats['input_mean_score']) / stats['input_mean_score']
    
    # Diversity improvement would need to be calculated separately with actual diversity metrics
    stats['diversity_improvement'] = 0.0  # Placeholder
    
    # Source composition
    if source_labels is not None:
        stats['source_composition'] = dict(Counter(source_labels))
    
    return stats