import matplotlib.pyplot as plt
import numpy as np

def visualize_results(advisor_names,
                      final_selected_predictions, final_min_scores, final_selected_grids,
                      random_score, greedy_score, improved_score):
    # Visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Final Submission Grids Analysis', fontsize=16)

    # 1. Min score distribution
    axes[0, 0].hist(final_min_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 0].axvline(0.75, color='red', linestyle='--', label='Validity (0.75)')
    axes[0, 0].set_xlabel('Minimum Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Final Selection - Min Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 2. Advisor score distributions
    advisor_data = [final_selected_predictions[:, i] for i in range(4)]
    box_plot = axes[0, 1].boxplot(advisor_data, labels=advisor_names, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Advisor Score Distributions')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(alpha=0.3)

    # 3. Diversity comparison
    methods = ['Random', 'Greedy', 'Hill Climb']
    scores = [random_score, greedy_score, improved_score]
    colors = ['lightcoral', 'lightgreen', 'gold']
    bars = axes[0, 2].bar(methods, scores, color=colors, edgecolor='black')
    axes[0, 2].set_ylabel('Mean Pairwise Distance')
    axes[0, 2].set_title('Diversity Optimization Results')
    axes[0, 2].grid(alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.2f}', ha='center', va='bottom')

    # 4. Sample grids visualization - handle case where we have fewer than 4 grids
    n_samples = min(4, len(final_selected_grids))
    if n_samples > 0:
        sample_indices = np.random.choice(len(final_selected_grids), n_samples, replace=False)
        for idx, grid_idx in enumerate(sample_indices):
            if idx < 2:  # First row
                ax = axes[1, idx]
            else:  # Second row (if needed)
                if idx == 2:
                    ax = fig.add_subplot(2, 3, 5)
                else:
                    ax = fig.add_subplot(2, 3, 6)
            
            grid = final_selected_grids[grid_idx]
            im = ax.imshow(grid, cmap='tab10', vmin=0, vmax=4)
            ax.set_title(f'Sample Grid {grid_idx+1}\nMin Score: {final_min_scores[grid_idx]:.3f}')
            ax.set_xticks([])
            ax.set_yticks([])

        # Remove unused subplots
        if n_samples < 4:
            if n_samples <= 2:
                fig.delaxes(axes[1, 2])
            if n_samples == 1:
                fig.delaxes(axes[1, 1])

        # Add colorbar for grids
        cbar = plt.colorbar(im, ax=axes[1, :n_samples], shrink=0.6, location='right')
        cbar.set_ticks([0, 1, 2, 3, 4])
        cbar.set_ticklabels(['Res', 'Ind', 'Com', 'Parks', 'Office'])
    else:
        # If no grids selected, remove all bottom plots
        for i in range(3):
            if i < len(axes[1]):
                fig.delaxes(axes[1, i])

    plt.tight_layout()
    plt.show()