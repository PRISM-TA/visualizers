import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os

def plot_multiple_stocks(stock_files, output_filepath):
    """
    Plot cumulative returns in the most basic way without log scaling.
    """
    stock_files = sorted(stock_files)
    fig, axes = plt.subplots(7, 4, figsize=(20, 35))
    axes = axes.flatten()

    for i, stock_file in enumerate(stock_files):
        if i >= 28:
            break
            
        try:
            df = pd.read_csv(stock_file)
            df['date'] = pd.to_datetime(df['date'])
            
            ax = axes[i]
            
            # Plot data
            ax.plot(df['date'], df['benchmark_cum_return_pct'], 'b-', label='Benchmark', linewidth=1)
            ax.plot(df['date'], df['target_cum_return_pct'], 'green', label='Target', linewidth=1)
            
            # Add a black solid line at y=0
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            # Set y-axis limits
            min_val = min(df['benchmark_cum_return_pct'].min(), df['target_cum_return_pct'].min())
            max_val = max(df['benchmark_cum_return_pct'].max(), df['target_cum_return_pct'].max())
            ax.set_ylim(bottom=min_val, top=max_val)
            
            # Title and labels
            title = os.path.splitext(os.path.basename(stock_file))[0]
            ax.set_title(title)
            ax.set_xlabel('Year')
            ax.set_ylabel('Cumulative Return (%)')
            
            # Format x-axis
            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        except Exception as e:
            print(f"Error processing {stock_file}: {str(e)}")
            continue

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()

# Example usage:
stock_files = glob.glob('*.csv')

if len(stock_files) == 0:
    print("No CSV files found in the specified directory!")
else:
    print(f"Found {len(stock_files)} CSV files")
    plot_multiple_stocks(stock_files, 'multiple_stocks_cumulative_return_actual_result.png')