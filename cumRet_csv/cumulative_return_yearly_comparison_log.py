import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def plot_multiple_stocks(stock_files, output_filepath):
    """
    Plot cumulative returns with:
    - Linear scale from -200% to 0%
    - Log scale for positive values with adaptive logarithmic ticks
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
            ax.plot(df['date'], df['target_cum_return_pct'], 'orange', label='Target', linewidth=1)
            
            # Add a black solid line at y=0
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            # Set y-axis to log scale with smaller linthresh to compress the middle region
            ax.set_yscale('symlog', linthresh=90)
            
            # Set adaptive y-axis limits
            min_val = min(df['benchmark_cum_return_pct'].min(), df['target_cum_return_pct'].min())
            ax.set_ylim(bottom=min_val, top=None)  # Adapt bottom limit to the minimum value
            
            # Find maximum value for adaptive tick locations
            max_val = max(df['benchmark_cum_return_pct'].max(), df['target_cum_return_pct'].max())
            max_power = np.ceil(np.log10(max_val))
            
            # Create custom tick locations
            negative_ticks = [-100] if min_val < -100 else []
            small_ticks = [0]
            
            # Generate logarithmically spaced ticks for positive values up to the maximum
            final_log_ticks = []
            for power in range(2, int(max_power) + 1):
                base = 10**power
                final_log_ticks.extend([base / 10, base])
            final_log_ticks = [x for x in final_log_ticks if x <= max_val * 1.1 and x != 10]  # Add 10% margin, exclude 10%

            # Combine all ticks
            ticks = negative_ticks + small_ticks + final_log_ticks
            
            # Explicitly set only these ticks
            ax.yaxis.set_major_locator(plt.FixedLocator(ticks))
            ax.yaxis.set_minor_locator(plt.NullLocator())
            ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}%'))
            
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
            
            # Print ranges for verification
            print(f"\nFile: {title}")
            print(f"Benchmark range: {df['benchmark_cum_return_pct'].min():.2f} to {df['benchmark_cum_return_pct'].max():.2f}")
            print(f"Target range: {df['target_cum_return_pct'].min():.2f} to {df['target_cum_return_pct'].max():.2f}")
            
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
    plot_multiple_stocks(stock_files, 'multiple_stocks_cumulative_return_fixed.png')