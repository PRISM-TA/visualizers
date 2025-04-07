import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to avoid Qt plugin issues
import matplotlib.pyplot as plt
import glob

def plot_multiple_stocks(stock_files, output_filepath):
    """
    Plot cumulative returns for multiple stocks in a single PNG file.

    Args:
        stock_files (list): List of file paths to the stock CSV files.
        output_filepath (str): Path to save the output PNG file.
    """
    # Sort stock files alphabetically by filename
    stock_files = sorted(stock_files, key=lambda x: x.split('/')[-1])

    # Set up the grid for 4 columns and 7 rows
    fig, axes = plt.subplots(7, 4, figsize=(20, 35))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    for i, stock_file in enumerate(stock_files):
        if i >= 28:  # Limit to 28 charts
            break

        # Load the CSV file
        data = pd.read_csv(stock_file)

        # Convert the 'date' column to datetime
        data['date'] = pd.to_datetime(data['date'])

        # Extract the year from the date
        data['year'] = data['date'].dt.year

        # Group by year and get the last cumulative return for each year
        yearly_benchmark_return = data.groupby('year')['benchmark_cum_return_pct'].last()
        yearly_target_return = data.groupby('year')['target_cum_return_pct'].last()

        # Plot both returns on the respective subplot
        ax = axes[i]
        ax.plot(yearly_benchmark_return.index, yearly_benchmark_return.values, 
                color='blue', marker='o', label='Benchmark')
        ax.plot(yearly_target_return.index, yearly_target_return.values, 
                color='orange', marker='o', label='Target')
        
        ax.set_title(stock_file.split('/')[-1].replace('.csv', ''))  # Use the filename as the title
        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_xticks(yearly_benchmark_return.index[::2])  # Show every other year to reduce overlap
        ax.grid(True)
        ax.legend()  # Add legend to show which line is which

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()

# Example usage:
# Get all stock CSV files (assuming they are in the same directory)
stock_files = glob.glob('*.csv')

# Plot and save the 28 charts
plot_multiple_stocks(stock_files, 'multiple_stocks_cumulative_return.png')