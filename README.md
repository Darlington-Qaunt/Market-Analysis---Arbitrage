# Market-Analysis---Arbitrage

Before searching for triangular arbitrage opportunities, it was important to first study the overall market behavior of the assets involved in the trading triangle. In this case, a time-series analysis of Solana (SOL) and Ethereum (ETH) was carried out on August 3, 2025, using 1-hour interval data from Binance’s historical records. This step helped to understand key factors such as volatility, price trends, and trading volumes, all of which directly affect how often arbitrage opportunities appear and how profitable they can be.
By looking at the price movements of SOL/USDT and ETH/USDT throughout the day, I was  able to spot periods of high volatility or unusual trading activity that might create price gaps across the related pairs. The analysis included candlestick charts, trading volume spikes, and hourly returns, which gave a more detailed view of market conditions. Statistical measures such as the correlation between SOL and ETH prices were also calculated. Correlation is especially important in triangular arbitrage: when assets move very closely together, arbitrage windows are often smaller; when they move more independently, opportunities are more likely to appear.
To make this clearer, a range of visualizations—such as price overview charts, volume distributions, and return patterns—were also produced and studied.

import time
import requests
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# API endpoint
BASE_URL = "https://api.binance.com"

# Target date: August 3, 2025
TARGET_DATE = datetime(2025, 8, 3, 0, 0, 0)
END_DATE = TARGET_DATE + timedelta(days=1)  # Full 24 hours

# Pairs to fetch
PAIRS = ['SOLUSDT', 'ETHUSDT']

# Output files
PRICE_CSV = "price_timeseries_aug3.csv"
PLOTS_DIR = "price_plots"


def get_historical_prices(symbol, start_time, end_time, interval='1h'):
    """Fetch historical klines/candlestick data for a trading pair."""
    try:
        url = f"{BASE_URL}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": 1000
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract all candle data
        prices = []
        for candle in data:
            prices.append({
                'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            })

        return prices
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None


def fetch_and_save_prices():
    """Fetch price data and save to CSV."""
    print(f"Fetching data for {TARGET_DATE.date()}...")

    all_data = []

    for pair in PAIRS:
        print(f"\nFetching {pair}...")
        prices = get_historical_prices(pair, TARGET_DATE, END_DATE, interval='1h')

        if prices is None:
            print(f"Failed to fetch data for {pair}")
            continue

        for price_data in prices:
            all_data.append({
                'Timestamp': price_data['timestamp'],
                'Pair': pair,
                'Open': price_data['open'],
                'High': price_data['high'],
                'Low': price_data['low'],
                'Close': price_data['close'],
                'Volume': price_data['volume']
            })

        print(f"   Fetched {len(prices)} data points for {pair}")
        time.sleep(0.5)  # Respect API rate limits

    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    df = df.sort_values(['Pair', 'Timestamp'])
    df.to_csv(PRICE_CSV, index=False)

    print(f"\n✅ Data saved to {PRICE_CSV}")
    return df


def create_price_plots(df):
    """Create comprehensive price analysis plots."""

    # Create plots directory
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # Pivot data for easier plotting
    sol_df = df[df['Pair'] == 'SOLUSDT'].copy()
    eth_df = df[df['Pair'] == 'ETHUSDT'].copy()

    print(f"\nCreating plots...")
    print(f"SOL data points: {len(sol_df)}")
    print(f"ETH data points: {len(eth_df)}")

    
    # 1. PRICE OVERVIEW - Both pairs
  
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # SOL Plot
    axes[0].plot(sol_df['Timestamp'], sol_df['Close'], 'b-', linewidth=2, label='Close')
    axes[0].fill_between(sol_df['Timestamp'], sol_df['Low'], sol_df['High'],
                         alpha=0.3, color='blue', label='High-Low Range')
    axes[0].set_ylabel('SOL/USDT Price', fontsize=12, fontweight='bold')
    axes[0].set_title('SOL Price - August 3, 2025 (1-Hour Intervals)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Add statistics
    sol_change = ((sol_df['Close'].iloc[-1] / sol_df['Close'].iloc[0]) - 1) * 100
    sol_stats = f"Open: ${sol_df['Open'].iloc[0]:.2f}\n"
    sol_stats += f"Close: ${sol_df['Close'].iloc[-1]:.2f}\n"
    sol_stats += f"Change: {sol_change:+.2f}%\n"
    sol_stats += f"High: ${sol_df['High'].max():.2f}\n"
    sol_stats += f"Low: ${sol_df['Low'].min():.2f}"
    axes[0].text(0.02, 0.98, sol_stats, transform=axes[0].transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # ETH Plot
    axes[1].plot(eth_df['Timestamp'], eth_df['Close'], 'g-', linewidth=2, label='Close')
    axes[1].fill_between(eth_df['Timestamp'], eth_df['Low'], eth_df['High'],
                         alpha=0.3, color='green', label='High-Low Range')
    axes[1].set_xlabel('Time (UTC)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('ETH/USDT Price', fontsize=12, fontweight='bold')
    axes[1].set_title('ETH Price - August 3, 2025 (1-Hour Intervals)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Add statistics
    eth_change = ((eth_df['Close'].iloc[-1] / eth_df['Close'].iloc[0]) - 1) * 100
    eth_stats = f"Open: ${eth_df['Open'].iloc[0]:.2f}\n"
    eth_stats += f"Close: ${eth_df['Close'].iloc[-1]:.2f}\n"
    eth_stats += f"Change: {eth_change:+.2f}%\n"
    eth_stats += f"High: ${eth_df['High'].max():.2f}\n"
    eth_stats += f"Low: ${eth_df['Low'].min():.2f}"
    axes[1].text(0.02, 0.98, eth_stats, transform=axes[1].transAxes,
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/1_price_overview.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {PLOTS_DIR}/1_price_overview.png")

    
    # 2. CANDLESTICK CHART
  
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # SOL Candlestick
    for idx, row in sol_df.iterrows():
        color = 'green' if row['Close'] >= row['Open'] else 'red'
        # Draw high-low line
        axes[0].plot([row['Timestamp'], row['Timestamp']], [row['Low'], row['High']],
                     color=color, linewidth=1)
        # Draw open-close box
        height = abs(row['Close'] - row['Open'])
        bottom = min(row['Open'], row['Close'])
        axes[0].bar(row['Timestamp'], height, bottom=bottom, width=0.02,
                    color=color, alpha=0.8, edgecolor=color)

    axes[0].set_ylabel('SOL/USDT Price', fontsize=12, fontweight='bold')
    axes[0].set_title('SOL Candlestick Chart - August 3, 2025', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # ETH Candlestick
    for idx, row in eth_df.iterrows():
        color = 'green' if row['Close'] >= row['Open'] else 'red'
        axes[1].plot([row['Timestamp'], row['Timestamp']], [row['Low'], row['High']],
                     color=color, linewidth=1)
        height = abs(row['Close'] - row['Open'])
        bottom = min(row['Open'], row['Close'])
        axes[1].bar(row['Timestamp'], height, bottom=bottom, width=0.02,
                    color=color, alpha=0.8, edgecolor=color)

    axes[1].set_xlabel('Time (UTC)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('ETH/USDT Price', fontsize=12, fontweight='bold')
    axes[1].set_title('ETH Candlestick Chart - August 3, 2025', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/2_candlestick.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {PLOTS_DIR}/2_candlestick.png")

   
    # 3. VOLUME ANALYSIS
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # SOL Volume
    axes[0].bar(sol_df['Timestamp'], sol_df['Volume'], width=0.03,
                color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Volume (SOL)', fontsize=12, fontweight='bold')
    axes[0].set_title('SOL Trading Volume - August 3, 2025', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    avg_vol = sol_df['Volume'].mean()
    axes[0].axhline(y=avg_vol, color='red', linestyle='--', linewidth=2,
                    label=f'Avg Volume: {avg_vol:,.0f}')
    axes[0].legend()

    # ETH Volume
    axes[1].bar(eth_df['Timestamp'], eth_df['Volume'], width=0.03,
                color='seagreen', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Time (UTC)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Volume (ETH)', fontsize=12, fontweight='bold')
    axes[1].set_title('ETH Trading Volume - August 3, 2025', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    avg_vol = eth_df['Volume'].mean()
    axes[1].axhline(y=avg_vol, color='red', linestyle='--', linewidth=2,
                    label=f'Avg Volume: {avg_vol:,.0f}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/3_volume_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {PLOTS_DIR}/3_volume_analysis.png")

   
    # 4. RETURNS ANALYSIS
    
    sol_df['Returns'] = sol_df['Close'].pct_change() * 100
    eth_df['Returns'] = eth_df['Close'].pct_change() * 100

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # SOL Returns over time
    axes[0, 0].plot(sol_df['Timestamp'], sol_df['Returns'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].fill_between(sol_df['Timestamp'], 0, sol_df['Returns'],
                            where=(sol_df['Returns'] > 0), alpha=0.3, color='green')
    axes[0, 0].fill_between(sol_df['Timestamp'], 0, sol_df['Returns'],
                            where=(sol_df['Returns'] <= 0), alpha=0.3, color='red')
    axes[0, 0].set_title('SOL Hourly Returns', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Returns (%)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # SOL Returns distribution
    axes[0, 1].hist(sol_df['Returns'].dropna(), bins=15, color='steelblue',
                    alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('SOL Returns Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Returns (%)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # ETH Returns over time
    axes[1, 0].plot(eth_df['Timestamp'], eth_df['Returns'], 'g-', linewidth=2)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].fill_between(eth_df['Timestamp'], 0, eth_df['Returns'],
                            where=(eth_df['Returns'] > 0), alpha=0.3, color='green')
    axes[1, 0].fill_between(eth_df['Timestamp'], 0, eth_df['Returns'],
                            where=(eth_df['Returns'] <= 0), alpha=0.3, color='red')
    axes[1, 0].set_title('ETH Hourly Returns', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (UTC)', fontsize=11)
    axes[1, 0].set_ylabel('Returns (%)', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # ETH Returns distribution
    axes[1, 1].hist(eth_df['Returns'].dropna(), bins=15, color='seagreen',
                    alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('ETH Returns Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Returns (%)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/4_returns_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {PLOTS_DIR}/4_returns_analysis.png")

   
    # 5. CORRELATION ANALYSIS

    # Merge on timestamp for correlation
    merged = sol_df[['Timestamp', 'Close', 'Returns']].merge(
        eth_df[['Timestamp', 'Close', 'Returns']],
        on='Timestamp', suffixes=('_SOL', '_ETH')
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Price correlation
    axes[0].scatter(merged['Close_SOL'], merged['Close_ETH'],
                    alpha=0.6, s=100, c=range(len(merged)), cmap='viridis')
    axes[0].set_xlabel('SOL/USDT Price', fontsize=12)
    axes[0].set_ylabel('ETH/USDT Price', fontsize=12)
    axes[0].set_title('SOL vs ETH Price Correlation', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Add correlation coefficient
    corr_price = merged['Close_SOL'].corr(merged['Close_ETH'])
    axes[0].text(0.05, 0.95, f'Correlation: {corr_price:.4f}',
                 transform=axes[0].transAxes, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Returns correlation
    axes[1].scatter(merged['Returns_SOL'], merged['Returns_ETH'],
                    alpha=0.6, s=100, c=range(len(merged)), cmap='viridis')
    axes[1].set_xlabel('SOL Returns (%)', fontsize=12)
    axes[1].set_ylabel('ETH Returns (%)', fontsize=12)
    axes[1].set_title('SOL vs ETH Returns Correlation', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Add correlation coefficient
    corr_returns = merged['Returns_SOL'].corr(merged['Returns_ETH'])
    axes[1].text(0.05, 0.95, f'Correlation: {corr_returns:.4f}',
                 transform=axes[1].transAxes, fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/5_correlation_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {PLOTS_DIR}/5_correlation_analysis.png")

    print("\n✅ All plots created successfully!")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS - AUGUST 3, 2025")
    print("=" * 70)
    print("\nSOLANA (SOL/USDT):")
    print(f"  Open: ${sol_df['Open'].iloc[0]:.2f}")
    print(f"  Close: ${sol_df['Close'].iloc[-1]:.2f}")
    print(f"  Change: {sol_change:+.2f}%")
    print(f"  High: ${sol_df['High'].max():.2f}")
    print(f"  Low: ${sol_df['Low'].min():.2f}")
    print(f"  Avg Volume: {sol_df['Volume'].mean():,.0f} SOL")

    print("\nETHEREUM (ETH/USDT):")
    print(f"  Open: ${eth_df['Open'].iloc[0]:.2f}")
    print(f"  Close: ${eth_df['Close'].iloc[-1]:.2f}")
    print(f"  Change: {eth_change:+.2f}%")
    print(f"  High: ${eth_df['High'].max():.2f}")
    print(f"  Low: ${eth_df['Low'].min():.2f}")
    print(f"  Avg Volume: {eth_df['Volume'].mean():,.0f} ETH")

    print("\nCORRELATION:")
    print(f"  Price Correlation: {corr_price:.4f}")
    print(f"  Returns Correlation: {corr_returns:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    # Fetch data and save to CSV
    df = fetch_and_save_prices()

    # Create plots
    create_price_plots(df)

    print("\n🎉 Analysis complete!")
    print(f"📊 Data saved to: {PRICE_CSV}")
    print(f"📈 Plots saved to: {PLOTS_DIR}/")
