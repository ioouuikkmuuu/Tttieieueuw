import pandas as pd
import numpy as np
import logging
from datetime import datetime

def setup_logger():
    """
    Configure and return a logger for backtesting
    
    Creates a logger with both console and file handlers:
    - Console: INFO level for real-time monitoring
    - File: DEBUG level for detailed analysis
    """
    logger = logging.getLogger('backtest')
    logger.setLevel(logging.DEBUG)
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler for detailed debugging
    file_handler = logging.FileHandler('backtest.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Unified log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

class SectorBacktester:
    """
    Backtesting engine for sector rotation strategies with long/short support
    
    Core functionality:
    - Monthly rebalancing to new sectors
    - Support for both long and short positions
    - Daily NAV tracking
    - Detailed holdings management
    
    Parameters:
    :param df_returns: DataFrame with daily returns (index=dates, columns=sectors)
    :param initial_capital: Starting capital (default: 1,000,000)
    :param logger: Optional logger instance
    """
    
    def __init__(self, df_returns, initial_capital=1e6, logger=None):
        # Clean input data: replace missing returns with 0
        self.df_returns = df_returns.fillna(0)
        self.initial_capital = initial_capital
        self.dates = df_returns.index
        self.sectors = df_returns.columns
        self.logger = logger or logging.getLogger('backtest')
        
        # Initialize tracking structures
        self.rebalance_dates = []  # Dates when rebalancing occurred
        self.long_positions = []    # Currently held long positions
        self.short_positions = []   # Currently held short positions
        self.nav_record = None      # Daily NAV time series
        self.holdings = None         # Daily holdings matrix (positive for long, negative for short)

    def _initialize_records(self):
        """Initialize data structures for tracking backtest results"""
        self.logger.info("Initializing data records...")
        
        # NAV time series (aligned with trading dates)
        self.nav_record = pd.Series(index=self.dates, dtype=float)
        self.nav_record.iloc[0] = self.initial_capital
        
        # Holdings matrix (sectors × dates)
        # Positive values = long positions, negative values = short positions
        self.holdings = pd.DataFrame(
            0, 
            index=self.dates,
            columns=self.sectors,
            dtype=float
        )
        
        self.logger.info(f"Initial capital: ¥{self.initial_capital:,.2f}")

    def _log_holdings(self, date, title="Current Holdings"):
        """Log detailed holdings information for a specific date"""
        long_details = []
        short_details = []
        total_value = 0
        
        # Process long positions
        for sector in self.long_positions:
            value = self.holdings.loc[date, sector]
            long_details.append(f"LONG {sector}: ¥{value:,.2f}")
            total_value += value
        
        # Process short positions
        for sector in self.short_positions:
            value = self.holdings.loc[date, sector]
            short_details.append(f"SHORT {sector}: ¥{abs(value):,.2f}")
            total_value += value  # Short positions are negative, but we add them to total
            
        # Format output
        if long_details or short_details:
            output = []
            if long_details:
                output.append("Long Positions:\n  " + "\n  ".join(long_details))
            if short_details:
                output.append("Short Positions:\n  " + "\n  ".join(short_details))
            self.logger.info(f"{title}:\n" + "\n".join(output))
        else:
            self.logger.info(f"{title}: No holdings")
        
        self.logger.info(f"Total Portfolio Value: ¥{total_value:,.2f}")

    def _rebalance(self, date, prev_date):
        """
        Execute portfolio rebalancing to new sectors with long/short support
        
        Steps:
        1. Select new target sectors for long and short positions
        2. Clear existing holdings
        3. Allocate capital equally to new positions
        
        :param date: Current date (rebalance date)
        :param prev_date: Previous trading date
        """
        # Only rebalance on month transitions
        if date.month == prev_date.month:
            return False
            
        # Random sector selection (2-4 long, 1-3 short)
        n_long = np.random.randint(2, 5)
        n_short = np.random.randint(1, 4)
        
        # Ensure we don't select more sectors than available
        n_long = min(n_long, len(self.sectors))
        n_short = min(n_short, len(self.sectors) - n_long)
        
        # Select long and short positions
        all_sectors = np.random.permutation(self.sectors)
        long_sectors = all_sectors[:n_long].tolist()
        short_sectors = all_sectors[n_long:n_long+n_short].tolist()
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"🔁 Rebalance at {date.strftime('%Y-%m-%d')}")
        self.logger.info(f"Long sectors: {', '.join(long_sectors)}")
        self.logger.info(f"Short sectors: {', '.join(short_sectors)}")
        
        # Track rebalance date
        self.rebalance_dates.append(date)
        
        # Get NAV from previous day
        prev_nav = self.nav_record[prev_date]
        self.logger.info(f"Pre-rebalance NAV: ¥{prev_nav:,.2f}")
        
        # Clear current holdings (if any)
        if self.long_positions or self.short_positions:
            all_positions = self.long_positions + self.short_positions
            self.logger.info(f"Clearing existing positions: {', '.join(all_positions)}")
            self.holdings.loc[date, all_positions] = 0
        
        # Calculate allocation amounts
        total_positions = len(long_sectors) + len(short_sectors)
        allocation = prev_nav / total_positions if total_positions else 0
        
        # Allocate to long positions (positive values)
        self.holdings.loc[date, long_sectors] = allocation
        self.long_positions = long_sectors
        
        # Allocate to short positions (negative values)
        self.holdings.loc[date, short_sectors] = -allocation
        self.short_positions = short_sectors
        
        # Update NAV (remains same until daily update)
        self.nav_record[date] = prev_nav
        
        # Log results
        self.logger.info(f"Allocation: {total_positions} positions × ¥{allocation:,.2f}")
        self._log_holdings(date, "Post-rebalance Holdings")
        self.logger.info(f"{'='*50}")
        
        return True

    def _update_daily_holdings(self, date, prev_date):
        """
        Update holdings based on daily returns with long/short support
        
        Steps:
        1. Inherit holdings from previous day
        2. Apply daily returns to each position
        3. Calculate new NAV
        
        :param date: Current date
        :param prev_date: Previous trading date
        """
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"📈 Daily Update: {date.strftime('%Y-%m-%d')}")
        
        # Initialize with previous holdings if not set
        if self.holdings.loc[date].sum() == 0:
            self.logger.info("Inheriting holdings from previous day")
            self.holdings.loc[date] = self.holdings.loc[prev_date].copy()
        
        # Track changes
        total_change = 0
        prev_nav = self.nav_record[prev_date]
        
        self.logger.info("Applying daily returns:")
        all_positions = self.long_positions + self.short_positions
        
        for sector in all_positions:
            # Get previous value and current return
            prev_value = self.holdings.loc[prev_date, sector]
            ret = self.df_returns.loc[date, sector]
            
            # Handle invalid returns
            if pd.isna(ret):
                self.logger.warning(f"  {sector}: Invalid return (NaN), using 0")
                ret = 0
                
            # Determine position type and calculate return impact
            if sector in self.long_positions:
                # Long positions gain when returns are positive
                new_value = self.holdings.loc[date, sector] * (1 + ret)
                position_type = "LONG"
            else:
                # Short positions gain when returns are negative
                new_value = self.holdings.loc[date, sector] * (1 - ret)
                position_type = "SHORT"
                
            change = new_value - prev_value
            total_change += change
            
            # Update holding
            self.holdings.loc[date, sector] = new_value
            
            self.logger.info(
                f"  {position_type} {sector}: "
                f"Prev=¥{prev_value:,.2f}, "
                f"Return={ret:.4f}, "
                f"New=¥{new_value:,.2f}, "
                f"Δ=¥{change:+,.2f}"
            )
        
        # Update NAV
        current_nav = self.holdings.loc[date, all_positions].sum()
        self.nav_record[date] = current_nav
        
        # Log NAV change
        daily_return = (current_nav / prev_nav) - 1
        self.logger.info(
            f"NAV Change: "
            f"¥{prev_nav:,.2f} → ¥{current_nav:,.2f}, "
            f"Return={daily_return:.4f}, "
            f"Total Δ=¥{total_change:+,.2f}"
        )
        self.logger.info(f"{'='*50}")

    def run_backtest(self):
        """Execute the full backtest"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🚀 Starting Backtest")
        self.logger.info("="*50)
        
        # Initialize data structures
        self._initialize_records()
        
        # First day initialization
        self.logger.info("\n" + "="*50)
        self.logger.info(f"🌱 Initial Portfolio: {self.dates[0].strftime('%Y-%m-%d')}")
        
        # Select initial positions (2 long, 1 short)
        all_sectors = np.random.permutation(self.sectors)
        long_sectors = all_sectors[:2].tolist()
        short_sectors = all_sectors[2:3].tolist()
        
        total_positions = len(long_sectors) + len(short_sectors)
        allocation = self.initial_capital / total_positions
        
        # Set initial holdings
        self.holdings.loc[self.dates[0], long_sectors] = allocation
        self.holdings.loc[self.dates[0], short_sectors] = -allocation
        self.long_positions = long_sectors
        self.short_positions = short_sectors
        self.nav_record[self.dates[0]] = self.initial_capital
        
        self.logger.info(f"Long positions: {', '.join(long_sectors)}")
        self.logger.info(f"Short positions: {', '.join(short_sectors)}")
        self.logger.info(f"Allocation: {total_positions} positions × ¥{allocation:,.2f}")
        self._log_holdings(self.dates[0], "Initial Holdings")
        self.logger.info("="*50)
        
        # Main backtest loop
        for i in range(1, len(self.dates)):
            current_date = self.dates[i]
            prev_date = self.dates[i-1]
            
            # Check for rebalance (month transition)
            rebalanced = self._rebalance(current_date, prev_date)
            
            # Update daily holdings if not rebalanced
            if not rebalanced:
                self._update_daily_holdings(current_date, prev_date)
        
        self.logger.info("\n" + "="*50)
        self.logger.info("🏁 Backtest Completed Successfully")
        self.logger.info("="*50)
        
        return self._generate_results()

    def _generate_results(self):
        """Compile backtest results into a DataFrame"""
        results = pd.DataFrame(index=self.dates)
        
        # Core results
        results['Total_NAV'] = self.nav_record
        results['Daily_Return'] = results['Total_NAV'].pct_change().fillna(0)
        results['Cumulative_Return'] = (1 + results['Daily_Return']).cumprod() - 1
        
        # Holdings data
        for sector in self.sectors:
            results[f'Holding_{sector}'] = self.holdings[sector]
        
        # Position flags
        results['Long_Positions'] = ""
        results['Short_Positions'] = ""
        
        for date in self.dates:
            if date in self.rebalance_dates:
                results.at[date, 'Long_Positions'] = ", ".join(self.long_positions)
                results.at[date, 'Short_Positions'] = ", ".join(self.short_positions)
        
        # Rebalance flag
        results['Rebalance_Day'] = False
        results.loc[self.rebalance_dates, 'Rebalance_Day'] = True
        
        return results

def generate_sample_data(start_date='2025-01-01', end_date='2025-06-30', n_sectors=31):
    """
    Generate sample sector return data
    
    Parameters:
    :param start_date: Start date (YYYY-MM-DD)
    :param end_date: End date (YYYY-MM-DD)
    :param n_sectors: Number of sectors to generate
    
    Returns:
    DataFrame with daily returns for each sector
    """
    # Generate business days
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    # Create sector names
    sectors = [f'Sector_{i:02d}' for i in range(1, n_sectors+1)]
    
    # Generate random returns (-5% to +5%)
    returns = np.random.uniform(-0.05, 0.05, size=(len(dates), n_sectors))
    
    return pd.DataFrame(returns, index=dates, columns=sectors)

if __name__ == "__main__":
    # Setup logger
    logger = setup_logger()
    logger.info("Starting Long/Short Sector Rotation Backtest")
    
    try:
        # Generate sample data
        logger.info("Generating sample data...")
        sample_data = generate_sample_data()
        logger.info(f"Data range: {sample_data.index[0].strftime('%Y-%m-%d')} to {sample_data.index[-1].strftime('%Y-%m-%d')}")
        logger.info(f"Sectors: {len(sample_data.columns)}")
        logger.info(f"Trading days: {len(sample_data)}")
        
        # Initialize backtester
        logger.info("Initializing backtester...")
        backtester = SectorBacktester(sample_data, logger=logger)
        
        # Run backtest
        logger.info("Running backtest...")
        results = backtester.run_backtest()
        
        # Output summary
        logger.info("\nBacktest Results Summary:")
        logger.info(f"Initial capital: ¥{backtester.initial_capital:,.2f}")
        logger.info(f"Final NAV: ¥{results['Total_NAV'].iloc[-1]:,.2f}")
        total_return = (results['Total_NAV'].iloc[-1] / backtester.initial_capital - 1) * 100
        logger.info(f"Total return: {total_return:.2f}%")
        
        # Calculate performance metrics
        daily_returns = results['Daily_Return']
        annualized_return = (1 + daily_returns.mean())**252 - 1
        annualized_vol = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        logger.info(f"Annualized return: {annualized_return:.2%}")
        logger.info(f"Annualized volatility: {annualized_vol:.2%}")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
        
        # Save results
        results.to_csv('long_short_backtest_results.csv')
        logger.info("Results saved to long_short_backtest_results.csv")
        
        # Data validation
        nan_count = results.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Data validation: Found {nan_count} NaN values")
        else:
            logger.info("Data validation: No NaN values detected")
            
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.exception(f"Backtest failed: {str(e)}")
        raise
