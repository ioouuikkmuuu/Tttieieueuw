import pandas as pd
import numpy as np
import logging
from datetime import datetime

def setup_logger():
    """Configure logger for backtesting"""
    logger = logging.getLogger('backtest')
    logger.setLevel(logging.DEBUG)
    
    # Console handler for INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler for DEBUG level
    file_handler = logging.FileHandler('backtest.log')
    file_handler.setLevel(logging.DEBUG)
    
    # Log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

class SectorBacktester:
    """Backtesting engine for sector rotation strategies"""
    
    def __init__(self, df_returns, initial_capital=1e6, logger=None):
        """Initialize backtester with returns data and capital"""
        self.df_returns = df_returns.fillna(0)  # Replace missing returns with 0
        self.initial_capital = initial_capital
        self.dates = df_returns.index
        self.sectors = df_returns.columns
        self.rebalance_dates = []  # Track rebalance dates
        self.logger = logger or logging.getLogger('backtest')
        
    def _initialize_records(self):
        """Initialize data structures for tracking results"""
        self.logger.info("Initializing data structures...")
        
        # NAV time series
        self.nav_record = pd.Series(index=self.dates, dtype=float)
        self.nav_record.iloc[0] = self.initial_capital
        
        # Holdings matrix (aligned with returns data)
        self.holdings = pd.DataFrame(
            0, 
            index=self.df_returns.index,
            columns=self.df_returns.columns
        )
        
        # Current holdings list
        self.current_holdings = []
        self.logger.info(f"Initial capital: ¥{self.initial_capital:,.2f}")

    def _log_holdings(self, date, title="Current Holdings"):
        """Log detailed holdings information"""
        if self.current_holdings:
            holding_details = []
            for sector in self.current_holdings:
                value = self.holdings.loc[date, sector]
                holding_details.append(f"{sector}: ¥{value:,.2f}")
            self.logger.info(f"{title}:\n  " + "\n  ".join(holding_details))
        else:
            self.logger.info(f"{title}: No holdings")
        
        total_nav = self.nav_record.get(date, 0)
        self.logger.info(f"Total NAV: ¥{total_nav:,.2f}")

    def _rebalance(self, date, prev_date):
        """Execute rebalancing at month start"""
        # Skip if not month transition
        if date.month == prev_date.month:
            return
            
        # Randomly select target sectors (3-5 sectors)
        target_sectors = np.random.choice(
            self.sectors, 
            size=np.random.randint(3, 6),
            replace=False
        ).tolist()
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"📊 Rebalance Day: {date.strftime('%Y-%m-%d')}")
        self.logger.info(f"Target Sectors: {', '.join(target_sectors)}")
        
        # Track rebalance date
        self.rebalance_dates.append(date)
        
        # Get previous day's NAV as base
        total_value = self.nav_record[prev_date]
        self.logger.info(f"Pre-rebalance NAV: ¥{total_value:,.2f}")
        
        # Clear current holdings
        if self.current_holdings:
            self.logger.info(f"Clearing holdings: {', '.join(self.current_holdings)}")
            self.holdings.loc[date, self.current_holdings] = 0
        
        # Allocate equally to new sectors
        n_sectors = len(target_sectors)
        allocation = total_value / n_sectors if n_sectors > 0 else 0
        
        # Assign to new sectors
        self.holdings.loc[date, target_sectors] = allocation
        self.current_holdings = target_sectors
        
        # Log rebalance details
        self.logger.info(f"Allocation: {n_sectors} sectors × ¥{allocation:,.2f}")
        self._log_holdings(date, "Post-rebalance Holdings")
        self.logger.info(f"{'='*50}\n")

    def _update_daily_holdings(self, date, prev_date):
        """Update holdings for non-rebalance days"""
        # Skip first day
        if date == self.dates[0]:
            return
            
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"📅 Date: {date.strftime('%Y-%m-%d')}")
        
        # Inherit holdings if not set
        if date not in self.holdings.index or self.holdings.loc[date].sum() == 0:
            self.logger.info("Inheriting previous holdings")
            self.holdings.loc[date] = self.holdings.loc[prev_date].copy()
        
        # Update holdings with daily returns
        self.logger.info("Holdings Update:")
        total_change = 0
        prev_total = self.nav_record.get(prev_date, self.initial_capital)
        
        for sector in self.current_holdings:
            prev_value = self.holdings.loc[prev_date, sector]
            current_value = self.holdings.loc[date, sector]
            
            # Use previous value if current not set
            if current_value == 0:
                current_value = prev_value
                
            ret = self.df_returns.loc[date, sector]
            
            # Handle NaN returns
            if pd.isna(ret):
                self.logger.warning(f"  {sector}: NaN return replaced with 0")
                ret = 0
                
            # Calculate value change
            new_value = current_value * (1 + ret)
            change = new_value - prev_value
            total_change += change
            
            self.holdings.loc[date, sector] = new_value
            
            self.logger.info(
                f"  {sector}: "
                f"Prev=¥{prev_value:,.2f}, "
                f"Return={ret:.4f}, "
                f"New=¥{new_value:,.2f}, "
                f"Δ=¥{change:+,.2f}"
            )
        
        # Update total NAV
        current_total = self.holdings.loc[date, self.current_holdings].sum()
        self.nav_record[date] = current_total
        
        # Log NAV change
        daily_ret = (current_total - prev_total) / prev_total
        self.logger.info(
            f"NAV Change: "
            f"Prev=¥{prev_total:,.2f} → "
            f"New=¥{current_total:,.2f}, "
            f"Return={daily_ret:.4f}, "
            f"Total Δ=¥{total_change:+,.2f}"
        )
        self.logger.info(f"{'='*50}\n")

    def run_backtest(self):
        """Execute backtest"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🏁 Starting Backtest")
        self.logger.info("="*50 + "\n")
        
        self._initialize_records()
        
        # First day initialization
        first_sectors = np.random.choice(
            self.sectors, 
            size=3, 
            replace=False
        ).tolist()
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"📊 Initialization: {self.dates[0].strftime('%Y-%m-%d')}")
        
        # Initial allocation
        allocation = self.initial_capital / len(first_sectors)
        self.holdings.loc[self.dates[0], first_sectors] = allocation
        self.current_holdings = first_sectors
        self.nav_record[self.dates[0]] = self.initial_capital
        
        self.logger.info(f"Allocation: {len(first_sectors)} sectors × ¥{allocation:,.2f}")
        self._log_holdings(self.dates[0], "Initial Holdings")
        self.logger.info(f"{'='*50}\n")
        
        # Main loop
        for i in range(1, len(self.dates)):
            current_date = self.dates[i]
            prev_date = self.dates[i-1]
            
            # Check for rebalance (month transition)
            self._rebalance(current_date, prev_date)
            
            # Update daily holdings
            self._update_daily_holdings(current_date, prev_date)
        
        self.logger.info("\n" + "="*50)
        self.logger.info("🏁 Backtest Complete")
        self.logger.info("="*50 + "\n")
        return self._generate_results()

    def _generate_results(self):
        """Generate results DataFrame"""
        results = pd.DataFrame(index=self.dates)
        results['Total_NAV'] = self.nav_record
        results['Daily_Return'] = results['Total_NAV'].pct_change().fillna(0)
        
        # Add sector holdings
        for sector in self.sectors:
            results[f'Holding_{sector}'] = self.holdings[sector]
        
        # Add rebalance flag
        results['Rebalance_Day'] = False
        results.loc[self.rebalance_dates, 'Rebalance_Day'] = True
        
        # Calculate cumulative returns
        results['Cumulative_Return'] = (1 + results['Daily_Return']).cumprod() - 1
        
        return results

def generate_sample_data(start_date='2025-01-01', end_date='2025-06-30', n_sectors=31):
    """Generate sample returns data"""
    # Business days in date range
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    # Sector names
    sectors = [f'Sector_{i:02d}' for i in range(1, n_sectors+1)]
    
    # Random returns (-2% to +3%)
    returns = np.random.uniform(-0.02, 0.03, size=(len(dates), n_sectors))
    
    # Create DataFrame
    return pd.DataFrame(returns, index=dates, columns=sectors)

if __name__ == "__main__":
    # Setup logger
    logger = setup_logger()
    
    # Generate sample data
    logger.info("Generating sample data...")
    sample_data = generate_sample_data()
    
    # Log data overview
    logger.info(f"Date Range: {sample_data.index[0].strftime('%Y-%m-%d')} to {sample_data.index[-1].strftime('%Y-%m-%d')}")
    logger.info(f"Sectors: {len(sample_data.columns)}")
    logger.info(f"Trading Days: {len(sample_data)}")
    
    # Initialize backtester
    logger.info("Initializing backtester...")
    backtester = SectorBacktester(sample_data, logger=logger)
    
    # Run backtest
    logger.info("Starting backtest...")
    results = backtester.run_backtest()
    
    # Output summary
    logger.info("Backtest Results:")
    logger.info(f"Initial Capital: ¥{backtester.initial_capital:,.2f}")
    logger.info(f"Final NAV: ¥{results['Total_NAV'].iloc[-1]:,.2f}")
    total_return = (results['Total_NAV'].iloc[-1]/backtester.initial_capital-1)*100
    logger.info(f"Total Return: {total_return:.2f}%")
    
    # Save results
    results.to_csv('sector_backtest_results.csv')
    logger.info("Results saved to sector_backtest_results.csv")
    logger.info("Log saved to backtest.log")
    
    # Validate data
    nan_count = results.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Warning: Found {nan_count} NaN values in results")
    else:
        logger.info("Data Validation: No NaN values found")
