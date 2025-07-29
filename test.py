import pandas as pd
import numpy as np
import logging
from datetime import datetime


def setup_logger():
    """配置日志记录器"""
    logger = logging.getLogger('backtest')
    logger.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler('backtest.log')
    file_handler.setLevel(logging.DEBUG)

    # 设置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class SectorBacktester:
    def __init__(self, df_returns, initial_capital=1e6, logger=None):
        self.df_returns = df_returns.fillna(0)
        self.initial_capital = initial_capital
        self.dates = df_returns.index
        self.sectors = df_returns.columns
        self.rebalance_dates = []
        self.logger = logger or logging.getLogger('backtest')

    def _initialize_records(self):
        self.logger.info("初始化记录数据结构...")

        # 资金曲线记录
        self.nav_record = pd.Series(index=self.dates, dtype=float)
        self.nav_record.iloc[0] = self.initial_capital

        # 持仓矩阵与收益率数据完全对齐
        self.holdings = pd.DataFrame(
            0,
            index=self.df_returns.index,
            columns=self.df_returns.columns
        )

        # 持仓行业列表记录
        self.current_holdings = []
        self.logger.info(f"初始资金: {self.initial_capital:,.2f}")

    def _log_holdings(self, date, title="当前持仓详情"):
        if self.current_holdings:
            holding_details = []
            for sector in self.current_holdings:
                value = self.holdings.loc[date, sector]
                holding_details.append(f"{sector}: ¥{value:,.2f}")
            self.logger.info(f"{title}:\n  " + "\n  ".join(holding_details))
        else:
            self.logger.info(f"{title}: 无持仓")

        total_nav = self.nav_record.get(date, 0)
        self.logger.info(f"总资产: ¥{total_nav:,.2f}")

    def _rebalance(self, date, prev_date):
        # 仅当月切换时执行调仓
        if date.month == prev_date.month:
            return

        # 随机选择3-5个行业作为目标持仓
        target_sectors = np.random.choice(
            self.sectors,
            size=np.random.randint(3, 6),
            replace=False
        ).tolist()

        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"📊 调仓日: {date.strftime('%Y-%m-%d')}")
        self.logger.info(f"目标行业: {', '.join(target_sectors)}")

        # 记录调仓日期
        self.rebalance_dates.append(date)

        # 获取前一日总资产作为调仓金额基础
        total_value = self.nav_record[prev_date]
        self.logger.info(f"调仓前总资产: ¥{total_value:,.2f}")

        # 仅清空当前持仓的行业（避免整行归零）
        if self.current_holdings:
            self.logger.info(f"清空持仓: {', '.join(self.current_holdings)}")
            self.holdings.loc[date, self.current_holdings] = 0

        # 平均分配资金到新行业
        n_sectors = len(target_sectors)
        allocation = total_value / n_sectors if n_sectors > 0 else 0

        # 仅对新持仓行业赋值
        self.holdings.loc[date, target_sectors] = allocation
        self.current_holdings = target_sectors

        # 记录调仓详情
        self.logger.info(f"分配资金: {n_sectors}个行业 × ¥{allocation:,.2f}")
        self._log_holdings(date, "调仓后持仓详情")
        self.logger.info(f"{'=' * 50}\n")

    def _update_daily_holdings(self, date, prev_date):
        # 跳过首日，不需要更新
        if date == self.dates[0]:
            return

        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"📅 日期: {date.strftime('%Y-%m-%d')}")

        # 如果调仓日未设置持仓，使用前一日持仓
        if date not in self.holdings.index or self.holdings.loc[date].sum() == 0:
            self.logger.info("继承前一日持仓")
            self.holdings.loc[date] = self.holdings.loc[prev_date].copy()

        # 更新当日收益
        self.logger.info("持仓收益变化:")
        total_change = 0
        prev_total = self.nav_record.get(prev_date, self.initial_capital)

        for sector in self.current_holdings:
            prev_value = self.holdings.loc[prev_date, sector]
            current_value = self.holdings.loc[date, sector]

            # 如果当前值未设置，使用前一日值作为起点
            if current_value == 0:
                current_value = prev_value

            ret = self.df_returns.loc[date, sector]

            if pd.isna(ret):
                self.logger.warning(f"  {sector}: 收益率为NaN，已替换为0")
                ret = 0

            # 计算持仓价值变化
            new_value = current_value * (1 + ret)
            change = new_value - prev_value
            total_change += change

            self.holdings.loc[date, sector] = new_value

            self.logger.info(
                f"  {sector}: "
                f"前值=¥{prev_value:,.2f}, "
                f"收益率={ret:.4f}, "
                f"现值=¥{new_value:,.2f}, "
                f"变化=¥{change:+,.2f}"
            )

        # 更新总资产
        current_total = self.holdings.loc[date, self.current_holdings].sum()
        self.nav_record[date] = current_total

        # 记录总资产变化
        daily_ret = (current_total - prev_total) / prev_total
        self.logger.info(
            f"总资产变化: "
            f"前值=¥{prev_total:,.2f} → "
            f"现值=¥{current_total:,.2f}, "
            f"日收益率={daily_ret:.4f}, "
            f"总变化=¥{total_change:+,.2f}"
        )
        self.logger.info(f"{'=' * 50}\n")

    def run_backtest(self):
        self.logger.info("\n" + "=" * 50)
        self.logger.info("🏁 开始回测")
        self.logger.info("=" * 50 + "\n")

        self._initialize_records()

        # 首日必须分配持仓！
        first_sectors = np.random.choice(
            self.sectors,
            size=3,
            replace=False
        ).tolist()

        self.logger.info(f"\n{'=' * 50}")
        self.logger.info(f"📊 首日初始化: {self.dates[0].strftime('%Y-%m-%d')}")

        # 初始分配
        allocation = self.initial_capital / len(first_sectors)
        self.holdings.loc[self.dates[0], first_sectors] = allocation
        self.current_holdings = first_sectors
        self.nav_record[self.dates[0]] = self.initial_capital

        self.logger.info(f"分配资金: {len(first_sectors)}个行业 × ¥{allocation:,.2f}")
        self._log_holdings(self.dates[0], "初始持仓详情")
        self.logger.info(f"{'=' * 50}\n")

        # 主循环
        for i in range(1, len(self.dates)):
            current_date = self.dates[i]
            prev_date = self.dates[i - 1]

            # 检查是否需要调仓（月初）
            self._rebalance(current_date, prev_date)

            # 更新每日持仓
            self._update_daily_holdings(current_date, prev_date)

        self.logger.info("\n" + "=" * 50)
        self.logger.info("🏁 回测完成")
        self.logger.info("=" * 50 + "\n")
        return self._generate_results()

    def _generate_results(self):
        """生成结果报告"""
        results = pd.DataFrame(index=self.dates)
        results['Total_NAV'] = self.nav_record
        results['Daily_Return'] = results['Total_NAV'].pct_change().fillna(0)

        # 添加持仓信息
        for sector in self.sectors:
            results[f'Holding_{sector}'] = self.holdings[sector]

        # 添加调仓标记
        results['Rebalance_Day'] = False
        results.loc[self.rebalance_dates, 'Rebalance_Day'] = True

        # 计算累计收益
        results['Cumulative_Return'] = (1 + results['Daily_Return']).cumprod() - 1

        return results


def generate_sample_data(start_date='2025-01-01', end_date='2025-06-30', n_sectors=31):
    """生成样本数据"""
    dates = pd.bdate_range(start=start_date, end=end_date)
    sectors = [f'行业_{i:02d}' for i in range(1, n_sectors + 1)]
    returns = np.random.uniform(-0.02, 0.03, size=(len(dates), n_sectors))
    return pd.DataFrame(returns, index=dates, columns=sectors)


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("生成样本数据...")
    sample_data = generate_sample_data()

    # 打印数据概览
    logger.info(
        f"数据范围: {sample_data.index[0].strftime('%Y-%m-%d')} 至 {sample_data.index[-1].strftime('%Y-%m-%d')}")
    logger.info(f"行业数量: {len(sample_data.columns)}")
    logger.info(f"交易日数量: {len(sample_data)}")

    # 初始化回测引擎
    logger.info("初始化回测引擎...")
    backtester = SectorBacktester(sample_data, logger=logger)

    # 运行回测
    logger.info("开始回测...")
    results = backtester.run_backtest()

    # 输出结果
    logger.info("回测结果摘要:")
    logger.info(f"初始资金: ¥{backtester.initial_capital:,.2f}")
    logger.info(f"最终资金: ¥{results['Total_NAV'].iloc[-1]:,.2f}")
    total_return = (results['Total_NAV'].iloc[-1] / backtester.initial_capital - 1) * 100
    logger.info(f"总收益率: {total_return:.2f}%")

    # 保存结果
    results.to_csv('sector_backtest_results.csv')
    logger.info("详细结果已保存至 sector_backtest_results.csv")
    logger.info("日志已保存至 backtest.log")

    # 检查NaN值
    nan_count = results.isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"警告: 结果中存在 {nan_count} 个NaN值")
    else:
        logger.info("数据验证: 无NaN值")
