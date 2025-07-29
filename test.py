import pandas as pd
import numpy as np
from datetime import datetime


# ======================
# 核心模块：回测引擎
# ======================
class SectorBacktester:
    """
    行业轮动回测引擎
    :param df_returns: 输入数据框，index为交易日，columns为行业名称，value为日收益率
    :param initial_capital: 初始资金 (默认100万)
    """

    def __init__(self, df_returns, initial_capital=1e6):
        self.df_returns = df_returns
        self.initial_capital = initial_capital
        self.dates = df_returns.index
        self.sectors = df_returns.columns
        self.rebalance_dates = []  # 记录调仓日期

    def _initialize_records(self):
        """初始化记录数据结构"""
        # 资金曲线记录
        self.nav_record = pd.Series(index=self.dates, dtype=float)
        self.nav_record.iloc[0] = self.initial_capital

        # 持仓记录 (每个行业的市值)
        self.holdings = pd.DataFrame(
            index=self.dates,
            columns=self.sectors,
            dtype=float
        ).fillna(0)

        # 持仓行业列表记录
        self.current_holdings = []

    def _rebalance(self, date, target_sectors):
        """
        执行调仓操作
        :param date: 调仓日期
        :param target_sectors: 目标持仓行业列表
        """
        # 记录调仓日期
        self.rebalance_dates.append(date)

        # 清空当前持仓
        total_value = self.nav_record[date]
        self.holdings.loc[date, self.current_holdings] = 0

        # 平均分配资金到新行业
        n_sectors = len(target_sectors)
        allocation = total_value / n_sectors if n_sectors > 0 else 0

        # 更新持仓
        self.current_holdings = target_sectors
        self.holdings.loc[date, target_sectors] = allocation

    def _update_holdings(self, date_idx):
        """更新每日持仓市值"""
        current_date = self.dates[date_idx]
        prev_date = self.dates[date_idx - 1]

        # 复制前一日持仓
        self.holdings.loc[current_date] = self.holdings.loc[prev_date]

        # 更新当日收益
        for sector in self.current_holdings:
            ret = self.df_returns.loc[current_date, sector]
            self.holdings.loc[current_date, sector] *= (1 + ret)

        # 更新总资产
        self.nav_record[current_date] = self.holdings.loc[current_date, self.current_holdings].sum()

    def run_backtest(self):
        """执行回测"""
        self._initialize_records()

        # 首日初始化 (假设第一个交易日需要调仓)
        first_date = self.dates[0]
        self._rebalance(first_date, [])

        # 主循环
        for i in range(1, len(self.dates)):
            current_date = self.dates[i]
            prev_date = self.dates[i - 1]

            # 月初调仓逻辑 (实际应用中替换为具体调仓规则)
            if current_date.month != prev_date.month:
                # 示例：随机选择3-5个行业 (实际应替换为策略逻辑)
                target_sectors = np.random.choice(
                    self.sectors,
                    size=np.random.randint(3, 6),
                    replace=False
                ).tolist()
                self._rebalance(current_date, target_sectors)
            else:
                self._update_holdings(i)

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

        return results


# ======================
# 辅助模块：样本数据生成
# ======================
def generate_sample_data(start_date='2025-01-01', end_date='2025-06-30', n_sectors=31):
    """
    生成样本数据
    :param start_date: 起始日期
    :param end_date: 结束日期
    :param n_sectors: 行业数量
    """
    # 生成交易日
    dates = pd.bdate_range(start=start_date, end=end_date)

    # 生成行业名称
    sectors = [f'Sector_{i:02d}' for i in range(1, n_sectors + 1)]

    # 生成随机收益率 (-2% 到 +3%)
    returns = np.random.uniform(-0.02, 0.03, size=(len(dates), n_sectors))

    # 创建DataFrame
    df = pd.DataFrame(returns, index=dates, columns=sectors)

    # 添加日期特征
    df['Month'] = df.index.month
    df['Day'] = df.index.day

    return df.drop(columns=['Month', 'Day'])


# ======================
# 执行示例
# ======================
if __name__ == "__main__":
    # 生成样本数据 (31个行业，6个月数据)
    sample_data = generate_sample_data()
    print("样本数据预览:")
    print(sample_data.head())
    print("\n数据形状:", sample_data.shape)

    # 初始化回测引擎
    backtester = SectorBacktester(sample_data)

    # 运行回测
    results = backtester.run_backtest()

    # 输出结果
    print("\n回测结果:")
    print(results[['Total_NAV', 'Daily_Return', 'Rebalance_Day']].head())

    # 分析结果
    print("\n绩效指标:")
    print(f"初始资金: {backtester.initial_capital:,.2f}")
    print(f"最终资金: {results['Total_NAV'].iloc[-1]:,.2f}")
    print(f"总收益率: {(results['Total_NAV'].iloc[-1] / backtester.initial_capital - 1) * 100:.2f}%")

    # 保存结果
    results.to_csv('sector_backtest_results.csv')
    print("\n结果已保存至 sector_backtest_results.csv")
