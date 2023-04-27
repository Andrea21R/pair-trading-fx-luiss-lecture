import os
import psutil
from itertools import combinations
from typing import *

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint as eg_coint

import ray


@ray.remote
def clean_single_ticker(data: pd.Series, col: str, threshold_ret: float, verbose: bool) -> list:
    data_ = data.copy(deep=True)
    fix_done = 0
    for t in range(1, len(data)):
        tmp_ret = data_.iloc[t] / data_.iloc[t - 1] - 1
        if abs(tmp_ret) > threshold_ret:
            data_.iloc[t] = data_.iloc[t - 1]
            fix_done += 1
    if verbose:
        print(f'fixes done to {col}: {fix_done}')

    return [col, data_]


def clean_data(data: pd.DataFrame, anomalous_ret: float, plot: bool, verbose: bool) -> pd.DataFrame:
    if plot:
        data.pct_change().plot(title='Data Returns pre Cleaning', legend=None, figsize=(14, 4), grid=True)

    ray.init()
    id_results = []
    for col in data.columns:
        id_results.append(clean_single_ticker.remote(data[col], col, anomalous_ret, verbose))
    res = ray.get(id_results)
    ray.shutdown()

    clean_data = pd.DataFrame({k: v for k, v in res})

    if plot:
        clean_data.pct_change().plot(title='Data Returns post Cleaning', legend=None, figsize=(14, 4), grid=True)

    return clean_data


@ray.remote
def calc_pair_statistical_info(input: List[Any]) -> list:

    # ---------------- input unpack
    ccy_1, ccy_2 = input[0][0], input[0][1]
    price_ccy_1, price_ccy_2 = input[1][ccy_1], input[1][ccy_2]
    norm_prices_ccy_1, norm_prices_ccy_2 = input[2][ccy_1], input[2][ccy_2]
    verbose, idx = input[3], input[4]

    # ---------------- computation
    combo_key = ccy_1 + "|" + ccy_2
    if verbose:
        print(f"{idx}| Computing: {combo_key}")
    diff = norm_prices_ccy_1 - norm_prices_ccy_2
    tv = np.mean(diff.pow(2))
    spread_sd = np.std(diff)
    # get min value between coint(ccy_1, ccy_2) and coint(ccy_2, ccy_1)
    eg_coint_pval = min(
        eg_coint(price_ccy_1, price_ccy_2, method='aeg')[1],
        eg_coint(price_ccy_2, price_ccy_1, method='aeg')[1]
    )
    return [{'combo_key': combo_key, 'tv': tv, 'spread_sd': spread_sd, 'eg_coint_pval': eg_coint_pval}]


def calc_tv_and_coint_for_pair_trading(price_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:

    norm_prices = price_df.div(price_df.iloc[0])
    combo = combinations(price_df.columns, r=2)
    combo = [(ccy_1, ccy_2) for ccy_1, ccy_2 in combo if ccy_1 != ccy_2[-3:] + ccy_2[:3]]

    ray.init()
    result_ids = []
    for idx, ccys in enumerate(combo):
        result_ids.append(
            calc_pair_statistical_info.remote([ccys, price_df, norm_prices, verbose, idx])
        )
    result = ray.get(result_ids)
    ray.shutdown()

    assets_info = {}
    for output in result:
        output = output[0]
        assets_info[output['combo_key']] = {
            'tv'           : output['tv'],
            'spread_sd'    : output['spread_sd'],
            'eg_coint_pval': output['eg_coint_pval']
        }
    assets_info = pd.DataFrame(assets_info).transpose().sort_values(by='tv', ascending=True)
    return assets_info


def find_best_assets_for_pair_trading(
        price_df: pd.DataFrame,
        tv_quantile_threshold: float = 0.1,
        coint_threshold: float = 0.05,
        top_n_assets: int = 10,
        verbose: bool = False
) -> pd.DataFrame:
    res = calc_tv_and_coint_for_pair_trading(price_df, verbose=verbose)

    tv_threshold    = res['tv'].quantile(tv_quantile_threshold)
    top_n = top_n_assets

    top_pairs = res[(res['tv'] <= tv_threshold) & (res['eg_coint_pval'] <= coint_threshold)].iloc[:top_n]

    return top_pairs.sort_values(by='tv', ascending=True)


def backtest_strategy(
        price_df: pd.DataFrame,
        tgt_pairs: List[str],
        pairs_spread_sd: Dict[str, float],
        n_std_open: float = 2,
        n_std_close: float = 0.5,
        stop_loss: Optional[dict] = None  # {'extra_spread_sd': 2, 'spread_sd_to_restart': 1.5}
) -> Dict[str, pd.DataFrame]:

    rets = price_df.pct_change()
    norm_prices = price_df.div(price_df.iloc[0])

    pnl = {pair: pd.Series(dtype="float64", index=price_df.index, name=pair) for pair in tgt_pairs}

    for pair in tgt_pairs:

        ccy_1, ccy_2 = pair.split("|")

        trade_open = False
        stop_trading = False
        last_delta = None
        for t, t_data in norm_prices.iterrows():
            delta = t_data[ccy_1] - t_data[ccy_2]

            if not stop_trading:
                if not trade_open:
                    if abs(delta) >= pairs_spread_sd[pair] * n_std_open:
                        trade_open = True
                        if t_data[ccy_1] > t_data[ccy_2]:
                            short_ccy = ccy_1
                            long_ccy  = ccy_2
                        else:
                            short_ccy = ccy_2
                            long_ccy  = ccy_1
                # if the trade is open, compute the pnl
                else:
                    short_ret = -rets.loc[t, short_ccy]
                    long_ret  = rets.loc[t, long_ccy]
                    tot_rets  = short_ret + long_ret

                    pnl[pair].loc[t] = tot_rets

                if trade_open:
                    # take profit
                    if (abs(delta) < pairs_spread_sd[pair] * n_std_close) or (last_delta * delta < 0):
                        trade_open = False
                    # stop loss
                    if stop_loss and (abs(delta) > pairs_spread_sd[pair] * (n_std_open + stop_loss['extra_spread_sd'])):
                        trade_open = False
                        stop_trading = True
            else:
                if abs(delta) < pairs_spread_sd[pair] * stop_loss['spread_sd_to_restart']:
                    stop_trading = False

            last_delta = delta

    pnl = pd.concat(pnl, axis=1).fillna(0)
    tot_pnl = pnl.sum(axis=1)

    return {
        'pnl_breakdown': pnl,
        'pnl_total': tot_pnl,
        'cum_pnl_breakdown': pnl.cumsum(),  # not compounded
        'cum_pnl_total': tot_pnl.cumsum(),  # not compounded
    }


def get_stats_trade_duration(pnl_breakdown: pd.DataFrame) -> Dict[str, Any]:
    trades_durations = []
    for col in pnl_breakdown.columns:
        tgt = pnl_breakdown[col]
        count = 0
        for t in range(1, len(pnl_breakdown)):
            if bool(tgt.iloc[t]):
                count += 1
            else:
                if count:
                    trades_durations.append(count)
                count = 0

    return {
        '#trade': len(trades_durations),
        'avg': sum(trades_durations) / len(trades_durations),
        'median': np.median(trades_durations),
        'min': min(trades_durations),
        'max': max(trades_durations),
        'std': np.std(trades_durations),
        'skew': pd.Series(trades_durations).skew(),
        'kurt': pd.Series(trades_durations).kurt(),
    }


def step_rolling_train_test_split(
        data: pd.DataFrame,
        train_period: int,        
        test_period: int
) -> Tuple[List[Dict[str, pd.DataFrame]], List]:

    train_test_sets = []
    for t in range(train_period, len(data), test_period):
        train = data.iloc[t - train_period: t]
        if t + test_period > len(data):
            test = data.iloc[t:]
        else:
            test = data.iloc[t: t + test_period]

        train_test_sets.append(dict(train=train, test=test))

    train_test_info = [
        {
            'train': {'start': str(set['train'].index[0]), 'end': str(set['train'].index[-1])},
            'test': {'start': str(set['test'].index[0]),   'end': str(set['test'].index[-1])}
        }
        for set in train_test_sets
    ]

    return train_test_sets, train_test_info


def backtest_strategy_with_step_rolling_approach(
        price_df: pd.DataFrame,
        train_period: int = 252 * 7,  # 7y
        test_period: int = 252 * 5,  # 5y
        tv_quantile_threshold: float = 0.10,
        coint_threshold: float = 0.05,
        max_asset_in_portfolio: int = 10,
        n_tv_to_open: float = 2,
        n_tv_to_close: float = 0.5,
        stop_loss: Optional[Dict[str, Any]] = None,
        view_ram_usage: bool = False,
        partial_return: bool = False,
        verbose: bool = False
):
    train_test_set, train_test_info  = step_rolling_train_test_split(price_df, train_period, test_period)

    pnls = []
    for idx, set in enumerate(train_test_set):
        train_set = set['train']
        test_set  = set['test']

        if verbose:
            print(f'\n{idx}|Testing period: FROM {str(test_set.index[0])} TO {str(test_set.index[-1])}')
            print('    ... Training')
        best_assets_info = find_best_assets_for_pair_trading(
            price_df=train_set,
            tv_quantile_threshold=tv_quantile_threshold,
            coint_threshold=coint_threshold,
            top_n_assets=max_asset_in_portfolio,
            verbose=verbose
        )
        if verbose:
            print('    ... Testing')
        pnls.append(
            backtest_strategy(
                price_df=test_set,
                tgt_pairs=best_assets_info.index,
                pairs_spread_sd=best_assets_info['spread_sd'],
                n_std_open=n_tv_to_open,
                n_std_close=n_tv_to_close,
                stop_loss=stop_loss
                )
        )
        if view_ram_usage:
            pid = os.getpid()
            py = psutil.Process(pid)
            memory_use = py.memory_info()[0]/2.**30  # converts in GB
            print(f"RAM-usage: {memory_use:.2f} GB")

    pnl = pd.concat([tmp_pnl['pnl_breakdown'] for tmp_pnl in pnls], axis=0).fillna(0)
    tot_pnl = pnl.sum(axis=1)

    print('\n----> BACKTESTING FINISHED')

    if not partial_return:
        pnls = {
            'pnl_breakdown': pnl,
            'pnl_total': tot_pnl,
            'cum_pnl_breakdown': pnl.cumsum(),
            'cum_pnl_total': tot_pnl.cumsum()
        }
        return pnls, train_test_info
    else:
        return pnl
