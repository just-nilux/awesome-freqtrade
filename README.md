<div style="width: 100%;">
  <a href="https://github.com/just-nilux/awesome-freqtrade/blame/main/awesome-freqtrade.svg">
    <img src="awesome-freqtrade.svg" style="width: 100%;" alt="Click to see the source">
  </a>
</div>

# Freqtrade & FreqAI Snippets [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A collection of Freqtrade & FreqAI snippets, all in one place. Since Discord provides such a poor search function, this list should make it easier to find all the useful code posted. 

[Freqtrade Discord](https://discord.com/invite/T7SmVvQ8sD) | [Documentation](https://www.freqtrade.io/en/stable/) | [Github](https://github.com/freqtrade/freqtrade)

#### Contribute
:memo: **It's easy to contribute!** Create an Issue or PR with the content you like to add.

#### Credits
What I are doing here is mostly collecting useful snippets from all over the [Freqtrade Discord](https://discord.com/invite/T7SmVvQ8sD) and [FreqAI Discord](https://discord.com/invite/hYuzJYKFjz) into one place. While I’ve been trying to credit where due, things might be missing. If you believe anything here is your work and credits should be given, let me know, or just send a PR.

## Table of Contents
- [Freqtrade](#freqtrade)
    - [Code Snippets](#freqtrade-code-snippets)
    - [Indicators](#freqtrade-indicators)
    - [Strategies](#freqtrade-strategies)
- [FreqAI](#freqai-general)
    - [Code Snippets](#freqai-code-snippets)
    - [Feature Engineering](#freqai-feature-snippets)
    - [Custom Models](#freqai-models)
    - [Strategies](#freqai-strategies)
- [Backtesting / HyperOpt](#freqtrade-backtesting)
    - [Backtesting](#backtest-placeholder)
    - [HyperOpt](#hyperopt-placeholder)
- [Knowledge Base](#freqtrade-knowledge-base)
    - [Trading with Python](#placeholder)
    - [Awesome-Quant](#placeholder)

## Freqtrade
Freqtrade is a free and open source crypto trading bot written in Python. It is designed to support all major exchanges and be controlled via Telegram or webUI. It contains backtesting, plotting and money management tools as well as strategy optimization by machine learning. 

[Discord](https://discord.com/invite/T7SmVvQ8sD) | [Documentation](https://www.freqtrade.io/en/stable/) | [Github](https://github.com/freqtrade/freqtrade)

### Freqtrade Code Snippets
Various code snippets
<details>
<summary>Custom Trailing Stoploss</summary>
  
#### Custom Trailing Stoploss by Permeister
```python
    # In the beging of the strategy file
    # Hyperopt Parameters
    # hard stoploss profit
    pHSL = DecimalParameter(-0.200, -0.040, default=-0.10, decimals=3, space='sell', optimize=True, load=True)
    # profit threshold 1, trigger point, SL_1 is used
    pPF_1 = DecimalParameter(0.008, 0.020, default=0.016, decimals=3, space='sell', optimize=True, load=True)
    pSL_1 = DecimalParameter(0.008, 0.020, default=0.011, decimals=3, space='sell', optimize=True, load=True)
    # profit threshold 2, SL_2 is used
    pPF_2 = DecimalParameter(0.040, 0.100, default=0.070, decimals=3, space='sell', optimize=True, load=True)
    pSL_2 = DecimalParameter(0.020, 0.070, default=0.030, decimals=3, space='sell', optimize=True, load=True)
```

```python
def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> float:

    # hard stoploss profit
    HSL = self.pHSL.value
    PF_1 = self.pPF_1.value
    SL_1 = self.pSL_1.value
    PF_2 = self.pPF_2.value
    SL_2 = self.pSL_2.value

    # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated 
    # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
    # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

    if (current_profit > PF_2):
        sl_profit = SL_2 + (current_profit - PF_2)
    elif (current_profit > PF_1):
        sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
    else:
        sl_profit = HSL

    # Only for hyperopt invalid return
    if (sl_profit >= current_profit):
        return -0.99

    return stoploss_from_open(sl_profit, current_profit)
```
</details>

### Freqtrade Indicators
Various Freqtrade Indicators

#### SMI Trend (by nilux)
```Python
def smi_trend(df: DataFrame, k_length=9, d_length=3, smoothing_type='EMA', smoothing=10):
    """     
    Stochastic Momentum Index (SMI) Trend Indicator 
        
    SMI > 0 and SMI > MA: (2) Bull
    SMI < 0 and SMI > MA: (1) Possible Bullish Reversal

    SMI > 0 and SMI < MA: (-1) Possible Bearish Reversal
    SMI < 0 and SMI < MA: (-2) Bear
        
    Returns:
        pandas.Series: New feature generated 
    """
    
    ll = df['low'].rolling(window=k_length).min()
    hh = df['high'].rolling(window=k_length).max()

    diff = hh - ll
    rdiff = df['close'] - (hh + ll) / 2

    avgrel = rdiff.ewm(span=d_length).mean().ewm(span=d_length).mean()
    avgdiff = diff.ewm(span=d_length).mean().ewm(span=d_length).mean()

    smi = np.where(avgdiff != 0, (avgrel / (avgdiff / 2) * 100), 0)
    
    if smoothing_type == 'SMA':
        smi_ma = ta.SMA(smi, timeperiod=smoothing)
    elif smoothing_type == 'EMA':
        smi_ma = ta.EMA(smi, timeperiod=smoothing)
    elif smoothing_type == 'WMA':
        smi_ma = ta.WMA(smi, timeperiod=smoothing)
    elif smoothing_type == 'DEMA':
        smi_ma = ta.DEMA(smi, timeperiod=smoothing)
    elif smoothing_type == 'TEMA':
        smi_ma = ta.TEMA(smi, timeperiod=smoothing)
    else:
        raise ValueError("Choose an MA Type: 'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA'")

    conditions = [
        (np.greater(smi, 0) & np.greater(smi, smi_ma)), # (2) Bull 
        (np.less(smi, 0) & np.greater(smi, smi_ma)),    # (1) Possible Bullish Reversal
        (np.greater(smi, 0) & np.less(smi, smi_ma)),    # (-1) Possible Bearish Reversal
        (np.less(smi, 0) & np.less(smi, smi_ma))        # (-2) Bear
    ]

    smi_trend = np.select(conditions, [2, 1, -1, -2])

    return smi, smi_ma, smi_trend

```
