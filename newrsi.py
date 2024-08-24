import pandas as pd
import ccxt
import requests
import numpy as np
import time

# Telegram Bot配置
TOKEN = '5959250165:AAF5Xh6hLIEndRY9YzjsmMuCJwJwTsMFa8M' 
CHAT_ID = '1640026631'
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TOKEN}/sendMessage'

# Initialize ccxt exchange
exchange = ccxt.binance()
interval = '1h'

# List of cryptocurrencies to monitor
CURRENCY_IDS = [
    'BTC/USDT', 'ETH/USDT', 'UNI/USDT', 'SHIB/USDT', 'XRP/USDT', 
    'BNB/USDT', 'ADA/USDT', 'WLD/USDT', 'SOL/USDT', 'AVAX/USDT',
    'DOT/USDT', 'TON/USDT', 'DOGE/USDT'
]

def fetch_data(symbol, timeframe, limit=100):
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Convolution function
def convolve(weights, kernel, iterations):
    convolution = np.copy(weights)

    if iterations > 0:
        for _ in range(iterations):
            temp = np.zeros(len(convolution) + len(kernel) - 1)
            
            for i in range(len(temp)):
                sum_val = 0.0
                
                for j in range(len(kernel)):
                    index = i - j
                    
                    if 0 <= index < len(convolution):
                        sum_val += convolution[index] * kernel[j]
                
                temp[i] = sum_val

            convolution = np.copy(temp)

    return convolution

# Truncate weights function
def truncate_weights(weights, enable=True):
    if enable:
        max_idx = np.argmax(weights)
        
        if max_idx > 0:
            weights = weights[max_idx:]
    
    return weights

# Binomial MA function
def binomial_ma(source, length, enable):
    pre_filter = np.convolve(source, np.ones(2)/2, mode='valid')
    h = np.array([0.5, 0.5])
    weights = convolve(h, h, length * 10 - 1)
    weights = truncate_weights(weights)

    if enable and not np.isnan(source).all():
        weight = 0
        sum_val = 0

        for i in range(min(len(weights), len(source))):
            w = weights[i]
            weight += w
            sum_val += pre_filter[i] * w

        return sum_val / weight if weight != 0 else np.nan
    else:
        return np.nan

def precalculate_phi_coefficients(length, phase):
    coefficients = np.zeros(length)
    E = 0.0

    SQRT_PIx2 = np.sqrt(2.0 * np.pi)
    MULTIPLIER = -0.5 / 0.93
    length_1 = length - 1
    length_2 = length * 0.52353

    for i in range(length):
        alpha = (i + phase - length_2) * MULTIPLIER
        beta = 1.0 / (0.2316419 * np.abs(alpha) + 1.0)
        phi = (np.exp(-0.5 * alpha ** 2) * -0.398942280) * beta * (
            0.319381530 + beta * (
                -0.356563782 + beta * (
                    1.781477937 + beta * (
                        -1.821255978 + beta * 1.330274429
                    )
                )
            )
        ) + 1.011
        
        if alpha < 0.0:
            phi = 1.0 - phi

        weight = phi / SQRT_PIx2
        E += weight
        coefficients[i] = weight

    return coefficients, E

# Phi Smoother function
def phi_smoother(source, coefficients, E, hf_ratio=0.5):
    sma2 = source * min(max(0.5, hf_ratio), 1) + np.roll(source, 1) * max(min(0.5, 1 - hf_ratio), 0)
    length = len(coefficients)
    
    if length > 1:
        W = 0.0
        for i in range(length):
            weight = coefficients[i]
            W += weight * sma2[i]
        return W / E if E != 0 else np.nan
    else:
        return source

# RMA Calculation
def rma(source, length):
    alpha = 1 / length
    smoothed = np.zeros(len(source))
    
    # Initialize the first value with the average of the first 'length' values
    smoothed[length - 1] = np.mean(source[:length])
    
    # Compute the RMA
    for i in range(length, len(source)):
        smoothed[i] = alpha * source[i] + (1 - alpha) * smoothed[i - 1]
    
    return smoothed

# RSI Calculation
def rsi(source, length):
    # Calculate price changes
    change = np.diff(source, prepend=source[0])
    
    # Calculate gains and losses
    gain = np.maximum(change, 0)
    loss = np.abs(np.minimum(change, 0))
    
    # Calculate average gain and average loss
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    
    # Calculate RSI
    rs = avg_gain / avg_loss
    rs[avg_loss == 0] = np.nan  # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Reverse RSI Calculation
def reverse_rsi(source, length, value):
    src = (source - rma(source, length)) / (length / 2)
    alpha = 1 / length
    average_up_count = np.zeros(len(source))
    average_down_count = np.zeros(len(source))

    for i in range(1, len(source)):
        if src[i] > src[i - 1]:
            average_up_count[i] = alpha * (src[i] - src[i - 1]) + (1 - alpha) * average_up_count[i - 1]
        else:
            average_up_count[i] = (1 - alpha) * average_up_count[i - 1]

        if src[i] > src[i - 1]:
            average_down_count[i] = (1 - alpha) * average_down_count[i - 1]
        else:
            average_down_count[i] = alpha * (src[i - 1] - src[i]) + (1 - alpha) * average_down_count[i - 1]

    reversed_value = (length - 1) * (average_down_count * value / (100 - value) - average_up_count)
    reverse_rsi = src + reversed_value if reversed_value >= 0 else src + reversed_value * (100 - value) / value

    return reverse_rsi

def calculate_indicators(df):
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['SMA20'] = df['volume'].rolling(window=20).mean()
    df['RSI'] = rsi(df['close'].values, 14)  # 使用自定义 RSI 计算方法
    return df

def calculate_and_analyze(df):
    analysis = []

    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
    df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()

    # Calculate Reverse RSI
    df['Reverse_RSI'] = reverse_rsi(df['close'].values, 14, 50)  # 使用自定义 Reverse RSI 计算方法

    latest_row = df.iloc[-1]

    # Check for overbought conditions
    overbought = (
        latest_row['close'] > latest_row['bb_upper'] and 
        latest_row['RSI'] > 70 and 
        latest_row['Reverse_RSI'] > 80
    )

    # Check for oversold conditions
    oversold = (
        latest_row['close'] < latest_row['bb_lower'] and 
        latest_row['RSI'] < 30 and 
        latest_row['Reverse_RSI'] < 20
    )

    if overbought:
        analysis.append('Price above upper Bollinger Band, RSI indicates overbought, and Reverse RSI indicates overbought (bullish),多头排列发散型无效')
    elif oversold:
        analysis.append('Price below lower Bollinger Band, RSI indicates oversold, and Reverse RSI indicates oversold (bearish)，多头排列发散型无效')

    return analysis

def send_telegram_message(message):
    payload = {
        'chat_id': CHAT_ID,
        'text': message
    }
    response = requests.post(TELEGRAM_API_URL, json=payload)
    return response

def main():
    while True:
        for symbol in CURRENCY_IDS:
            df = fetch_data(symbol, interval, limit=100)
            df = calculate_indicators(df)
            analysis = calculate_and_analyze(df)

            for message in analysis:
                send_telegram_message(message)

        time.sleep(300)  # Wait for 5 minutes before the next iteration

if __name__ == "__main__":
    main()
