import requests
import time

# Constants
API_SECRET = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVdJ9'
BOT_TOKEN = '7087052045:AAF3eJLHSvBGKtqqa2l_e7su_ESiteL84i8'
CHAT_ID = '1640026631'
INTERVAL = '1h'
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT']

BASE_URL = 'https://api.taapi.io'

def get_taapi_indicator(indicator, symbol, period=60):
    """ 获取 TAAPI 指标数据 """
    url = f"{BASE_URL}/{indicator}"
    params = {
        'secret': API_SECRET,
        'symbol': symbol,
        'interval': INTERVAL,
        'optInTimePeriod': period
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data.get('value', None)

def send_telegram_message(message):
    """ 发送 Telegram 消息 """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {
        'chat_id': CHAT_ID,
        'text': message
    }
    response = requests.get(url, params=params)
    return response.json()

def calculate_posterior_up(ema, sma, dema, vwma, ema_fast, sma_fast, dema_fast, vwma_fast, gap_length, gap):
    """ 计算上涨的后验概率 """
    def sig(src, gap):
        result = [1 if src[i] >= src[i-gap] else 0 for i in range(gap, len(src))]
        return sum(result) / len(result) if result else 0
    
    ema_trend = sig(ema, gap)
    sma_trend = sig(sma, gap)
    dema_trend = sig(dema, gap)
    vwma_trend = sig(vwma, gap)

    ema_trend_fast = sig(ema_fast, gap)
    sma_trend_fast = sig(sma_fast, gap)
    dema_trend_fast = sig(dema_fast, gap)
    vwma_trend_fast = sig(vwma_fast, gap)
    
    prior_up = (ema_trend + sma_trend + dema_trend + vwma_trend) / 4
    prior_down = 1 - prior_up

    likelihood_up = (ema_trend_fast + sma_trend_fast + dema_trend_fast + vwma_trend_fast) / 4
    likelihood_down = 1 - likelihood_up

    posterior_up = prior_up * likelihood_up / (prior_up * likelihood_up + prior_down * likelihood_down)
    
    if posterior_up is None:
        posterior_up = 0
    
    return posterior_up

def check_and_notify():
    """ 检查指标并发送通知 """
    for symbol in SYMBOLS:
        ema = get_taapi_indicator('ema', symbol)
        time.sleep(15)  # 等待 15 秒
        sma = get_taapi_indicator('sma', symbol)
        time.sleep(15)
        dema = get_taapi_indicator('dema', symbol)
        time.sleep(15)
        vwma = get_taapi_indicator('vwma', symbol)
        time.sleep(15)

        # 快速 MA 指标
        ema_fast = get_taapi_indicator('ema', symbol, period=60-20)
        time.sleep(15)
        sma_fast = get_taapi_indicator('sma', symbol, period=60-20)
        time.sleep(15)
        dema_fast = get_taapi_indicator('dema', symbol, period=60-20)
        time.sleep(15)
        vwma_fast = get_taapi_indicator('vwma', symbol, period=60-20)
        time.sleep(15)

        if None in [ema, sma, dema, vwma, ema_fast, sma_fast, dema_fast, vwma_fast]:
            continue  # 如果任何指标为 None，则跳过

        # 计算后验概率
        posterior_up = calculate_posterior_up(ema, sma, dema, vwma, ema_fast, sma_fast, dema_fast, vwma_fast, gap_length=20, gap=10)
        
        # 定义红色和绿色方块条件
        green_condition = posterior_up > 0.52
        red_condition = posterior_up < 0.48

        if green_condition:
            message = (f"Symbol: {symbol}\n"
                       f"EMA: {ema}\n"
                       f"SMA: {sma}\n"
                       f"DEMA: {dema}\n"
                       f"VWMA: {vwma}\n"
                       f"Probability of Up Trend: {posterior_up*100:.2f}%\n"
                       f"Condition: Green Block")
            send_telegram_message(message)
        elif red_condition:
            message = (f"Symbol: {symbol}\n"
                       f"EMA: {ema}\n"
                       f"SMA: {sma}\n"
                       f"DEMA: {dema}\n"
                       f"VWMA: {vwma}\n"
                       f"Probability of Up Trend: {posterior_up*100:.2f}%\n"
                       f"Condition: Red Block")
            send_telegram_message(message)

def main():
    """ 主函数，每 5 分钟运行一次检查 """
    while True:
        check_and_notify()
        time.sleep(300)  # 每 5 分钟运行一次

if __name__ == '__main__':
    main()
