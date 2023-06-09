# Skynet is a program that takes user input [stock ticker(s)], and outputs an excel spreadsheet with the following information:
# 1: Price data, Volume Periodic Return, MACD, RSI, A/D Slope.
# 2: Beta, Price/Book Ratio, Market Cap, Frequency that ticker outperforms market, Frequency that ticker outperforms its industry, Regression prediction for next period.
# 3: Market's expectation of the stock's future prices (based off a volume-weighted expectation of option contracts for the stock).
# Skynet will then ask the user to input a path where he/she/they would like the excel sheet saved.

# Imported Packages
import pandas as pd
import yfinance as yf
import numpy as np
import pandas_ta as ta
from sklearn.linear_model import LinearRegression as LR
from bs4 import BeautifulSoup as bs
from fake_useragent import FakeUserAgent
import requests


# Using user input to fetch price data + Daily t_returns
stock=input("Stock Ticker: ")
ticker_caps=stock.upper()
tickers=list(ticker_caps.split())
prices=dict()
period="1y"
interval="1d"

for t in tickers:
    try:
        ask = yf.download(t, period=period, interval=interval)
        print(t, ": Stock data downloaded")
        ask["Returns"] = (ask["Adj Close"] - ask["Open"]) / ask["Open"]
        prices[t] = ask
    except:
        print(t,": Unable to download stock data")
        continue

company_name = dict()
for t in tickers:
    try:
        url = 'https://finance.yahoo.com/quote/' + t + '?p=' + t + '&.tsrc=fin-srch'
        agent = FakeUserAgent()
        rand = agent.chrome
        r = requests.get(url, headers={'User-Agent': rand})
        soup = bs(r.text, 'html.parser')
        body = soup.find_all('body')
        for rows in body:
            hs = rows.find_all('h1', class_='D(ib) Fz(18px)')
            for h in hs:
                name_stuff = list()
                val = h.string
                val = val.lower()
                name_stuff.append(val.split())
                company_name[t] = name_stuff[0]
    except:
        continue


#Creating MACD
MACD=dict()
for t in tickers:
    try:
        refer = prices[t]
        macd = ta.macd(refer['Close'])
        macd = macd.dropna(axis=0)
        MACD[t] = macd["MACDh_12_26_9"]
        prices[t]["MACD"] = MACD[t]
        print(t, ": MACD Calculated")
    except:
        continue


#Beta for input-stocks
spy=yf.download("SPY",period=period,interval=interval)
spy["Returns"]=(spy["Adj Close"]-spy["Open"])/spy["Open"]
beta=dict()
for t in tickers:
    try:
        y=yf.Ticker(t)
        beta_t=y.info['beta']
        beta[t]=beta_t
    except:
        print(t,": Beta not found")
        beta[t]="Not Found"
        continue

    print(t,": Beta downloaded")

#Creating Money-Flow Multiplier:
for t in tickers:
    refer=prices[t]
    CML = refer['Close']-refer['Low']
    HMC = refer['High']-refer['Close']
    range = refer['High']-refer['Low']
    mfm = (CML-HMC)/range
    prices[t]['MFM']= mfm
    print(t,": MFM Calculated")

#Volume Percentage change:
volume_ch=dict()
for t in tickers:
    refer=prices[t]
    volch=refer['Volume'][1:].pct_change()
    volume_ch[t]=volch

#Accumulation Distribution Line Movement:
for t in tickers:
    try:
        refer = prices[t]
        AD = ta.ad(refer['High'], refer['Low'], refer['Close'], refer['Volume'])
        prices[t]['A/D'] = AD
        print(t, ": A/D Movement Calculated")
        temp = refer['A/D'][1:]
        k = 0
        AD_change = list()
        for ad in refer['A/D']:
            try:
                change = (temp[k] - refer['A/D'][k]) / refer['A/D'][k]
                AD_change.append(change)
                k = k + 1
            except:
                AD_change.insert(0, " ")
        prices[t]['A/D Movement'] = AD_change
        print(t,": A/D Calculated")
    except:
        continue

#Market Cap
mktcap=dict()
for t in tickers:
    try:
        tickmkt=yf.Ticker(t)
        infomkt=tickmkt.info
        mcap=infomkt['marketCap']
        mktcap[t]=mcap
    except:
        print(t,": Market Cap not found")
        mktcap[t]="Not Found"
        continue

    print(t,": Market Cap Downloaded")

#RSI
for t in tickers:
    refer=prices[t]
    close=refer['Close']
    rsi_t=ta.rsi(close)
    prices[t]['RSI']=rsi_t

    print(t,": RSI Calculated")


#P/B Ratio
pb=dict()
for t in tickers:
    try:
        tickpb=yf.Ticker(t)
        infopb=tickpb.info
        pbratio=infopb['priceToBook']
        pb[t]=pbratio
    except:
        print(t,": P/B Not Found")
        pb[t]="Not Found"
        continue

    print(t,": Price/Book Ratio Downloaded")

#Average Daily Return
avg_daily_ret=dict()
for t in tickers:
    refer=prices[t]['Returns']
    mean=np.mean(refer)
    avg_daily_ret[t]=mean

    print(t,": Average Daily Return Calculated")

#Standard Deviation - Returns:
stdev=dict()
for t in tickers:
    refer=prices[t]["Returns"]
    std=np.std(refer)
    stdev[t]=std

    print(t,": Standard Deviation Calculated")


#Regression (Autocorrelate,SPY,RSI,AD Movement,Volume):
regression_prediction=dict()
for t in tickers:
    refer = prices[t]
    try:
        rsi_count = len(refer['RSI'].dropna())
        y = refer['Returns'].dropna()
        y = y.drop(index=y.index[:-rsi_count])
        y = y[1:]
    except:
        continue

    x_list=list()
    x_names=list()
    x_items=list()

    try:
        x1 = refer["Returns"]
        x1 = x1.dropna(axis=0)
        x1 = x1.drop(index=x1.index[:-rsi_count])
        x1 = x1.drop(index=x1.index[-1])
        x1_corr = np.corrcoef(x1.astype(float), y)
        x1_corr = x1_corr[0,1]
        if x1_corr>=0.2:
            x_list.append(x1)
            x_names.append("St-Ret")
            x_items.append('x1')
    except:
        pass

    try:
        x2 = spy['Returns']
        x2 = x2.dropna(axis=0)
        x2 = x2.drop(index=x2.index[:-rsi_count])
        x2 = x2.drop(index=x2.index[-1])
        x2_corr = np.corrcoef(x2.astype(float), y)
        x2_corr = x2_corr[0,1]
        if x2_corr >= 0.2:
            x_list.append(x2)
            x_names.append("SPY-Ret")
            x_items.append('x2')
    except:
        pass

    try:
        x3 = refer['RSI'].dropna()
        x3 = x3.drop(index=x3.index[-1])
        x3_corr = np.corrcoef(x3.astype(float), y)
        x3_corr = x3_corr[0,1]
        if x3_corr >= 0.2:
            x_list.append(x3)
            x_names.append("St-RSI")
            x_items.append('x3')
    except:
        pass

    try:
        x4 = refer['A/D Movement'].dropna()
        x4 = x4.drop(index=x4.index[:-rsi_count])
        x4 = x4.drop(index=x4.index[-1])
        x4_corr = np.corrcoef(x4.astype(float), y)
        x4_corr = x4_corr[0,1]
        if x4_corr >= 0.2:
            x_list.append(x4)
            x_names.append("St-ADMovement")
            x_items.append('x4')
    except:
        pass

    try:
        x5 = volume_ch[t].dropna()
        x5 = x5.drop(index=x5.index[:-rsi_count])
        x5 = x5.drop(index=x5.index[-1])
        x5_corr = np.corrcoef(x5.astype(float),y)
        x5_corr = x5_corr[0,1]
        if x5_corr >= 0.2:
            x_list.append(x5)
            x_names.append("Vol_Change")
            x_items.append('x5')
    except:
        pass

    try:

        if len(x_list)>0:
            x=pd.DataFrame(x_list,index=[x_names])
            x = x.transpose()

            reg=LR()
            reg.fit(x.values,y)

            variables=list()

            c1=refer['Returns'][-1]
            if 'x1' in x_items:
                variables.append(c1)

            c2=spy['Returns'][-1]
            if 'x2' in x_items:
                variables.append(c2)

            c3=refer['RSI'][-1]
            if 'x3' in x_items:
                variables.append(c3)

            c4=refer['A/D Movement'][-1]
            if 'x4' in x_items:
                variables.append(c4)

            c5=volume_ch[t][-1]
            if 'x5' in x_items:
                variables.append(c5)
    except:
        pass

        try:
            point=np.array(variables)
            num=len(x_list)
            point = point.reshape(1, num)
            pred = reg.predict(point)
            regression_prediction[t]=float(pred)
        except:
            regression_prediction[t]="No Correlation Found"

    else:
        regression_prediction[t]="No Correlation Found"

    print(t,": Regression Completed")

#Pricing using Options (market expectations):
price_expectation=dict()
for t in tickers:
    try:
        price_exp_by_date = dict()
        url = 'https://finance.yahoo.com/quote/' + t + '/options?p=' + t
        user = FakeUserAgent()
        rand1 = user.random
        date_list = list()
        date_dict = dict()
        r = requests.get(url, headers={'User-Agent': rand1})
        soup = bs(r.text, 'html.parser')
        body = soup.find_all('body')
        for row in body:
            divs = row.find_all('div', class_='Fl(start) Pend(18px)')
            for div in divs:
                select = div.find('select')
                for sel in select:
                    date_list.append(sel['value'])
                    date_dict[sel['value']] = sel.string

        for date in date_list:
            url1 = 'https://finance.yahoo.com/quote/' + t + '/options?p=' + t + '&date=' + date
            rand2 = user.chrome
            r = requests.get(url1, headers={'User-Agent': rand2})
            opt = pd.read_html(r.text)
            callopt = opt[0]
            putopt = opt[1]

            for x in callopt['Volume']:
                callopt['Volume'] = callopt['Volume'].replace('-', '0')
                callopt["Strike"] = callopt['Strike'].replace('-', '0')
                callopt['Ask'] = callopt['Ask'].replace('-', '0')
            callopt['Volume'] = callopt["Volume"].astype(int)
            callopt['Strike'] = callopt['Strike'].astype(float)
            callopt['Ask'] = callopt["Ask"].astype(float)
            for x in putopt["Volume"]:
                putopt["Volume"] = putopt["Volume"].replace('-', '0')
                putopt['Strike'] = putopt['Strike'].replace('-', '0')
                putopt['Ask'] = putopt['Ask'].replace('-', '0')
            putopt["Volume"] = putopt['Volume'].astype(int)
            putopt['Strike'] = putopt['Strike'].astype(float)
            putopt['Ask'] = putopt['Ask'].astype(float)

            vol = sum(callopt["Volume"]) + sum(putopt["Volume"])

            callopt['vol weight'] = callopt["Volume"] / vol
            callopt['exp'] = callopt['Strike'] + callopt['Ask']

            putopt['vol weight'] = putopt['Volume'] / vol
            putopt['exp'] = putopt['Strike'] - putopt['Ask']

            callopt['VWE'] = callopt['exp'] * callopt['vol weight']
            putopt['VWE'] = putopt['exp'] * putopt['vol weight']

            vol_weighted_prediction = sum(callopt['VWE']) + sum(putopt['VWE'])
            current_date = date_dict[date]
            price_exp_by_date[current_date] = vol_weighted_prediction
            price_expectation[t] = price_exp_by_date
        print(t, ": Created future prices based on market expectations")

    except:
        print(t,": Unable to use options to find market expectations")
        continue

mkt_expectation = pd.DataFrame.from_dict(price_expectation, orient='index')
mkt_expectation=mkt_expectation.transpose()


#Chances of Outperforming SPY
outperform_mkt=dict()
for t in tickers:
    count=0
    refer=prices[t]
    t_returns=refer["Returns"]
    df=pd.DataFrame([t_returns,spy["Returns"]],index=["Security","S&P"])
    df=df.transpose()
    df["performance"]=df["Security"]-df["S&P"]
    for day in df["performance"]:
        if day>0:
            count=count+1
    outperform_mkt[t]= count / len(t_returns)

    print(t,": Market-Outperforming Ratio Calculcated")

# Chances of Outperforming Industry:
Industries = ['Utilities',
              'Industrials',
              'Healthcare',
              'Technology',
              'Communication Services',
              'Consumer Cyclical',
              'Consumer Defensive',
              'Energy',
              'Financial Services',
              'Basic Materials',
              'Real Estate']
ETFS = {'Utilities': 'IDU',
        'Industrials': 'IYJ',
        'Healthcare': 'IYH',
        'Technology': 'IYW',
        'Communication Services': 'IYZ',
        'Consumer Cyclical': 'IYC',
        'Consumer Defensive': 'IYK',
        'Energy': 'IYE',
        'Financial Services': 'IYF',
        'Basic Materials': 'IYM',
        'Real Estate': 'IYR'}

outperform_ind=dict()
for t in tickers:
    refer = yf.Ticker(t)
    info = refer.info
    try:
        sec = info['sector']
        for ind in Industries:
            if sec == ind:
                etf = ETFS[ind]
                security = prices[t]
                current = yf.download(etf, period=period, interval=interval)
                current["Returns"] = (current['Adj Close'] - current['Open']) / current['Open']
                df = pd.DataFrame([security['Returns'], current["Returns"]], index=['Stock', 'Industry'])
                df = df.transpose()
                count = 0
                for row in df.index:
                    if df["Stock"][row] > df['Industry'][row]:
                        count = count + 1
                outperf_freq = count / len(df["Stock"])
                outperform_ind[t] = outperf_freq
                print(t, ": Industry Outperformance Ratio Calculated")
    except:
        print(t,": Unable to compare returns to industry returns")
        outperform_ind[t]='Sector not found'


#Information Data
stats=pd.DataFrame(index=tickers)
stats["Beta"]=stats.from_dict(beta,orient='index')
stats["MktCap"]=stats.from_dict(mktcap,orient='index')
stats["Price/Book"]=stats.from_dict(pb,orient='index')
stats["Average Daily Return"]=stats.from_dict(avg_daily_ret,orient='index')
stats["Standard Deviation of Daily Returns"]=stats.from_dict(stdev,orient='index')
stats["Next-Period Prediction (From Regression)"]=stats.from_dict(regression_prediction,orient='index')
stats["Market-Outperformance Ratio"]=stats.from_dict(outperform_mkt, orient='index')
stats["Industry-Outperformance Ratio"]=stats.from_dict(outperform_ind,orient='index')

#Single DataFrame
price_data=pd.DataFrame()
for t in prices:
    current=pd.DataFrame.from_dict(prices[t])
    for col in current.columns:
        price_data[t]= " "
        price_data.insert(loc=len(price_data.columns), column=col, value=current[col], allow_duplicates=True)
price_data=price_data.iloc[::-1]

#Excel
with pd.ExcelWriter('/Users/rohanbanerjea/Desktop/Stocks/stocksfrompy.xlsx') as writer:
    price_data.to_excel(writer, sheet_name='Price Data')
    stats.to_excel(writer,sheet_name='Statistics')
    mkt_expectation.to_excel(writer,sheet_name='Market Expectation of Prices')





