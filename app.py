import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import time
from flask import Flask
import os

# Initialize the Dash app with server variable
# app = dash.Dash(__name__)
# server = app.server  # This is needed for deployment

# Modify the Dash initialization
server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True
)

def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX)"""
    # Calculate True Range
    df['TR'] = np.maximum(
        np.maximum(
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1))
        ),
        abs(df['low'] - df['close'].shift(1))
    )
    
    # Calculate +DM and -DM
    df['plus_DM'] = np.where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
        np.maximum(df['high'] - df['high'].shift(1), 0),
        0
    )
    df['minus_DM'] = np.where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
        np.maximum(df['low'].shift(1) - df['low'], 0),
        0
    )
    
    # Calculate smoothed TR and DM
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df['plus_DI'] = 100 * (df['plus_DM'].rolling(window=period).mean() / df['ATR'])
    df['minus_DI'] = 100 * (df['minus_DM'].rolling(window=period).mean() / df['ATR'])
    
    # Calculate ADX
    df['DX'] = 100 * abs(df['plus_DI'] - df['minus_DI']) / (df['plus_DI'] + df['minus_DI'])
    df['ADX'] = df['DX'].rolling(window=period).mean()
    
    return df

def calculate_indicators(df):
    """Calculate multiple technical indicators"""
    # Moving Averages for trend
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # Volume Moving Average
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    
    # ADX
    df = calculate_adx(df)
    
    return df

def check_trend(row):
    """Determine trend direction and strength with relaxed conditions"""
    trend = 0
    
    # Reduced ADX threshold to 20 and made EMA crossover optional
    if row['ADX'] > 20:
        # Uptrend conditions - now only requires one condition
        if row['EMA_20'] > row['EMA_50'] or row['plus_DI'] > row['minus_DI']:
            trend = 1
        # Downtrend conditions - now only requires one condition
        elif row['EMA_20'] < row['EMA_50'] or row['plus_DI'] < row['minus_DI']:
            trend = -1
            
    return trend

def generate_complex_signals(df):
    """Generate trading signals with adjusted parameters"""
    signals = pd.Series(index=df.index, data=np.nan)
    
    # Calculate trend for each point
    df['trend'] = df.apply(check_trend, axis=1)
    
    for i in range(2, len(df)):
        # Skip if we don't have enough confirmation candles
        if i < 3:
            continue
            
        # Initialize signal strength
        buy_strength = 0
        sell_strength = 0
        
        # RSI conditions with relaxed thresholds
        if df['RSI'].iloc[i] < 35:  # Relaxed from 30
            buy_strength += 1
        elif df['RSI'].iloc[i] > 65:  # Relaxed from 70
            sell_strength += 1
        
        # MACD conditions
        if df['MACD'].iloc[i] > df['Signal_Line'].iloc[i]:
            buy_strength += 1
        elif df['MACD'].iloc[i] < df['Signal_Line'].iloc[i]:
            sell_strength += 1
        
        # Bollinger Bands
        if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
            buy_strength += 1
        elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
            sell_strength += 1
        
        # Volume confirmation
        if df['volume'].iloc[i] > df['Volume_MA'].iloc[i]:
            buy_strength += 0.5
            sell_strength += 0.5
        
        # Generate signals with relaxed trend requirements
        if df['trend'].iloc[i] >= 0 and buy_strength >= 1.5:  # Reduced from 2
            signals.iloc[i] = 1
        elif df['trend'].iloc[i] <= 0 and sell_strength >= 1.5:  # Reduced from 2
            signals.iloc[i] = -1
    
    return signals

class BinanceDataStream:
    def __init__(self, symbol='ADAUSDT', interval='1h', limit=500):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.base_url = 'https://api.binance.us/api/v3/klines'

    def fetch_data(self):
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': self.limit
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 
                                           'volume', 'close_time', 'quote_volume', 'trades',
                                           'taker_buy_base', 'taker_buy_quote', 'ignore'])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

def create_figure(df, signals):
    """Create the plotly figure with all indicators"""
    fig = make_subplots(rows=5, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       subplot_titles=('Price with EMAs & BB', 'RSI', 'MACD', 'ADX', 'Volume'),
                       row_heights=[0.4, 0.15, 0.15, 0.15, 0.15])

    # Price, EMAs and Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price',
                            line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20',
                            line=dict(color='orange', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], name='EMA 50',
                            line=dict(color='purple', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                            line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                            line=dict(color='gray', dash='dash')), row=1, col=1)

    # Add buy signals
    buy_signals = signals[signals == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(x=buy_signals.index, y=df.loc[buy_signals.index, 'close'],
                      mode='markers', name='Buy Signal',
                      marker=dict(symbol='triangle-up', size=15, color='green')),
            row=1, col=1)

    # Add sell signals
    sell_signals = signals[signals == -1]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(x=sell_signals.index, y=df.loc[sell_signals.index, 'close'],
                      mode='markers', name='Sell Signal',
                      marker=dict(symbol='triangle-down', size=15, color='red')),
            row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                            line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red",
                  row=2, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green",
                  row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                            line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal Line',
                            line=dict(color='orange')), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='MACD Histogram',
                        marker_color='gray'), row=3, col=1)

    # ADX
    fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], name='ADX',
                            line=dict(color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['plus_DI'], name='+DI',
                            line=dict(color='green')), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['minus_DI'], name='-DI',
                            line=dict(color='red')), row=4, col=1)
    fig.add_hline(y=25, line_width=1, line_dash="dash", line_color="gray",
                  row=4, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume',
                        marker_color='lightblue'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Volume_MA'], name='Volume MA',
                            line=dict(color='darkblue')), row=5, col=1)

    # Update layout
    fig.update_layout(
        title=f'Real-time Crypto Analysis - Last Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        showlegend=True,
        height=1400,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    return fig

# Create the layout
app.layout = html.Div([
    html.H1("Real-time Crypto Analysis", style={'textAlign': 'center'}),
    
    # Controls div
    html.Div([
        html.Label('Update Interval (seconds): '),
        dcc.Input(
            id='update-interval',
            type='number',
            value=5,
            min=1,
            max=60
        ),
        html.Label(' Symbol: '),
        dcc.Input(
            id='symbol-input',
            type='text',
            value='ADAUSDT'
        ),
    ], style={'margin': '20px', 'textAlign': 'center'}),
    
    # Add interval component for updates
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds (5 seconds)
        n_intervals=0
    ),
    
    # Add the graph
    dcc.Graph(id='live-graph', style={'height': '1400px'}),
])

@app.callback(
    Output('interval-component', 'interval'),
    Input('update-interval', 'value')
)
def update_interval(value):
    return value * 1000  # Convert to milliseconds

@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('symbol-input', 'value')]
)
def update_graph(n, symbol):
    # Create data stream instance
    stream = BinanceDataStream(symbol=symbol)
    
    # Fetch new data
    df = stream.fetch_data()
    
    if df is None:
        return go.Figure()  # Return empty figure if data fetch fails
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Generate signals
    signals = generate_complex_signals(df)
    
    # Create and return the figure
    return create_figure(df, signals)

def main():
    # Run the server
    app.run_server(debug=True, port=8050)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)

