import numpy as np
import pandas as pd
import requests
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import warnings
warnings.filterwarnings('ignore')

class BinanceDataStream:
    def __init__(self, symbol='VANAUSDT', interval='1m', limit=100):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.base_url = 'https://api.binance.com/api/v3/klines'

    def fetch_data(self):
        """Fetch historical data from Binance with error handling"""
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': self.limit
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"No data available for {self.symbol}")
                return None
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            df.set_index('timestamp', inplace=True)
            
            print(f"Fetched {len(df)} data points for {self.symbol}")
            return df
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return None

class MarketStructureAnalyzer:
    def __init__(self):
        self.support_levels = []
        self.resistance_levels = []
        self.timeframes = ['1h', '4h', '1d']  # Multiple timeframes
        
    def identify_swing_points(self, df, window=5):
        """Identify swing highs and lows with data validation"""
        df = df.copy()
        
        # Ensure minimum window size and data length
        min_window = 3
        window = max(min_window, min(window, len(df) // 2))
        
        if len(df) < window:
            print(f"Warning: Not enough data points ({len(df)}) for swing point calculation. Minimum required: {window}")
            # Set default values
            df['swing_high'] = False
            df['swing_low'] = False
            return df
        
        try:
            # Identify swing highs and lows - ensure boolean output
            df['swing_high'] = df['high'].rolling(window=window, center=True).apply(
                lambda x: x[window//2] == max(x) if len(x) > window//2 else False, 
                raw=True
            ).astype(bool)
            
            df['swing_low'] = df['low'].rolling(window=window, center=True).apply(
                lambda x: x[window//2] == min(x) if len(x) > window//2 else False, 
                raw=True
            ).astype(bool)
        except Exception as e:
            print(f"Warning: Error calculating swing points: {e}")
            # Fallback to simple high/low detection
            df['swing_high'] = df['high'] > df['high'].shift(1)
            df['swing_low'] = df['low'] < df['low'].shift(1)
        
        return df
    
    def analyze_market_structure(self, df, lookback=20):
        """Analyze market structure including trends and key levels"""
        df = df.copy()
        
        # Identify swing points
        df = self.identify_swing_points(df)
        
        # Analyze trend structure - ensure proper type conversion
        df['higher_high'] = (
            df['swing_high'].astype(bool) & 
            (df['high'] > df['high'].rolling(lookback).max().shift(1))
        )
        
        df['lower_low'] = (
            df['swing_low'].astype(bool) & 
            (df['low'] < df['low'].rolling(lookback).min().shift(1))
        )
        
        # Determine trend based on moving averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
        
        # Ensure boolean output for trend conditions
        df['uptrend'] = (
            (df['sma_20'] > df['sma_50']) & 
            (df['sma_50'] > df['sma_200'])
        ).astype(bool)
        
        df['downtrend'] = (
            (df['sma_20'] < df['sma_50']) & 
            (df['sma_50'] < df['sma_200'])
        ).astype(bool)
        
        # Identify support and resistance levels
        self.identify_key_levels(df)
        
        return df
    
    def identify_key_levels(self, df, threshold=0.02):
        """Identify potential support and resistance levels"""
        # Convert to boolean before extracting values
        swing_highs = df[df['swing_high'].astype(bool)]['high'].values
        swing_lows = df[df['swing_low'].astype(bool)]['low'].values
        
        self.support_levels = []
        self.resistance_levels = []
        
        # Group similar price levels
        for high in swing_highs:
            if not any(abs(level - high)/high < threshold for level in self.resistance_levels):
                self.resistance_levels.append(high)
        
        for low in swing_lows:
            if not any(abs(level - low)/low < threshold for level in self.support_levels):
                self.support_levels.append(low)
        
        # Sort levels
        self.support_levels.sort()
        self.resistance_levels.sort()

    def determine_market_phase(self, df, index, lookback=50):
        """Determine the current market phase"""
        # Get recent price action
        recent_price = df.iloc[index-lookback:index+1]
        
        # Calculate key metrics
        price_trend = (df['ema_20'].iloc[index] - df['ema_20'].iloc[index-lookback]) / df['ema_20'].iloc[index-lookback]
        volume_trend = (df['volume'].iloc[index-10:index].mean() - df['volume'].iloc[index-lookback:-10].mean()) / df['volume'].iloc[index-lookback:-10].mean()
        volatility = df['atr'].iloc[index] / df['atr'].iloc[index-lookback:index].mean()
        
        # Determine phase
        if price_trend > 0.02 and volume_trend > 0:
            return 'markup'
        elif price_trend < -0.02 and volume_trend > 0:
            return 'markdown'
        elif volatility < 0.8 and abs(price_trend) < 0.01:
            if volume_trend > 0:
                return 'accumulation'
            else:
                return 'distribution'
        return 'ranging'

    def check_higher_timeframe_alignment(self, df, index, signal_type='buy'):
        """Check if higher timeframes align with the trade direction"""
        # Calculate EMAs for multiple timeframes
        alignments = []
        
        for tf in ['1h', '4h', '1d']:
            if tf == '1h':
                tf_data = df
            else:
                # Resample to higher timeframe
                tf_data = df.resample(tf).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
            
            # Calculate EMAs for this timeframe
            tf_data['ema20'] = ta.trend.ema_indicator(tf_data['close'], window=20)
            tf_data['ema50'] = ta.trend.ema_indicator(tf_data['close'], window=50)
            
            # Check alignment
            if signal_type == 'buy':
                alignments.append(
                    tf_data['ema20'].iloc[-1] > tf_data['ema50'].iloc[-1] and
                    tf_data['close'].iloc[-1] > tf_data['ema20'].iloc[-1]
                )
            else:
                alignments.append(
                    tf_data['ema20'].iloc[-1] < tf_data['ema50'].iloc[-1] and
                    tf_data['close'].iloc[-1] < tf_data['ema20'].iloc[-1]
                )
        
        # Return True if at least 2 timeframes align
        return sum(alignments) >= 2

    def is_forming_higher_low(self, df, index, lookback=20):
        """Check if price is forming a higher low"""
        recent_lows = df['low'].iloc[index-lookback:index].nsmallest(3)
        return all(recent_lows.iloc[i] > recent_lows.iloc[i+1] for i in range(len(recent_lows)-1))

    def is_breaking_resistance(self, df, index, lookback=20):
        """Check if price is breaking resistance"""
        recent_high = df['high'].iloc[index-lookback:index].max()
        return (df['close'].iloc[index] > recent_high and
                df['volume'].iloc[index] > df['volume'].iloc[index-lookback:index].mean() * 1.5)

    def get_volatility_state(self, df, index, lookback=20):
        """Determine current volatility state"""
        current_atr = df['atr'].iloc[index]
        avg_atr = df['atr'].iloc[index-lookback:index].mean()
        atr_std = df['atr'].iloc[index-lookback:index].std()
        
        if current_atr > avg_atr + 2 * atr_std:
            return 'high'
        elif current_atr < avg_atr - atr_std:
            return 'low'
        return 'normal'

    def calculate_trend_strength(self, df, index, lookback=20):
        """Calculate trend strength using multiple indicators"""
        # ADX for trend strength
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx().iloc[index]
        
        # Price momentum
        momentum = (df['close'].iloc[index] - df['close'].iloc[index-lookback]) / df['close'].iloc[index-lookback]
        
        # EMA alignment
        ema_alignment = abs(df['ema_20'].iloc[index] - df['ema_50'].iloc[index]) / df['ema_50'].iloc[index]
        
        # Combine metrics
        trend_strength = (0.4 * adx/25 + 0.3 * abs(momentum) + 0.3 * ema_alignment)
        return trend_strength

    def analyze_market_context(self, df, index):
        """Analyze overall market context"""
        # Check if near key levels
        near_support = self.is_near_support(df, index)
        near_resistance = self.is_near_resistance(df, index)
        
        # Check market volatility state
        volatility_state = self.get_volatility_state(df, index)
        
        # Check trend strength
        trend_strength = self.calculate_trend_strength(df, index)
        
        # Check market phase
        market_phase = self.determine_market_phase(df, index)
        
        # Check higher timeframe alignment
        htf_aligned = self.check_higher_timeframe_alignment(df, index)
        
        return {
            'near_support': near_support,
            'near_resistance': near_resistance,
            'volatility_state': volatility_state,
            'trend_strength': trend_strength,
            'market_phase': market_phase,
            'htf_aligned': htf_aligned
        }

    def is_near_support(self, df, index, threshold=0.01):
        """Check if price is near support level"""
        current_price = df['close'].iloc[index]
        
        # Check dynamic support (recent lows)
        recent_low = df['low'].iloc[index-20:index].min()
        if abs(current_price - recent_low) / current_price < threshold:
            return True
        
        # Check static support levels
        for level in self.support_levels:
            if abs(current_price - level) / current_price < threshold:
                return True
        
        return False

    def is_near_resistance(self, df, index, threshold=0.01):
        """Check if price is near resistance level"""
        current_price = df['close'].iloc[index]
        
        # Check dynamic resistance (recent highs)
        recent_high = df['high'].iloc[index-20:index].max()
        if abs(current_price - recent_high) / current_price < threshold:
            return True
        
        # Check static resistance levels
        for level in self.resistance_levels:
            if abs(current_price - level) / current_price < threshold:
                return True
        
        return False

class VolumeAnalyzer:
    def __init__(self):
        self.vwap_levels = []
        
    def analyze_volume_pattern(self, df, index, lookback=20):
        """Enhanced volume pattern analysis"""
        # Check for volume climax
        volume_climax = df['volume'].iloc[index] > df['volume'].iloc[index-lookback:index].max()
        
        # Check for volume expansion
        avg_volume = df['volume'].iloc[index-lookback:index].mean()
        volume_expansion = df['volume'].iloc[index] > avg_volume * 1.5
        
        # Check volume trend
        volume_trend = df['volume'].iloc[index-5:index].mean() > avg_volume
        
        # Volume price correlation
        price_change = df['close'].iloc[index] - df['close'].iloc[index-1]
        volume_change = df['volume'].iloc[index] - df['volume'].iloc[index-1]
        volume_price_aligned = (price_change > 0 and volume_change > 0) or (price_change < 0 and volume_change > 0)
        
        return {
            'climax': volume_climax,
            'expansion': volume_expansion,
            'trend': volume_trend,
            'price_aligned': volume_price_aligned
        }
        
    def analyze_volume_profile(self, df):
        """Enhanced volume analysis with order flow insights"""
        df = df.copy()
        
        # Basic volume metrics
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_std'] = df['volume'].rolling(20).std()
        df['significant_volume'] = df['volume'] > (df['volume_ma'] + df['volume_std'])
        
        # Volume-weighted metrics
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['volume_force'] = (df['close'] - df['close'].shift(1)) * df['volume']
        
        # Enhanced volume analysis
        df['volume_trend'] = df['volume'].rolling(window=10).mean() / df['volume'].rolling(window=30).mean()
        df['volume_momentum'] = df['volume'].pct_change(3)
        
        # Order flow analysis
        df['buying_pressure'] = np.where(
            df['close'] > df['open'],
            df['volume'] * (df['close'] - df['low']) / (df['high'] - df['low']),
            df['volume'] * (df['close'] - df['high']) / (df['low'] - df['high'])
        )
        
        df['selling_pressure'] = np.where(
            df['close'] < df['open'],
            df['volume'] * (df['high'] - df['close']) / (df['high'] - df['low']),
            df['volume'] * (df['low'] - df['close']) / (df['low'] - df['high'])
        )
        
        # Volume zones
        df['volume_zone'] = pd.qcut(df['volume'], q=5, labels=['very_low', 'low', 'normal', 'high', 'very_high'])
        
        # Calculate cumulative delta
        df['delta'] = df['buying_pressure'] - df['selling_pressure']
        df['cumulative_delta'] = df['delta'].cumsum()
        
        # Volume profile analysis
        self.calculate_volume_profile(df)
        
        return df
    
    def calculate_volume_profile(self, df, num_bins=10):
        """Enhanced volume profile analysis"""
        # Calculate price levels
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / num_bins
        
        df['price_bin'] = ((df['close'] - df['low'].min()) // bin_size) * bin_size + df['low'].min()
        
        # Volume Profile
        volume_profile = df.groupby('price_bin')['volume'].sum()
        
        # Point of Control (POC)
        self.poc_level = volume_profile.idxmax()
        
        # Value Area
        total_volume = volume_profile.sum()
        volume_threshold = total_volume * 0.7  # 70% of total volume
        
        sorted_profile = volume_profile.sort_values(ascending=False)
        cumsum_volume = sorted_profile.cumsum()
        value_area_profile = sorted_profile[cumsum_volume <= volume_threshold]
        
        self.value_area_high = value_area_profile.index.max()
        self.value_area_low = value_area_profile.index.min()
        
        # Store VWAP levels
        self.vwap_levels = volume_profile.sort_values(ascending=False).index[:3].values
        
        return volume_profile

class RiskManager:
    def __init__(self, account_balance=10000, risk_per_trade=1.0, max_drawdown=20.0):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade  # percentage
        self.max_drawdown = max_drawdown  # maximum allowed drawdown percentage
        self.open_positions = []
        self.equity_curve = [account_balance]
        self.max_equity = account_balance
        self.current_drawdown = 0
        
    def calculate_position_size(self, df, index, entry_price, stop_loss):
        """Enhanced position size calculation with volatility adjustment"""
        if stop_loss == entry_price:
            return 0
            
        # Calculate base risk amount
        risk_amount = (self.account_balance * self.risk_per_trade) / 100
        
        # Adjust risk based on volatility
        volatility_factor = self.calculate_volatility_factor(df, index)
        adjusted_risk = risk_amount * volatility_factor
        
        # Adjust risk based on current drawdown
        drawdown_factor = self.calculate_drawdown_factor()
        final_risk = adjusted_risk * drawdown_factor
        
        # Calculate position size
        position_size = final_risk / abs(entry_price - stop_loss)
        
        # Apply maximum position size limit
        max_position_value = self.account_balance * 0.2  # Max 20% of account per trade
        position_size = min(position_size, max_position_value / entry_price)
        
        return position_size
    
    def calculate_volatility_factor(self, df, index, lookback=20):
        """Calculate risk adjustment factor based on market volatility"""
        current_atr = df['atr'].iloc[index]
        avg_atr = df['atr'].iloc[index-lookback:index].mean()
        
        if current_atr > avg_atr * 1.5:
            return 0.5  # Reduce risk in high volatility
        elif current_atr < avg_atr * 0.5:
            return 1.2  # Increase risk in low volatility
        return 1.0
    
    def calculate_drawdown_factor(self):
        """Adjust risk based on current drawdown"""
        if self.current_drawdown > self.max_drawdown * 0.7:
            return 0.5  # Reduce risk when approaching max drawdown
        elif self.current_drawdown > self.max_drawdown * 0.5:
            return 0.7
        return 1.0
    
    def calculate_stop_loss(self, df, index, signal_type):
        """Enhanced stop loss calculation"""
        # Get ATR for volatility-based stops
        atr = df['atr'].iloc[index]
        
        # Base stop distance on ATR and market conditions
        if signal_type == 'buy':
            # Find recent swing low
            swing_low = df['low'].iloc[index-20:index].min()
            # Use the larger of ATR-based stop or swing low
            atr_stop = df['close'].iloc[index] - (atr * 2)
            stop_loss = min(atr_stop, swing_low)
            
        else:  # sell signal
            # Find recent swing high
            swing_high = df['high'].iloc[index-20:index].max()
            # Use the larger of ATR-based stop or swing high
            atr_stop = df['close'].iloc[index] + (atr * 2)
            stop_loss = max(atr_stop, swing_high)
        
        return stop_loss
    
    def calculate_take_profit(self, df, index, entry_price, stop_loss):
        """Enhanced take profit calculation with multiple targets"""
        risk = abs(entry_price - stop_loss)
        
        if risk == 0:
            return entry_price, entry_price
        
        # Calculate multiple take profit levels
        if entry_price > stop_loss:  # Long position
            take_profit_1 = entry_price + (risk * 1.5)  # First target
            take_profit_2 = entry_price + (risk * 2.5)  # Second target
        else:  # Short position
            take_profit_1 = entry_price - (risk * 1.5)
            take_profit_2 = entry_price - (risk * 2.5)
        
        return take_profit_1, take_profit_2
    
    def calculate_trailing_stop(self, df, index, position_type, current_stop):
        """Calculate trailing stop level"""
        atr = df['atr'].iloc[index]
        
        if position_type == 'long':
            # Trail by 2 ATR below recent highs
            new_stop = df['high'].iloc[index] - (atr * 2)
            return max(new_stop, current_stop)  # Move stop up only
            
        else:  # short position
            # Trail by 2 ATR above recent lows
            new_stop = df['low'].iloc[index] + (atr * 2)
            return min(new_stop, current_stop)  # Move stop down only
    
    def update_equity(self, pnl):
        """Update equity curve and drawdown calculations"""
        current_equity = self.equity_curve[-1] * (1 + pnl)
        self.equity_curve.append(current_equity)
        
        # Update maximum equity and drawdown
        self.max_equity = max(self.max_equity, current_equity)
        self.current_drawdown = (self.max_equity - current_equity) / self.max_equity * 100
        
        # Check if max drawdown exceeded
        if self.current_drawdown > self.max_drawdown:
            print(f"Warning: Maximum drawdown exceeded! Current drawdown: {self.current_drawdown:.2f}%")
            return False
        return True
    
    def get_risk_metrics(self):
        """Calculate and return risk metrics"""
        return {
            'current_equity': self.equity_curve[-1],
            'max_equity': self.max_equity,
            'current_drawdown': self.current_drawdown,
            'max_position_size': self.account_balance * 0.2,
            'risk_per_trade': self.risk_per_trade * self.calculate_drawdown_factor()
        }

class EnhancedSignalGenerator:
    def __init__(self, lookback_period=30):
        self.lookback_period = lookback_period
        self.market_structure = MarketStructureAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.risk_manager = RiskManager()
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            random_state=42
        )
        self.lstm_model = None
        self.scaler = StandardScaler()
        
    def prepare_technical_features(self, df):
        """Create comprehensive technical indicators with data validation"""
        df = df.copy()
        
        # Check if we have enough data
        if len(df) < 3:  # Absolute minimum required data points
            print(f"Error: Insufficient data points ({len(df)}). Need at least 3 data points.")
            raise ValueError("Insufficient data for analysis")
        
        min_required_data = 200  # Ideal minimum data points
        if len(df) < min_required_data:
            print(f"Warning: Limited data points ({len(df)}). Using simplified analysis.")
            
            # Adjust window sizes based on available data
            window_20 = max(2, min(20, len(df) // 4))
            window_50 = max(2, min(50, len(df) // 2))
            window_200 = max(2, min(200, len(df) - 1))
        else:
            window_20 = 20
            window_50 = 50
            window_200 = 200

        # Calculate basic indicators with adjusted windows
        try:
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=window_20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=window_50)
            df['sma_200'] = ta.trend.sma_indicator(df['close'], window=window_200)
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=window_20)
            df['ema_50'] = ta.trend.ema_indicator(df['close'], window=window_50)
        except Exception as e:
            print(f"Warning: Error calculating moving averages: {e}")
            # Fallback to simple moving averages
            df['sma_20'] = df['close'].rolling(window=window_20, min_periods=1).mean()
            df['sma_50'] = df['close'].rolling(window=window_50, min_periods=1).mean()
            df['sma_200'] = df['close'].rolling(window=window_200, min_periods=1).mean()
            df['ema_20'] = df['sma_20']
            df['ema_50'] = df['sma_50']
        
        # Calculate swing points with adjusted window
        swing_window = max(3, min(5, len(df) // 10))
        df = self.market_structure.identify_swing_points(df, window=swing_window)
        
        # Calculate RSI with adjusted window
        rsi_window = min(14, len(df) // 4)
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_window).rsi()
        df['rsi_divergence'] = self.calculate_rsi_divergence(df)
        
        # MACD with adjusted windows
        macd_fast = min(12, len(df) // 5)
        macd_slow = min(26, len(df) // 3)
        macd_signal = min(9, len(df) // 6)
        macd = ta.trend.MACD(
            df['close'], 
            window_fast=macd_fast,
            window_slow=macd_slow,
            window_sign=macd_signal
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Volatility Indicators with adjusted windows
        atr_window = min(14, len(df) // 4)
        try:
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'],
                window=atr_window
            )
        except Exception as e:
            print(f"Warning: Error calculating ATR: {e}")
            # Fallback: Use simple volatility measure
            df['atr'] = df['high'] - df['low']
        
        bb_window = min(20, len(df) // 4)
        try:
            bollinger = ta.volatility.BollingerBands(df['close'], window=bb_window)
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
        except Exception as e:
            print(f"Warning: Error calculating Bollinger Bands: {e}")
            # Set to NaN if calculation fails
            df['bb_upper'] = df['bb_middle'] = df['bb_lower'] = float('nan')
        
        # Volume Indicators
        try:
            df = self.volume_analyzer.analyze_volume_profile(df)
        except Exception as e:
            print(f"Warning: Error calculating volume profile: {e}")
            # Add basic volume metrics as fallback
            df['volume_ma'] = df['volume'].rolling(window=window_20).mean()
            df['volume_std'] = df['volume'].rolling(window=window_20).std()
        
        # Market Structure with adjusted parameters
        try:
            df = self.market_structure.analyze_market_structure(df, lookback=min(20, len(df) // 4))
        except Exception as e:
            print(f"Warning: Error analyzing market structure: {e}")
            # Add basic trend indicators as fallback
            df['uptrend'] = df['close'] > df['close'].shift(1)
            df['downtrend'] = df['close'] < df['close'].shift(1)
        
        return df
    
    def calculate_rsi_divergence(self, df):
        """Calculate RSI divergence"""
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        divergence = pd.Series(index=df.index, data=0)
        
        for i in range(20, len(df)):
            if df['swing_high'].iloc[i]:  # Now swing_high exists
                # Look for bearish divergence
                if (df['high'].iloc[i] > df['high'].iloc[i-20:i].max() and 
                    rsi.iloc[i] < rsi.iloc[i-20:i].max()):
                    divergence.iloc[i] = -1
            elif df['swing_low'].iloc[i]:  # Now swing_low exists
                # Look for bullish divergence
                if (df['low'].iloc[i] < df['low'].iloc[i-20:i].min() and 
                    rsi.iloc[i] > rsi.iloc[i-20:i].min()):
                    divergence.iloc[i] = 1
                    
        return divergence
    
    def prepare_ml_features(self, df):
        """Prepare features for machine learning"""
        df = self.prepare_technical_features(df)
        
        # Updated feature columns to match available indicators
        feature_columns = [
            'rsi',
            'macd', 
            'macd_signal', 
            'macd_hist',
            'atr',
            'volume_ma',
            'volume_trend',
            'volume_momentum',
            'buying_pressure',
            'selling_pressure',
            'delta',
            'higher_high',
            'lower_low',
            'uptrend',
            'downtrend'
        ]
        
        # Ensure all features exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Remove missing features from the list
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        print(f"Using features: {feature_columns}")
        
        target = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        
        return df[feature_columns], target
    
    def train_models(self, df):
        """Train ML models with enhanced feature set and data validation"""
        X, y = self.prepare_ml_features(df)
        
        # Check if we have enough data
        min_required_samples = 10  # Absolute minimum samples needed
        if len(X) < min_required_samples:
            print(f"Warning: Not enough samples ({len(X)}) for ML training. Using simplified predictions.")
            self.rf_model = None
            self.lstm_model = None
            return
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest if we have enough data
        if len(X_train) >= 20:  # Minimum samples for RF
            try:
                self.rf_model.fit(X_train_scaled, y_train)
            except Exception as e:
                print(f"Warning: Error training Random Forest: {e}")
                self.rf_model = None
        else:
            print("Not enough data for Random Forest training")
            self.rf_model = None
        
        # Prepare LSTM data
        sequence_length = min(10, len(X_train) // 2)  # Adjust sequence length based on data
        if sequence_length < 2:  # Need at least 2 timesteps
            print("Not enough data for LSTM training")
            self.lstm_model = None
            return
        
        try:
            X_lstm = self.prepare_lstm_sequences(X_train_scaled, sequence_length)
            y_lstm = y_train[sequence_length:]
            
            if len(X_lstm) < 5:  # Need at least 5 sequences
                print("Not enough sequences for LSTM training")
                self.lstm_model = None
                return
            
            # Create and train LSTM with adjusted parameters
            self.lstm_model = self.create_lstm_model((sequence_length, X_train.shape[1]))
            
            # Adjust validation split based on data size
            validation_split = 0.2 if len(X_lstm) >= 20 else 0.1
            
            self.lstm_model.fit(
                X_lstm,
                y_lstm,
                validation_split=validation_split,
                epochs=min(50, max(10, len(X_lstm))),  # Adjust epochs based on data size
                batch_size=min(32, max(2, len(X_lstm) // 4)),  # Adjust batch size
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='loss',  # Monitor training loss instead of validation
                        patience=3,
                        restore_best_weights=True
                    )
                ],
                verbose=0
            )
        except Exception as e:
            print(f"Warning: Error training LSTM: {e}")
            self.lstm_model = None
    
    def create_lstm_model(self, input_shape):
        """Create LSTM model with improved architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_lstm_sequences(self, X, sequence_length):
        """Prepare sequences for LSTM"""
        sequences = []
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
        return np.array(sequences)

    def confirm_signal(self, df, index, signal_type='buy'):
        """Simplified signal confirmation with more lenient conditions"""
        # Get market context
        context = self.market_structure.analyze_market_context(df, index)
        
        # More lenient market phase check
        if signal_type == 'buy':
            if context['market_phase'] not in ['markup', 'accumulation', 'ranging']:  # Added ranging
                print(f"Rejected {signal_type} signal: Wrong market phase {context['market_phase']}")
                return False
        else:  # sell signal
            if context['market_phase'] not in ['markdown', 'distribution', 'ranging']:  # Added ranging
                print(f"Rejected {signal_type} signal: Wrong market phase {context['market_phase']}")
                return False

        # Simplified trend strength check
        if context['trend_strength'] < 0.2:  # Lowered from 0.3
            print("Rejected signal: Very weak trend")
            return False

        # Check volume pattern
        volume_pattern = self.volume_analyzer.analyze_volume_pattern(df, index)
        if not volume_pattern['trend']:  # Simplified volume check
            print("Rejected signal: Volume pattern not confirmed")
            return False

        print(f"Signal confirmed: {signal_type}")
        return True

    def generate_signals(self, df):
        """Generate trading signals with enhanced trend and momentum detection"""
        if len(df) < 3:  # Minimum required data points
            print("Error: Insufficient data for signal generation")
            return pd.Series(index=df.index, data=np.nan), df
        
        print(f"Data shape: {df.shape}")
        signals = pd.Series(index=df.index, data=np.nan)
        df = self.prepare_technical_features(df)
        
        try:
            self.train_models(df)
        except Exception as e:
            print(f"Warning: Error in model training: {e}")
            # Continue without ML models
            self.rf_model = None
            self.lstm_model = None
        
        # Add momentum and trend strength indicators
        df['Price_Change'] = df['close'].pct_change(3)  # 3-period price change
        df['Volume_Change'] = df['volume'].pct_change(3)
        df['ATR_Change'] = df['atr'].pct_change(3)
        
        # Calculate price momentum
        df['Price_Momentum'] = df['close'].diff(3) / df['close'].shift(3) * 100
        
        # Calculate trend strength
        df['Trend_Strength'] = abs((df['ema_20'] - df['ema_50']) / df['ema_50'] * 100)
        
        # Volatility breakout detection
        df['BB_Width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['BB_Position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Store ML confidence for plotting
        df['ML_Confidence'] = np.nan
        
        # Get ML predictions if models are available
        if self.rf_model is not None and self.lstm_model is not None:
            try:
                X, _ = self.prepare_ml_features(df)
                X_scaled = self.scaler.transform(X)
                rf_predictions = self.rf_model.predict_proba(X_scaled)
                
                sequence_length = min(10, len(X_scaled) // 2)
                X_lstm = self.prepare_lstm_sequences(X_scaled, sequence_length)
                lstm_predictions = self.lstm_model.predict(X_lstm, verbose=0)
            except Exception as e:
                print(f"Warning: Error in ML predictions: {e}")
                rf_predictions = None
                lstm_predictions = None
        else:
            rf_predictions = None
            lstm_predictions = None
        
        for i in range(sequence_length if self.lstm_model else 1, len(df)):
            # Calculate ML confidence if available
            if rf_predictions is not None and lstm_predictions is not None:
                try:
                    rf_conf = rf_predictions[i][1] if i < len(rf_predictions) else 0.5
                    lstm_conf = lstm_predictions[i-sequence_length][0] if i-sequence_length < len(lstm_predictions) else 0.5
                    ml_confidence = 0.6 * rf_conf + 0.4 * lstm_conf
                except Exception as e:
                    print(f"Warning: Error calculating ML confidence: {e}")
                    ml_confidence = 0.5
            else:
                ml_confidence = 0.5
            
            df.loc[df.index[i], 'ML_Confidence'] = ml_confidence
            
            # Reduced cooldown to catch more signals
            if i > 0 and signals.iloc[max(0, i-1):i].notna().any():
                continue
            
            current_price = df['close'].iloc[i]
            
            # More sensitive trend detection
            uptrend = (
                df['ema_20'].iloc[i] > df['ema_50'].iloc[i] and
                df['Price_Momentum'].iloc[i] > 0.2  # Reduced from 0.5
            )
            
            downtrend = (
                df['ema_20'].iloc[i] < df['ema_50'].iloc[i] and
                df['Price_Momentum'].iloc[i] < -0.2  # Reduced from -0.5
            )
            
            # Volatility and volume conditions
            high_volatility = df['BB_Width'].iloc[i] > df['BB_Width'].rolling(20).mean().iloc[i]
            volume_surge = df['Volume_Change'].iloc[i] > df['Volume_Change'].rolling(20).mean().iloc[i]
            
            # Buy conditions with multiple scenarios
            buy_conditions = [
                # Strong uptrend
                uptrend and ml_confidence > 0.45 and df['rsi'].iloc[i] < 70,
                
                # Oversold bounce with momentum
                df['rsi'].iloc[i] < 40 and df['macd_hist'].iloc[i] > df['macd_hist'].iloc[i-1] and
                df['close'].iloc[i] > df['close'].iloc[i-1],
                
                # Breakout with volume
                high_volatility and df['close'].iloc[i] > df['bb_upper'].iloc[i] and volume_surge,
                
                # Support bounce with EMA crossover
                df['close'].iloc[i] > df['ema_20'].iloc[i] and 
                df['close'].iloc[i-1] < df['ema_20'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].rolling(10).mean().iloc[i],
                
                # Price momentum with RSI confirmation
                df['Price_Momentum'].iloc[i] > 0.5 and
                df['rsi'].iloc[i] > 30 and df['rsi'].iloc[i] < 70 and
                df['volume'].iloc[i] > df['volume'].rolling(20).mean().iloc[i]
            ]
            
            # Sell conditions with multiple scenarios
            sell_conditions = [
                # Strong downtrend
                downtrend and ml_confidence < 0.55 and df['rsi'].iloc[i] > 30,
                
                # Overbought reversal with momentum
                df['rsi'].iloc[i] > 60 and df['macd_hist'].iloc[i] < df['macd_hist'].iloc[i-1] and
                df['close'].iloc[i] < df['close'].iloc[i-1],
                
                # Breakdown with volume
                high_volatility and df['close'].iloc[i] < df['bb_lower'].iloc[i] and volume_surge,
                
                # Resistance rejection with EMA
                df['close'].iloc[i] < df['ema_20'].iloc[i] and 
                df['close'].iloc[i-1] > df['ema_20'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].rolling(10).mean().iloc[i],
                
                # Sharp drop with volume
                df['Price_Change'].iloc[i] < -1.0 and volume_surge
            ]
            
            # Generate signals with simplified confirmation
            if any(buy_conditions):
                signals.iloc[i] = 1
                print(f"\nBuy Signal at {df.index[i]}")
                print(f"Price: {current_price:.2f}")
                print(f"ML Confidence: {ml_confidence:.2f}")
                print(f"RSI: {df['rsi'].iloc[i]:.2f}")
                print(f"Momentum: {df['Price_Momentum'].iloc[i]:.2f}")
                
            elif any(sell_conditions):
                signals.iloc[i] = -1
                print(f"\nSell Signal at {df.index[i]}")
                print(f"Price: {current_price:.2f}")
                print(f"ML Confidence: {ml_confidence:.2f}")
                print(f"RSI: {df['rsi'].iloc[i]:.2f}")
                print(f"Momentum: {df['Price_Momentum'].iloc[i]:.2f}")

        print(f"\nTotal signals generated: {(signals != 0).sum()}")
        return signals, df



def create_enhanced_figure(df, signals):
    """Create comprehensive trading chart"""
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            'Price Action & Market Structure',
            'Volume Profile',
            'RSI & Divergence',
            'MACD',
            'ATR & Volatility',
            'ML Confidence'
        ),
        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    
    # Price chart and EMAs
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ema_20'],
        name='EMA 20',
        line=dict(color='blue', dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ema_50'],
        name='EMA 50',
        line=dict(color='orange', dash='dash')
    ), row=1, col=1)
    
    # Add signals with larger markers
    buy_signals = signals == 1
    sell_signals = signals == -1
    
    if buy_signals.any():
        fig.add_trace(go.Scatter(
            x=df.index[buy_signals],
            y=df.loc[buy_signals, 'low'] * 0.998,  # Moved slightly lower
            mode='markers',
            name='Buy Signal',
            marker=dict(
                symbol='triangle-up',
                size=20,
                color='green',
                line=dict(color='darkgreen', width=2)
            )
        ), row=1, col=1)
    
    if sell_signals.any():
        fig.add_trace(go.Scatter(
            x=df.index[sell_signals],
            y=df.loc[sell_signals, 'high'] * 1.002,  # Moved slightly higher
            mode='markers',
            name='Sell Signal',
            marker=dict(
                symbol='triangle-down',
                size=20,
                color='red',
                line=dict(color='darkred', width=2)
            )
        ), row=1, col=1)
    
    # Add all other subplots
    # Volume
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'],
        name='Volume',
        marker_color=np.where(df['close'] >= df['open'], 'green', 'red')
    ), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['rsi'],
        name='RSI',
        line=dict(color='purple')
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=df['macd'],
        name='MACD',
        line=dict(color='blue')
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['macd_signal'],
        name='Signal',
        line=dict(color='orange')
    ), row=4, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df['macd_hist'],
        name='MACD Histogram',
        marker_color=np.where(df['macd_hist'] >= 0, 'green', 'red')
    ), row=4, col=1)
    
    # ATR
    fig.add_trace(go.Scatter(
        x=df.index, y=df['atr'],
        name='ATR',
        line=dict(color='blue')
    ), row=5, col=1)
    
    # ML Confidence
    if 'ML_Confidence' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ML_Confidence'],
            name='ML Confidence',
            line=dict(color='purple')
        ), row=6, col=1)
        fig.add_hline(y=0.6, line_dash="dash", line_color="green", row=6, col=1)
        fig.add_hline(y=0.4, line_dash="dash", line_color="red", row=6, col=1)
    
    fig.update_layout(
        title=f'Enhanced Crypto Analysis - Last Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        showlegend=True,
        height=1600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_enhanced_dash_app():
    """Create enhanced Dash application"""
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Enhanced Crypto Trading Analysis", 
                style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Control panel
        html.Div([
            # Trading pair selection
            html.Div([
                html.Label('Trading Pair:'),
                dcc.Input(
                    id='symbol-input',
                    type='text',
                    value='VANAUSDT',
                    style={'marginRight': '20px'}
                ),
            ], style={'marginBottom': '10px'}),
            
            # Timeframe selection
            html.Div([
                html.Label('Timeframe:'),
                dcc.Dropdown(
                    id='interval-dropdown',
                    options=[
                        {'label': '1 minute', 'value': '1m'},
                        {'label': '5 minutes', 'value': '5m'},
                        {'label': '15 minutes', 'value': '15m'},
                        {'label': '1 hour', 'value': '1h'},
                        {'label': '4 hours', 'value': '4h'},
                        {'label': '1 day', 'value': '1d'}
                    ],
                    value='1h',
                    style={'width': '200px'}
                ),
            ], style={'marginBottom': '10px'}),
            
            # Risk management settings
            html.Div([
                html.Label('Risk Per Trade (%):'),
                dcc.Input(
                    id='risk-input',
                    type='number',
                    value=1.0,
                    min=0.1,
                    max=5.0,
                    step=0.1,
                    style={'marginRight': '20px'}
                ),
            ], style={'marginBottom': '10px'}),
            
        ], style={'margin': '20px', 'padding': '20px', 'border': '1px solid #ddd'}),
        
        # Market statistics
        html.Div(id='market-stats', 
                 style={'margin': '20px', 'padding': '20px', 'border': '1px solid #ddd'}),
        
        # Main chart
        dcc.Graph(id='live-chart', style={'height': '1600px'}),
        
        # Update interval
        dcc.Interval(
            id='interval-component',
            interval=120*1000,  # 1 minute in milliseconds
            n_intervals=0
        )
    ])
    
    @app.callback(
        [Output('live-chart', 'figure'),
         Output('market-stats', 'children')],
        [Input('interval-component', 'n_intervals'),
         Input('symbol-input', 'value'),
         Input('interval-dropdown', 'value'),
         Input('risk-input', 'value')]
    )
    def update_chart(n, symbol, interval, risk_per_trade):
        # Fetch data
        stream = BinanceDataStream(symbol=symbol, interval=interval, limit=100)
        df = stream.fetch_data()
        
        if df is None or len(df) < 3:
            return go.Figure(), "Error: Insufficient data for analysis"
        
        try:
            # Generate signals
            signal_generator = EnhancedSignalGenerator()
            signal_generator.risk_manager.risk_per_trade = risk_per_trade
            signals, df = signal_generator.generate_signals(df)
            
            # Create market stats with data validation
            stats = html.Div([
                html.H3("Market Statistics"),
                html.Div([
                    html.P(f"Current Price: ${df['close'].iloc[-1]:,.2f}"),
                    html.P(f"Volume: {df['volume'].iloc[-1]:,.2f}"),
                    
                    # Calculate 24h change only if we have enough data
                    html.P(
                        f"Change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}% "
                        f"(since {df.index[0].strftime('%Y-%m-%d %H:%M')})"
                    ) if len(df) > 1 else html.P("Change: Insufficient data"),
                    
                    # Show ATR if available, otherwise show basic volatility
                    html.P(
                        f"Current Volatility (ATR): {df['atr'].iloc[-1]:.2f}"
                        if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1])
                        else f"Current Volatility: {(df['high'].iloc[-1] - df['low'].iloc[-1]):.2f}"
                    ),
                    
                    # Show RSI if available
                    html.P(
                        f"RSI: {df['rsi'].iloc[-1]:.2f}"
                        if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1])
                        else "RSI: Calculating..."
                    ),
                    
                    # Show market structure if available
                    html.P(
                        f"Market Structure: {'Uptrend' if df['uptrend'].iloc[-1] else 'Downtrend' if df['downtrend'].iloc[-1] else 'Sideways'}"
                        if 'uptrend' in df.columns and 'downtrend' in df.columns
                        else "Market Structure: Insufficient data"
                    ),
                    
                    # Add warning for limited data
                    html.P(
                        f"Warning: Limited historical data ({len(df)} points)",
                        style={'color': 'orange', 'fontWeight': 'bold'}
                    ) if len(df) < 24 else None,
                ])
            ])
            
            # Create chart
            fig = create_enhanced_figure(df, signals)
            
            return fig, stats
            
        except Exception as e:
            print(f"Error in update_chart: {e}")
            # Return a more informative error message
            error_stats = html.Div([
                html.H3("Error in Analysis", style={'color': 'red'}),
                html.P(f"Error details: {str(e)}"),
                html.P(f"Available data points: {len(df)}"),
                html.P("Try switching to a shorter timeframe or a more established trading pair.")
            ])
            return go.Figure(), error_stats
    
    return app

def main():
    """Main function to run the application"""
    try:
        app = create_enhanced_dash_app()
        print("Starting enhanced trading analysis dashboard...")
        app.run_server(debug=True, port=8050)
        
    except Exception as e:
        print(f"Error running the application: {e}")

if __name__ == "__main__":
    main()
