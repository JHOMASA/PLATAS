import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from textblob import TextBlob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import time
import random
from sklearn.metrics import mean_absolute_error
from functools import lru_cache
import requests
from typing import  Dict, Any,Tuple, Optional,List
from sklearn.preprocessing import MinMaxScaler
import uuid

# Configuration

ALPHA_VANTAGE_API_KEY = "QDRRN1Y7K4EDJYT2"
FINGPT_API_KEY = "AIzaTRDjNFU6WAx6FJ74zhm2vQqWyD5MsYKUcOk"  # Replace with actual key
NEWS_API_KEY = "3f8e6bb1fb72490b835c800afcadd1aa"      # Replace with actual key

st.set_page_config(layout="wide")
st.title("📊 Advanced Stock Analysis Dashboard")
@lru_cache(maxsize=32)
def fetch_stock_data_yahoo(symbol: str, period: str = "1y") -> Tuple[pd.DataFrame, str]:
    """Fetch stock data from Yahoo Finance API"""
    try:
        # Map period to Yahoo Finance format
        period_map = {
            "1mo": "1mo",
            "3mo": "3mo",
            "6mo": "6mo",
            "1y": "1y",
            "2y": "2y",
            "5y": "5y"
        }
        
        yahoo_period = period_map.get(period, "1y")
        
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=yahoo_period)
        
        if df.empty:
            return pd.DataFrame(), "No data available for this symbol"
            
        # Clean and standardize the data
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        
        return df, None
        
    except Exception as e:
        return pd.DataFrame(), f"Error fetching data: {str(e)}"

@lru_cache(maxsize=32)
def fetch_stock_data_cached(symbol: str, period: str = "1y") -> Tuple[bool, str]:
    """Fetch stock data with caching"""
    try:
        df, error = fetch_stock_data_yahoo(symbol, period)
        if error:
            return False, error
        return True, df.to_json(date_format='iso')
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_stock_data(symbol: str, period: str = "1y") -> Tuple[pd.DataFrame, str]:
    """Main function to get stock data"""
    success, result = fetch_stock_data_cached(symbol, period)
    if success:
        return pd.read_json(result), None
    return pd.DataFrame(), result

def get_alpha_vantage_ratios(ticker: str) -> Dict[str, Optional[float]]:
    """Get ROE and ROA from Alpha Vantage API with proper typing"""
    ratios = {"ROE": None, "ROA": None}
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "ReturnOnEquityTTM" in data:
            try:
                ratios["ROE"] = float(data["ReturnOnEquityTTM"]) / 100  # Convert percentage to decimal
            except (ValueError, TypeError):
                pass
                
        if "ReturnOnAssetsTTM" in data:
            try:
                ratios["ROA"] = float(data["ReturnOnAssetsTTM"]) / 100  # Convert percentage to decimal
            except (ValueError, TypeError):
                pass
                
    except requests.exceptions.RequestException as e:
        st.error(f"Alpha Vantage API request failed: {str(e)}")
    except Exception as e:
        st.error(f"Error processing Alpha Vantage data: {str(e)}")
    
    return ratios

def safe_float(value: Any, div_by: int = 1) -> Optional[float]:
    """Enhanced safe float converter with division option"""
    try:
        if value is None:
            return None
        return float(value) / div_by
    except (ValueError, TypeError):
        return None
def get_sector_peers(ticker: str) -> Tuple[str, str, List[str]]:
    """Dynamically identify sector and peers for any stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        sector = info.get('sector', 'General')
        industry = info.get('industry', 'Various')
        peers = info.get('competitors', []) or []
        
        return sector, industry, peers
    
    except Exception as e:
        st.warning(f"Couldn't determine sector: {str(e)}")
        return "General", "Various", []

def get_sector_averages(sector: str) -> Dict[str, float]:
    """Get real-time sector averages from multiple reliable sources"""
    if sector in SECTOR_AVG_CACHE:
        return SECTOR_AVG_CACHE[sector]
    
    # First try: Financial Modeling Prep (most comprehensive)
    if FMP_API_KEY:
        try:
            url = f"https://financialmodelingprep.com/api/v4/industry/performance?name={sector}&apikey={FMP_API_KEY}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list):
                    averages = {
                        'P/E Ratio': safe_float(data[0].get('pe')),
                        'P/B Ratio': safe_float(data[0].get('priceToBook')),
                        'Debt/Equity': safe_float(data[0].get('debtToEquity')),
                        'Current Ratio': safe_float(data[0].get('currentRatio')),
                        'ROE': safe_float(data[0].get('roe')),
                        'ROA': safe_float(data[0].get('roa'))
                    }
                    SECTOR_AVG_CACHE[sector] = averages
                    return averages
        except Exception:
            pass

    # Second try: Alpha Vantage sector performance
    if ALPHA_VANTAGE_API_KEY:
        try:
            url = f"https://www.alphavantage.co/query?function=SECTOR&apikey={ALPHA_VANTAGE_API_KEY}"
            response = requests.get(url, timeout=10)
            data = response.json()
            sector_data = data.get('Rank E: Profitability', {}).get(sector, {})
            if sector_data:
                averages = {
                    'P/E Ratio': safe_float(sector_data.get('PE Ratio')),
                    'ROE': safe_float(sector_data.get('ROE')),
                    'ROA': safe_float(sector_data.get('ROA'))
                }
                SECTOR_AVG_CACHE[sector] = averages
                return averages
        except Exception:
            pass

    # Third try: Yahoo Finance industry averages
    try:
        sector_tickers = get_sector_tickers(sector)
        if sector_tickers:
            yf_sector = yf.Tickers(sector_tickers)
            sector_pe = yf_sector.stats('trailingPE').median()
            sector_pb = yf_sector.stats('priceToBook').median()
            averages = {
                'P/E Ratio': sector_pe,
                'P/B Ratio': sector_pb
            }
            SECTOR_AVG_CACHE[sector] = averages
            return averages
    except Exception:
        pass

    # Final fallback: Cached sector averages
    return get_cached_sector_averages(sector)
def get_sector_tickers(sector: str) -> List[str]:
    """Get representative tickers for a sector"""
    SECTOR_ETFS = {
        'Technology': ['XLK', 'VGT', 'QQQ'],
        'Financial Services': ['XLF', 'VFH'],
        'Healthcare': ['XLV', 'VHT'],
        'Consumer Cyclical': ['XLY', 'VCR'],
        'Communication Services': ['XLC', 'VOX'],
        'Industrials': ['XLI', 'VIS'],
        'Consumer Defensive': ['XLP', 'VDC'],
        'Energy': ['XLE', 'VDE'],
        'Utilities': ['XLU', 'VPU'],
        'Real Estate': ['XLRE', 'VNQ'],
        'Basic Materials': ['XLB', 'VAW']
    }
    return SECTOR_ETFS.get(sector, ['SPY'])

def get_cached_sector_averages(sector: str) -> Dict[str, float]:
    """Get recently cached sector averages with auto-update capability"""
    CACHE_FILE = "sector_cache.json"
    
    # Try to load from cache if recent
    if os.path.exists(CACHE_FILE):
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if file_age < timedelta(days=7):
            with open(CACHE_FILE) as f:
                cache = json.load(f)
                return cache.get(sector, cache.get('General'))
    
    # Default fallback values
    DEFAULTS = {
        'Technology': {'P/E Ratio': 28, 'P/B Ratio': 6.5, 'ROE': 18, 'ROA': 9},
        'Financial Services': {'P/E Ratio': 14, 'P/B Ratio': 1.2, 'ROE': 12, 'ROA': 1},
        'Healthcare': {'P/E Ratio': 22, 'P/B Ratio': 4, 'ROE': 15, 'ROA': 7},
        'General': {'P/E Ratio': 15, 'P/B Ratio': 2.5, 'ROE': 12, 'ROA': 6}
    }
    
    return DEFAULTS.get(sector, DEFAULTS['General'])

def get_yahoo_ratios(ticker: str, fmp_api_key: str = None) -> Dict[str, Any]:  # Added fmp_api_key parameter
    """Get financial ratios from Yahoo Finance"""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        
        if not info:    
            st.error("No financial data available for this ticker")
            return None
            
        # Extract relevant ratios
        ratios = {
            'priceEarningsRatio': info.get('trailingPE'),
            'priceToBookRatio': info.get('priceToBook'),
            'debtEquityRatio': info.get('debtToEquity'),
            'currentRatio': info.get('currentRatio'),
            'returnOnEquity': info.get('returnOnEquity'),  # Fixed typo in key
            'returnOnAssets': info.get('returnOnAssets')
        }
        
        # Add FMP fallback if needed
        if ratios['returnOnEquity'] is None or ratios['returnOnAssets'] is None:
            av_ratios = get_alpha_vantage_ratios(ticker)
            if av_ratios:
                ratios['returnOnEquity'] = ratios['returnOnEquity'] or av_ratios['returnOnEquity']
                ratios['returnOnAssets'] = ratios['returnOnAssets'] or av_ratios['returnOnAssets']
        
        return {k: float(v) if v is not None else None for k, v in ratios.items()}
    except Exception as e:
        st.error(f"Error fetching ratios: {str(e)}")
        return None

def display_financial_ratios(ratios: Dict[str, Any], ticker: str):
    """Enhanced ratio display with dynamic sector comparison"""
    try:
        if not ratios:
            st.error("No ratio data available")
            return
            
        # Get sector context
        sector, industry, peers = get_sector_peers(ticker)
        sector_avgs = get_sector_averages(sector)
        
        # Prepare display data
        display_data = prepare_display_data(ratios)
        
        if not display_data:
            st.error("No valid ratio data available for display")
            return

        # Create visualization with sector comparison
        create_dynamic_chart(display_data, ticker, sector, sector_avgs)
        show_metric_analysis(display_data, sector_avgs)
    except Exception as e:
        st.error(f"Error displaying ratios: {str(e)}")


def prepare_display_data(ratios: Dict[str, Any]) -> Dict[str, float]:
    """Prepares and validates ratio data for display."""
    ratio_map = {
        'priceEarningsRatio': 'P/E Ratio',
        'priceToBookRatio': 'P/B Ratio',
        'debtEquityRatio': 'Debt/Equity',
        'currentRatio': 'Current Ratio',
        'returnOnEquity': 'ROE',
        'returnOnAssets': 'ROA'
    }

    display_data = {}
    for api_key, display_name in ratio_map.items():
        if api_key in ratios and ratios[api_key] is not None:
            try:
                value = float(ratios[api_key])
                display_data[display_name] = value *(100 if display_name in ["ROE", "ROA"] else 1) 
            except(TypeError, ValueError):
                continue
    return display_data

def create_dynamic_chart(display_data: Dict[str, float], ticker: str, 
                        sector: str, sector_avgs: Dict[str, float]):
    """Creates interactive visualization with sector comparison"""
    fig = go.Figure()
    
    # Add company data
    fig.add_trace(go.Bar(
        x=list(display_data.keys()),
        y=list(display_data.values()),
        name=ticker,
        text=[f"{v:.2f}%" if k in ['ROE', 'ROA'] else f"{v:.2f}" 
              for k, v in display_data.items()],
        textposition='auto'
    ))
    
    # Add sector averages for available metrics
    sector_x = [k for k in display_data.keys() if k in sector_avgs]
    if sector_x:
        sector_y = [sector_avgs[k] for k in sector_x]
        fig.add_trace(go.Bar(
            x=sector_x,
            y=sector_y,
            name=f'{sector} Avg',
            text=[f"{v:.1f}%" if k in ['ROE', 'ROA'] else f"{v:.1f}" 
                  for k, v in zip(sector_x, sector_y)],
            textposition='auto'
        ))
    
    fig.update_layout(
        barmode='group',
        title=f"{ticker} vs {sector} Sector Averages",
        yaxis_title="Value",
        hovermode="x unified",
        height=max(400, len(display_data) * 60)  # Dynamic height
    )
    st.plotly_chart(fig, use_container_width=True)
def get_alpha_vantage_ratios(ticker: str) -> Dict[str, Optional[float]]:
    """Ensures proper decimal conversion from API"""
    ratios = {"ROE": None, "ROA": None}
    try:
        # ... API call code ...
        
        # These values are already in percentage form from API (e.g., 15.25 means 15.25%)
        # So we divide by 100 to convert to decimal (0.1525)
        if "ReturnOnEquityTTM" in data:
            ratios["ROE"] = safe_float(data["ReturnOnEquityTTM"], div_by=100)
        if "ReturnOnAssetsTTM" in data:
            ratios["ROA"] = safe_float(data["ReturnOnAssetsTTM"], div_by=100)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return ratios

def show_metric_analysis(display_data: Dict[str, float], sector_avgs: Dict[str, float]):
    """Displays detailed ratio analysis with correct percentage handling and enhanced visuals"""
    st.subheader("📊 Detailed Ratio Analysis")
    
    # Define ratio categories for proper assessment
    profitability_ratios = ['ROE', 'ROA']
    valuation_ratios = ['P/E Ratio', 'P/B Ratio']
    liquidity_ratios = ['Current Ratio']
    leverage_ratios = ['Debt/Equity']
    
    for ratio, displayed_value in display_data.items():
        # Determine if we should handle as percentage
        is_percentage = ratio in profitability_ratios
        
        # Convert displayed values back to decimal for calculations
        actual_value = displayed_value / 100 if is_percentage else displayed_value
        
        with st.expander(f"{ratio}", expanded=False):
            cols = st.columns([1, 1, 0.8, 1.2])  # Adjusted column ratios
            
            sector_avg = sector_avgs.get(ratio)
            
            with cols[0]:
                # Company value display
                st.metric(
                    label="Your Company",
                    value=f"{displayed_value:,.2f}{'%' if is_percentage else ''}",
                    help="Raw value from company financials"
                )
            
            if sector_avg is not None:
                # Convert sector average if needed
                sector_avg_actual = sector_avg / 100 if is_percentage else sector_avg
                
                # Calculate meaningful difference
                if sector_avg_actual != 0:  # Prevent division by zero
                    diff = (actual_value - sector_avg_actual) / sector_avg_actual * 100
                else:
                    diff = 0
                
                with cols[1]:
                    # Sector average display
                    st.metric(
                        label="Sector Benchmark",
                        value=f"{sector_avg:,.2f}{'%' if is_percentage else ''}",
                        delta=f"{diff:+.1f}%" if sector_avg_actual != 0 else "N/A",
                        help="Industry average for comparison"
                    )
                
                with cols[2]:
                    # Dynamic assessment indicator
                    if ratio in profitability_ratios + liquidity_ratios:
                        status = "✅" if diff > 0 else "⚠️" if diff < -10 else "➖"
                        assessment = "Better" if diff > 0 else "Worse" if diff < -10 else "Neutral"
                    elif ratio in valuation_ratios + leverage_ratios:
                        status = "✅" if diff < 0 else "⚠️" if diff > 10 else "➖"
                        assessment = "Better" if diff < 0 else "Worse" if diff > 10 else "Neutral"
                    else:
                        status = "➖"
                        assessment = "Neutral"
                    
                    st.metric(
                        label="Assessment",
                        value=f"{status} {assessment}",
                        help="Compared to sector average"
                    )
                
                with cols[3]:
                    # Enhanced ratio-specific insights
                    insight = generate_enhanced_insight(
                        ratio, 
                        actual_value, 
                        sector_avg_actual,
                        diff
                    )
                    st.info(insight)
                
                # Add visual gauge
                show_ratio_gauge(actual_value, sector_avg_actual, ratio)
            else:
                with cols[1]:
                    st.warning("No sector benchmark available")
                with cols[3]:
                    st.info(generate_standalone_insight(ratio, actual_value))

def generate_enhanced_insight(ratio: str, value: float, sector_avg: float, diff: float) -> str:
    """Generates more nuanced insights considering ratio magnitude"""
    if ratio == 'ROE':
        if value > 0.20:  # 20%
            base = "Excellent profitability"
        elif value > 0.15:
            base = "Strong profitability"
        elif value > 0.10:
            base = "Average profitability"
        else:
            base = "Weak profitability"
        
        if abs(diff) > 20:
            comp = "significantly {} than sector".format("higher" if diff > 0 else "lower")
        elif abs(diff) > 10:
            comp = "moderately {} than sector".format("higher" if diff > 0 else "lower")
        else:
            comp = "in line with sector"
        
        return f"{base} ({comp}). Target range: 15-20%"
    
    elif ratio == 'P/E Ratio':
        if value < 10:
            base = "Very low valuation"
        elif value < 15:
            base = "Low valuation"
        elif value < 25:
            base = "Reasonable valuation"
        else:
            base = "High valuation"
        
        return f"{base}. Sector average: {sector_avg:.1f}. Diff: {diff:+.1f}%"
    
    # Add similar enhanced insights for other ratios...
    return "Financial metric analysis"

def show_ratio_gauge(value: float, benchmark: float, ratio_name: str):
    """Visual gauge showing performance relative to benchmark"""
    if benchmark == 0:
        return
    
    max_val = max(value, benchmark) * 1.5
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{ratio_name} Performance"},
        delta = {'reference': benchmark},
        gauge = {
            'axis': {'range': [0, max_val]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, benchmark*0.7], 'color': "lightgray"},
                {'range': [benchmark*0.7, benchmark*1.3], 'color': "gray"},
                {'range': [benchmark*1.3, max_val], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': benchmark
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True, use_container_height=True)
def generate_standalone_insight(ratio: str, value: float) -> str:
    """Insights when no sector benchmark is available"""
    if ratio == 'ROE':
        if value > 0.20: return "Excellent profitability (>20%)"
        if value > 0.15: return "Strong profitability (15-20%)"
        if value > 0.10: return "Average profitability (10-15%)"
        return "Below average profitability (<10%)"
    
    elif ratio == 'P/E Ratio':
        if value < 10: return "Very low valuation (P/E <10)"
        if value < 15: return "Low valuation (P/E 10-15)"
        if value < 25: return "Reasonable valuation (P/E 15-25)"
        return "High valuation (P/E >25)"
    
    # Add other ratio insights...
    return "Financial metric analysis"

def calculate_risk_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate market risk metrics from price data."""
    try:
        if data.empty or 'Close' not in data.columns:
            return {}
        
        ratios = {}
        returns = np.log(1 + data['Close'].pct_change()).dropna()
        
        if not returns.empty:
            ratios.update({
                'volatility': returns.std() * np.sqrt(252),
                'sharpeRatio': (returns.mean() / returns.std() * np.sqrt(252)) 
                              if returns.std() != 0 else None,
            })
        
        rolling_max = data['Close'].cummax()
        daily_drawdown = data['Close']/rolling_max - 1
        ratios['maximumDrawdown'] = daily_drawdown.min()
        
        return {k: float(v) if v is not None else None for k, v in ratios.items()}
        
    except Exception as e:
        st.error(f"Risk calculation error: {str(e)}")
        return {}

def monte_carlo_simulation(data: pd.DataFrame, n_simulations: int = 1000, days: int = 180) -> dict:
    try:
        # Calculate daily returns
        returns = np.log(1 + data['Close'].pct_change())
        mu = returns.mean()
        sigma = returns.std()
        last_price = data['Close'].iloc[-1]

        # Generate random walks
        raw_simulations = np.zeros((days, n_simulations))
        raw_simulations[0] = last_price

        for day in range(1, days):
            shock = np.random.normal(mu, sigma, n_simulations)
            raw_simulations[day] = raw_simulations[day - 1] * np.exp(shock)

        # Validate raw simulations
        if np.any(raw_simulations <= 0):
            raise ValueError("Invalid price values in raw simulations")

        # Apply smoothing techniques
        window_size = max(5, min(20, days // 5))  # More balanced window

        # Simple Moving Average
        ma_simulations = np.zeros_like(raw_simulations)
        for i in range(n_simulations):
            ma_series = pd.Series(raw_simulations[:, i]).rolling(window=window_size, min_periods=1).mean()
            ma_simulations[:, i] = ma_series.bfill().ffill().values  # Better filling

        # Weighted Moving Average
        wma_simulations = np.zeros_like(raw_simulations)
        weights = np.arange(1, window_size + 1)
        weights = weights / weights.sum()

        for i in range(n_simulations):
            series = pd.Series(raw_simulations[:, i])
            wma_series = series.rolling(window=window_size, min_periods=1).apply(
                lambda x: np.dot(weights[-len(x):], x), raw=True
            )
            wma_simulations[:, i] = wma_series.bfill().ffill().values

        # Debug checks
        if np.all(ma_simulations == 0):
            st.error("MA simulations failed - all zeros")
        if np.all(wma_simulations == 0):
            st.error("WMA simulations failed - all zeros")

        return {
            'raw': raw_simulations,
            'ma': ma_simulations,
            'wma': wma_simulations
        }

    except Exception as e:
        raise Exception(f"Enhanced Monte Carlo simulation failed: {str(e)}")
def train_holt_winters(data: pd.DataFrame, seasonal_periods: int) -> Tuple[object, str]:
    """Train Holt-Winters forecasting model"""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        model = ExponentialSmoothing(
            data['Close'],
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add'
        ).fit()
        return model, None
    except Exception as e:
        return None, f"Holt-Winters training failed: {str(e)}"

def predict_holt_winters(model, periods: int = 30) -> pd.Series:
    """Generate predictions using Holt-Winters model"""
    try:
        return model.forecast(periods)
    except Exception as e:
        raise Exception(f"Holt-Winters prediction failed: {str(e)}")

def create_lagged_features(data: pd.DataFrame, lags: int = 34) -> pd.DataFrame:
    """Create exactly 34 lagged features (including the Close price)"""
    df = data.copy()
    if 'Date' in df.columns:
        df = df.set_index('Date')
    
    # Create exactly 34 lag features (lag_1 to lag_34)
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    
    # Keep only Close price and the lag features
    df = df[['Close'] + [f'lag_{i}' for i in range(1, lags + 1)]]
    df.dropna(inplace=True)
    
    return df

def train_random_forest(data: pd.DataFrame) -> object:
    """Train Random Forest model with exactly 34 features"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Create features - will produce 34 lag features + Close price
        df = create_lagged_features(data, lags=34)
        
        # Verify feature count (34 lag features + Close = 35 columns total)
        if len(df.columns) != 35:
            raise ValueError(f"Feature count mismatch. Expected 35 columns (Close + 34 lags), got {len(df.columns)}")
            
        # Prepare features (34 lag features) and target (Close price)
        X = df.drop(columns=['Close'])  # This should be exactly 34 features
        y = df['Close']
        
        # Verify X has exactly 34 features
        if X.shape[1] != 34:
            raise ValueError(f"Expected 34 features, got {X.shape[1]}")
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)
        
        return model
        
    except Exception as e:
        raise Exception(f"Random Forest training failed: {str(e)}")

def predict_random_forest(model, data: pd.DataFrame, periods: int = 30) -> np.ndarray:
    """Generate predictions using the trained Random Forest model"""
    try:
        # Create working copy of data
        current_data = data.copy()
        if 'Date' in current_data.columns:
            current_data = current_data.set_index('Date')
            
        # Verify sufficient history (need at least 34 previous values)
        if len(current_data) < 34:
            raise ValueError(f"Need at least 34 days of history, got {len(current_data)}")
            
        predictions = []
        
        for _ in range(periods):
            # Create feature vector with exactly 34 lagged values
            latest_features = [current_data['Close'].iloc[-i] for i in range(1, 35)]
            
            # Make prediction
            pred = model.predict([latest_features])[0]
            predictions.append(pred)
            
            # Update data with new prediction
            new_date = current_data.index[-1] + pd.Timedelta(days=1)
            new_row = pd.DataFrame({'Close': [pred]}, index=[new_date])
            current_data = pd.concat([current_data, new_row])
        
        return np.array(predictions)
        
    except Exception as e:
        raise Exception(f"Random Forest prediction failed: {str(e)}")
        
    except Exception as e:
        raise Exception(f"Random Forest prediction failed: {str(e)}")
def train_lstm_model(data: pd.DataFrame) -> Tuple[object, object]:
    """Basic LSTM model training"""
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['Close']].values)
        
        # Prepare sequences
        X, y = [], []
        n_lookback = 60  # Number of days to look back
        for i in range(n_lookback, len(scaled_data)):
            X.append(scaled_data[i-n_lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        return model, scaler
    except Exception as e:
        raise Exception(f"LSTM training failed: {str(e)}")

def predict_lstm(model, scaler, data: pd.DataFrame, periods: int = 30) -> np.ndarray:
    """Generate LSTM predictions"""
    try:
        inputs = data['Close'].values[-60:].reshape(-1,1)
        inputs = scaler.transform(inputs)
        
        predictions = []
        for _ in range(periods):
            x_input = inputs[-60:].reshape(1,60,1)
            pred = model.predict(x_input, verbose=0)
            inputs = np.append(inputs, pred)
            predictions.append(pred[0,0])
            
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
        return predictions.flatten()
    except Exception as e:
        raise Exception(f"LSTM prediction failed: {str(e)}")


def train_arima_model(data: pd.DataFrame) -> object:
    """Train ARIMA model"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(data['Close'], order=(5,1,0))
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        raise Exception(f"ARIMA training failed: {str(e)}")

def predict_arima(model, periods: int = 30) -> pd.Series:
    """Generate ARIMA predictions"""
    try:
        predictions = model.forecast(steps=periods)
        return predictions
    except Exception as e:
        raise Exception(f"ARIMA prediction failed: {str(e)}")

def train_xgboost_model(data: pd.DataFrame) -> object:
    """Train XGBoost model"""
    try:
        from xgboost import XGBRegressor
        from sklearn.preprocessing import MinMaxScaler
        
        # Create features (using lagged values)
        df = data.copy()
        for i in range(1, 31):
            df[f'lag_{i}'] = df['Close'].shift(i)
        df.dropna(inplace=True)
        
        X = df.drop(columns=['Close'])
        y = df['Close']
        
        model = XGBRegressor(n_estimators=100)
        model.fit(X, y)
        return model
    except Exception as e:
        raise Exception(f"XGBoost training failed: {str(e)}")

def predict_xgboost(model, data: pd.DataFrame, periods: int = 30) -> np.ndarray:
    """Generate XGBoost predictions"""
    try:
        # Create future dataframe with lagged values
        future = data.copy()
        for i in range(1, periods+1):
            if i == 1:
                future.loc[future.index[-1] + pd.Timedelta(days=1), 'Close'] = np.nan
            future[f'lag_{i}'] = future['Close'].shift(i)
        
        # Predict
        X_pred = future.drop(columns=['Close']).iloc[-periods:]
        predictions = model.predict(X_pred)
        return predictions
    except Exception as e:
        raise Exception(f"XGBoost prediction failed: {str(e)}")


    
def display_stock_analysis(stock_data, ticker):
    col1, col2 = st.columns(2)
    
    with col1:
        # Price History
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price'))
        fig1.update_layout(title=f"{ticker} Price History", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig1, use_container_width=True)
      
    with col2:
        # Volume Analysis
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume'))
        fig2.update_layout(title="Trading Volume", xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Technical Indicators
    st.subheader("Technical Indicators")
    indicators = st.multiselect("Select indicators", 
                               ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"],
                               default=["SMA", "RSI"])
    
    if indicators:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price'))
        
        if "SMA" in indicators:
            sma = stock_data['Close'].rolling(20).mean()
            fig3.add_trace(go.Scatter(x=stock_data.index, y=sma, name='20-day SMA'))
        
        if "RSI" in indicators:
            delta = stock_data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=rsi, name='RSI'))
            fig_rsi.update_layout(title="Relative Strength Index (RSI)", yaxis_range=[0,100])
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        fig3.update_layout(title="Technical Indicators")
        st.plotly_chart(fig3, use_container_width=True)


def display_monte_carlo(simulations):
    st.subheader("Simulation Smoothing Options")
    smooth_type = st.radio("Select smoothing type", ["Raw", "Moving Average", "Weighted MA"], horizontal=True)

    if smooth_type == "Moving Average":
        data = simulations['ma']
    elif smooth_type == "Weighted MA":
        data = simulations['wma']
    else:
        data = simulations['raw']

    # ===== PROPERLY INDENTED DEBUG VISUALIZATION =====
    if st.checkbox("Show debug visualization (first 5 paths)"):
        fig_debug = go.Figure()
        
        # Plot raw paths
        for i in range(min(5, simulations['raw'].shape[1])):
            fig_debug.add_trace(go.Scatter(
                x=np.arange(simulations['raw'].shape[0]),
                y=simulations['raw'][:, i],
                mode='lines',
                name=f'Raw {i+1}',
                line=dict(color='blue', width=1)
            ))
        
        # Plot MA paths if available
        if 'ma' in simulations:
            for i in range(min(5, simulations['ma'].shape[1])):
                fig_debug.add_trace(go.Scatter(
                    x=np.arange(simulations['ma'].shape[0]),
                    y=simulations['ma'][:, i],
                    mode='lines',
                    name=f'MA {i+1}',
                    line=dict(color='green', width=1)
                ))
        
        # Plot WMA paths if available
        if 'wma' in simulations:
            for i in range(min(5, simulations['wma'].shape[1])):
                fig_debug.add_trace(go.Scatter(
                    x=np.arange(simulations['wma'].shape[0]),
                    y=simulations['wma'][:, i],
                    mode='lines',
                    name=f'WMA {i+1}',
                    line=dict(color='red', width=1)
                ))
        
        fig_debug.update_layout(
            title="Debug View: First 5 Paths Comparison",
            xaxis_title="Days",
            yaxis_title="Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_debug, use_container_width=True)
    # ===== END DEBUG VISUALIZATION =====

    if data.shape[1] == 0:
        st.warning(f"No data available for {smooth_type}. Try increasing simulation size or adjusting inputs.")
        return
    # ===== END DEBUG VISUALIZATION =====

    if DEBUG_MODE:  # Set this as a global variable or parameter
    fig = go.Figure()
    for i in range(min(5, n_simulations)):
        fig.add_trace(go.Scatter(y=raw_simulations[:, i], name=f'Raw {i}'))
        fig.add_trace(go.Scatter(y=ma_simulations[:, i], name=f'MA {i}'))
        fig.add_trace(go.Scatter(y=wma_simulations[:, i], name=f'WMA {i}'))
    fig.update_layout(title='First 5 Paths Comparison')
    st.plotly_chart(fig)

    if data.shape[1] == 0:
        st.warning(f"No data available for {smooth_type}. Try increasing simulation size or adjusting inputs.")
        return



def display_financial_ratios(ratios: Dict[str, Any], ticker: str):
    """
    Displays financial ratios from FMP API data
    Args:
        ratios: Dictionary from FMP's /v3/ratios endpoint
        ticker: Stock ticker symbol for display purposes
    """
    try:
        if not ratios:
            st.error("No ratio data available")
            return
        if 'go' not in globals():
            raise ImportError("Plotly graph_objects not imported properly")
            
        # Create the figure safely
        fig = go.Figure()  # This will now work

        # FMP field to display name mapping
        ratio_map = {
            'priceEarningsRatio': 'P/E Ratio',
            'priceToBookRatio': 'P/B Ratio',
            'debtEquityRatio': 'Debt/Equity',
            'currentRatio': 'Current Ratio',
            'returnOnEquity': 'ROE',
            'returnOnAssets': 'ROA'
        }

        # Mock sector averages (replace with actual FMP sector data)
        sector_avg = {
            'priceEarningsRatio': 15.2,
            'priceToBookRatio': 2.8,
            'debtEquityRatio': 0.85,
            'currentRatio': 1.5,
            'returnOnEquity': 0.15,
            'returnOnAssets': 0.075
        }

        # Prepare display data
        display_data = {}
        for api_key, display_name in ratio_map.items():
            if api_key in ratios and ratios[api_key] is not None:
                # Convert decimals to percentages for ROE/ROA
                if display_name in ['ROE', 'ROA']:
                    display_data[display_name] = f"{ratios[api_key] * 100:.2f}%"
                else:
                    display_data[display_name] = f"{ratios[api_key]:.2f}"

        if not display_data:
            st.error("No valid ratio data available for display")
            return

        # Create visualization
        st.subheader(f"Financial Ratios for {ticker}")
        
        # Bar chart
        fig = go.Figure()
        
        # Add company bars
        fig.add_trace(go.Bar(
            x=list(display_data.keys()),
            y=[float(v.strip('%')) if '%' in v else float(v) for v in display_data.values()],
            name=ticker,
            text=list(display_data.values()),
            textposition='auto'
        ))
        
        # Add sector average bars (only for available metrics)
        sector_x = []
        sector_y = []
        for display_name in display_data.keys():
            api_key = next(k for k, v in ratio_map.items() if v == display_name)
            if api_key in sector_avg:
                sector_x.append(display_name)
                if display_name in ['ROE', 'ROA']:
                    sector_y.append(sector_avg[api_key] * 1)
                else:
                    sector_y.append(sector_avg[api_key])
        
        fig.add_trace(go.Bar(
            x=sector_x,
            y=sector_y,
            name='Sector Average',
            text=[f"{y:.1f}{'%' if x in ['ROE', 'ROA'] else ''}" for x, y in zip(sector_x, sector_y)],
            textposition='auto'
        ))
        
        fig.update_layout(
            barmode='group',
            title=f"{ticker} vs Sector Averages",
            yaxis_title="Value"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metric analysis
        st.subheader("Metric Analysis")
        
        cols = st.columns(2)
        with cols[0]:
            if 'P/E Ratio' in display_data:
                pe = float(display_data['P/E Ratio'])
                st.metric("P/E Ratio", 
                         display_data['P/E Ratio'],
                         f"{'High' if pe > 20 else 'Normal' if pe > 10 else 'Low'} vs market")
            
            if 'Current Ratio' in display_data:
                cr = float(display_data['Current Ratio'])
                st.metric("Current Ratio", 
                         display_data['Current Ratio'],
                         "Strong" if cr > 2 else "Adequate" if cr > 1 else "Weak")
        
        with cols[1]:
            if 'Debt/Equity' in display_data:
                de = float(display_data['Debt/Equity'])
                st.metric("Debt/Equity", 
                         display_data['Debt/Equity'],
                         "High" if de > 1 else "Moderate" if de > 0.5 else "Low")
            
            if 'ROE' in display_data:
                roe = float(display_data['ROE'].strip('%'))
                st.metric("Return on Equity", 
                         display_data['ROE'],
                         "Strong" if roe > 15 else "Average" if roe > 8 else "Weak")

    except Exception as e:
        st.error(f"Error displaying ratios: {str(e)}")




def display_predictions(historical_data, predictions, model_name):
    fig = go.Figure()
    
    # Historical Data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        name='Historical Prices',
        line=dict(color='blue')
    ))
    
    # Predictions
    future_dates = pd.date_range(
        start=historical_data.index[-1],
        periods=len(predictions)+1
    )[1:]
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        name=f'{model_name} Forecast',
        line=dict(color='green', dash='dot')
    ))
    
    # Confidence Interval (if available)
    if hasattr(predictions, 'conf_int'):
        ci = predictions.conf_int()
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=ci.iloc[:, 0],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=ci.iloc[:, 1],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            name='Confidence Interval'
        ))
    
    fig.update_layout(
        title=f"{model_name} Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Metrics
    if len(historical_data) > 30:  # Only show if sufficient history
        test = historical_data['Close'].values[-30:]
        mae = mean_absolute_error(test, predictions[:30])
        st.metric("Mean Absolute Error (30-day backtest)", f"${mae:.2f}")

# Updated main app structure
def main():
    # All main() content indented 4 spaces
    st.sidebar.header("Navigation")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["Stock Analysis", "Monte Carlo", "Financial Ratios", "Predictions"]
    )
    
    # This line should have exactly 4 spaces of indentation
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").strip().upper()
    if not ticker:
        st.error("Please enter a valid ticker symbol")
        return

    
    # Date Range Selector
    period = st.sidebar.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    # Fetch Data
    
    try:
        data, error = get_stock_data(ticker, period)
        
        if data.empty:
            if error and "rate limit" in error.lower():
                st.error("⚠️ API rate limit reached. Please wait and try again.")
            else:
                st.error(f"Error fetching data: {error if error else 'Unknown error'}")
            return
            
    except Exception as e:
        st.error(f"Unexpected error during data fetch: {str(e)}")
        return
    
    # Analysis Sections
    try:
        if analysis_type == "Stock Analysis":
            display_stock_analysis(data, ticker)
            
        elif analysis_type == "Monte Carlo":
            st.header("🎲 Monte Carlo Simulation")
            n_simulations = st.slider("Number of Simulations", 100, 5000, 1000)
            time_horizon = st.slider("Time Horizon (days)", 30, 365, 180)
            
            if st.button("Run Simulation"):
                try:
                    simulations = monte_carlo_simulation(data, n_simulations, time_horizon)
                    display_monte_carlo(simulations)
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")
        
        elif analysis_type == "Financial Ratios":
            st.header("📈 Financial Ratios Analysis")

            # Create tabs for different ratio types
            tab1, tab2  = st.tabs(["Fundamental Ratios", "Market Risk Metrics"])
    
            with tab1:
                st.subheader("Fundamental Ratios")
                try:
                    ratios = get_yahoo_ratios(ticker)
            
                    if ratios:
                        # Get all three values properly
                        sector, industry, peers = get_sector_peers(ticker)
                
                        fundamental_ratios = {
                            'priceEarningsRatio': ratios.get('priceEarningsRatio'),
                            'priceToBookRatio': ratios.get('priceToBookRatio'),
                            'debtEquityRatio': ratios.get('debtEquityRatio'),
                            'currentRatio': ratios.get('currentRatio'),
                            'returnOnEquity': ratios.get('returnOnEquity'),
                            'returnOnAssets': ratios.get('returnOnAssets')
                        }
                
                        if any(v is not None for v in fundamental_ratios.values()):
                            display_financial_ratios(fundamental_ratios, ticker)
                        else:
                            st.warning("No fundamental ratio data available")
            
                    # Show sector context if available
                    if sector and industry:
                        st.caption(f"Sector: {sector} | Industry: {industry}")
                    if peers:
                        st.caption(f"Peers: {', '.join(peers[:3])}{'...' if len(peers) > 3 else ''}")
                
                except Exception as e:
                    st.error(f"Fundamental ratios analysis failed: {str(e)}")
    
            with tab2:
                st.subheader("🎯 Risk Metrics Analysis")
                try:
                    with st.spinner("Calculating risk metrics..."):
                        risk_metrics = calculate_risk_metrics(data)
            
                        if risk_metrics:
                            # Create a metrics dashboard
                            st.markdown("### 📉 Risk Profile Summary")
                
                            # Main metrics in columns
                            m1, m2, m3 = st.columns(3)
                
                            # Column 1: Annual Volatility
                            with m1:
                                st.metric(
                                    "Annual Volatility", 
                                     f"{risk_metrics.get('volatility', 0):.2%}",
                                     help="1-year standard deviation of returns"
                                )
                                st.progress(
                                    min(risk_metrics.get('volatility', 0)/0.5,1.0) 
                                )
                                st.caption("🛈 <0.5 = Low, >1.0 = High")
                
                            # Column 2: Maximum Drawdown
                            with m2:
                                st.metric(
                                    "Max Drawdown", 
                                    f"{risk_metrics.get('maximumDrawdown', 0):.2%}",
                                    help="Worst historical peak-to-trough decline"
                                )
                                st.progress(
                                    min(abs(risk_metrics.get('maximumDrawdown', 0))/0.5,1.0)                                    
                                )
                                st.caption("🛈 <10% = Low, >30% = High")
                
                            # Column 3: Sharpe Ratio
                            with m3:
                                sharpe = risk_metrics.get('sharpeRatio', 0)
                                st.metric(
                                    "Sharpe Ratio", 
                                    f"{sharpe:.2f}",
                                    delta="Good" if sharpe > 1 else "Fair" if sharpe > 0 else "Poor",
                                    help="Risk-adjusted returns (0 risk-free rate)"
                                )
                                st.progress(
                                    (sharpe+1)/3, 
                                    text="<0 = Poor, >1 = Good"
                                )
                
                            # Visualizations
                            st.markdown("### 📊 Risk Over Time")
                            if not data.empty and 'Close' in data.columns:
                                # Volatility chart
                                c1, c2 = st.columns(2)
                    
                                with c1:
                                    st.markdown("#### 30-Day Rolling Volatility")
                                    returns = np.log(data['Close']).diff()
                                    rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=rolling_vol.index,
                                        y=rolling_vol,
                                        mode='lines',
                                        line=dict(color='#FF4B4B', width=2),
                                        name='Volatility'
                                    ))
                                    fig.update_layout(
                                        yaxis_tickformat=".0%",
                                        hovermode="x",
                                        height=300,
                                        margin=dict(t=30)
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    
                                with c2:
                                    st.markdown("#### Cumulative Drawdown")
                                    rolling_max = data['Close'].cummax()
                                    daily_drawdown = data['Close']/rolling_max - 1
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=daily_drawdown.index,
                                        y=daily_drawdown,
                                        fill='tozeroy',
                                        fillcolor='rgba(255, 75, 75, 0.3)',
                                        line=dict(color='#FF4B4B'),
                                        name='Drawdown'
                                    ))
                                    fig.update_layout(
                                        yaxis_tickformat=".0%",
                                        hovermode="x",
                                        height=300,
                                        margin=dict(t=30)
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not calculate risk metrics")
    
                except Exception as e:
                    st.error(f"Risk analysis failed: {str(e)}")

           
        
        elif analysis_type == "Predictions":
            st.header("🔮 Price Predictions")
    
            col1, col2 = st.columns(2)
    
            with col1:
                model_type = st.    selectbox(
                    "Select Prediction Model",
                    ["Holt-Winters", "Arima", "LSTM", "Random Forest", "XGBoost"]
                )
            seasonal_periods = 5
            if model_type == "Holt-Winters":
                with col2:
                    seasonality_choice = st.radio(
                        "Seasonality",
                        ["Weekly (5)", "Monthly (21)", "Quarterly (63)"],
                        horizontal=True
                    )
                    seasonal_periods = int(seasonality_choice.split("(")[1].replace(")", ""))
    
            if st.button("Generate Predictions"):
                with st.spinner(f"Training {model_type} model..."):
                    try:
                        if model_type == "Holt-Winters":
                            model, error = train_holt_winters(data, seasonal_periods)
                            if model is None:
                                st.error(error)
                            else:
                                predictions = predict_holt_winters(model, 30)
                                display_predictions(data, predictions, "Holt-Winters")
            
                        elif model_type == "Arima":
                            model = train_arima_model(data)
                            predictions = predict_arima(model, 30)
                            display_predictions(data, predictions, "Arima")
            
                        elif model_type == "Random Forest":
                            model = train_random_forest(data)
                            predictions = predict_random_forest(model, data, 30)
                            display_predictions(data, predictions, "Random Forest")

                            # Show feature importance
                            try:
                                importances = model.feature_importances_
                                features = [f"Day-{i}" for i in range(1, 31)]
                                fig = go.Figure([go.Bar(
                                    x=features, 
                                    y=importances,
                                    marker_color='#636EFA'
                                )])
                                fig.update_layout(
                                    title="Feature Importance (Which Past Days Matter Most)",
                                    xaxis_title="Days Back",
                                    yaxis_title="Importance Score",
                                    hovermode="x"
                                )    
                                st.plotly_chart(fig)
                            except Exception as e:
                                st.warning(f"Couldn't generate feature importance: {str(e)}")
            
                        elif model_type == "LSTM":
                            model, scaler = train_lstm_model(data)
                            predictions = predict_lstm(model, scaler, data, 30)
                            display_predictions(data, predictions, "LSTM")
            
                        elif model_type == "XGBoost":
                            model = train_xgboost_model(data)
                            predictions = predict_xgboost(model, data, 30)
                            display_predictions(data, predictions, "XGBoost")
        
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        if "Random Forest" in str(e):
                            st.info("Try with at least 60 days of historical data")
                        elif "Prophet" in str(e):
                            st.info("Check your date format (YYYY-MM-DD required)")
                        elif "LSTM" in str(e):
                            st.info("Try reducing the lookback window or using more data")
                        elif "XGBoost" in str(e):
                            st.info("Ensure no missing values in your historical data")
    except Exception as e :
        st.error(f"Application error: {str(e)}")
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application crashed: {str(e)}")
        import traceback 
        st.text(traceback.format_exc())
