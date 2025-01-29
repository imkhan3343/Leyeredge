# memecoin_bot.py
import os
import re
import sqlite3
import time
import asyncio
import smtplib
import numpy as np
import pandas as pd
import requests
import tweepy
import aiohttp
from dotenv import load_dotenv
from email.message import EmailMessage
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solana.rpc.types import TxOpts
from solana.rpc.core import RPCException
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "db_name": "memecoins.db",
    "solana_rpc": os.getenv("SOLANA_RPC", "https://api.mainnet-beta.solana.com"),
    "check_interval": 3600,  # 1 hour
    "max_slippage": 0.3,
    "profit_target": 15,
    "trade_size": 1_000_000_000  # 1 SOL in lamports
}

# Initialize clients
twitter_client = tweepy.Client(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))
solana_client = AsyncClient(CONFIG["solana_rpc"])

# Database setup
def create_tables():
    """Initialize database schema"""
    conn = sqlite3.connect(CONFIG["db_name"])
    c = conn.cursor()
    
    # Memecoins and contracts
    c.execute('''
        CREATE TABLE IF NOT EXISTS memecoins (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            price REAL,
            market_cap REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            tweet_count INTEGER DEFAULT 0
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS contracts (
            contract_address TEXT PRIMARY KEY,
            symbol TEXT,
            name TEXT,
            kol_username TEXT,
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_checked DATETIME,
            is_on_pumpfun BOOLEAN DEFAULT 0,
            is_on_gmgn BOOLEAN DEFAULT 0,
            tweet_count INTEGER DEFAULT 0
        )
    ''')
    
    # Historical data and predictions
    c.execute('''
        CREATE TABLE IF NOT EXISTS historical_data (
            symbol TEXT,
            timestamp DATETIME,
            price REAL,
            volume REAL,
            market_cap REAL,
            tweet_count INTEGER,
            PRIMARY KEY (symbol, timestamp)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            symbol TEXT,
            timestamp DATETIME,
            predicted_change REAL,
            score REAL,
            confidence REAL,
            PRIMARY KEY (symbol, timestamp)
        )
    ''')
    
    # Trading records
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_address TEXT,
            amount REAL,
            entry_price REAL,
            target_price REAL,
            status TEXT DEFAULT 'open',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(contract_address) REFERENCES contracts(contract_address)
        )
    ''')
    
    conn.commit()
    conn.close()

# Core functionality
class MemecoinTracker:
    def fetch_gmgn_memecoins(self):
        """Fetch top Solana memecoins from GMGN API"""
        url = 'https://api.gmgn.ai/defi/quotes/v1/top'
        params = {'chain': 'solana', 'category': 'memecoin'}
        headers = {'Authorization': f'Bearer {os.getenv("GMGN_API_KEY")}'}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            return response.json().get('results', [])
        except Exception as e:
            print(f"GMGN error: {e}")
            return []

    def process_twitter_mentions(self):
        """Scan Twitter for memecoin mentions"""
        symbols = self.get_db_symbols()
        for symbol in symbols:
            count = self.get_tweet_count(symbol)
            self.update_tweet_count(symbol, count)

    def get_tweet_count(self, symbol):
        """Count recent tweets mentioning a symbol"""
        try:
            response = twitter_client.get_recent_tweets_count(
                query=f'${symbol} -is:retweet',
                granularity='hour'
            )
            return response.data[0]['tweet_count'] if response.data else 0
        except tweepy.TweepyException as e:
            print(f"Twitter error: {e}")
            return 0

    def update_tweet_count(self, symbol, count):
        """Update database with new tweet count"""
        conn = sqlite3.connect(CONFIG["db_name"])
        c = conn.cursor()
        c.execute('UPDATE memecoins SET tweet_count = ? WHERE symbol = ?', (count, symbol))
        conn.commit()
        conn.close()

    def get_db_symbols(self):
        """Retrieve tracked symbols from database"""
        conn = sqlite3.connect(CONFIG["db_name"])
        c = conn.cursor()
        c.execute('SELECT symbol FROM memecoins')
        symbols = [row[0] for row in c.fetchall()]
        conn.close()
        return symbols

# Prediction system
class PricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        """Load trained model"""
        try:
            saved = load('forecaster.joblib')
            self.model = saved['model']
            self.scaler = saved['scaler']
        except:
            self.model = None

    def predict(self, token_data):
        """Make price prediction"""
        if not self.model:
            return None
            
        features = np.array([
            token_data['price_change_24h'],
            token_data['volume_change_24h'],
            token_data['tweet_count_24h'],
            token_data['kol_mentions_7d'],
            token_data['market_cap'],
            token_data['volatility_7d']
        ]).reshape(1, -1)
        
        return self.model.predict(self.scaler.transform(features))[0]

# Trading engine
class TradingEngine:
    def __init__(self):
        self.wallet = Keypair.from_bytes(bytes.fromhex(os.getenv("WALLET_PRIVATE_KEY")))
        self.priority_fee = int(os.getenv("PRIORITY_FEE_MICRO_LAMPORTS", 100000))

    async def execute_trade(self, contract_address: str, buy: bool):
        """Execute buy/sell order"""
        try:
            txn = Transaction()
            # Add swap logic here using Jupiter API
            # Placeholder for actual swap implementation
            result = await solana_client.send_transaction(
                txn,
                self.wallet,
                opts=TxOpts(skip_preflight=False)
            )
            return result.value
        except RPCException as e:
            print(f"Trade failed: {e}")
            return None

# Main application
class MemecoinBot:
    def __init__(self):
        self.tracker = MemecoinTracker()
        self.predictor = PricePredictor()
        self.trader = TradingEngine()
        create_tables()

    async def run(self):
        """Main execution loop"""
        while True:
            # Update market data
            self.update_market_data()
            
            # Process social metrics
            self.tracker.process_twitter_mentions()
            
            # Make predictions
            self.generate_predictions()
            
            # Check alerts
            self.check_alerts()
            
            # Execute trades
            await self.execute_auto_trades()
            
            # Sleep until next cycle
            await asyncio.sleep(CONFIG["check_interval"])

    def update_market_data(self):
        """Fetch latest market data"""
        memecoins = self.tracker.fetch_gmgn_memecoins()
        conn = sqlite3.connect(CONFIG["db_name"])
        c = conn.cursor()
        for coin in memecoins:
            c.execute('''
                INSERT OR REPLACE INTO memecoins 
                (symbol, name, price, market_cap)
                VALUES (?, ?, ?, ?)
            ''', (coin['symbol'], coin['name'], coin['price'], coin['market_cap']))
        conn.commit()
        conn.close()

    def generate_predictions(self):
        """Generate price predictions"""
        conn = sqlite3.connect(CONFIG["db_name"])
        query = '''
            SELECT symbol, price, market_cap, tweet_count 
            FROM memecoins
        '''
        data = pd.read_sql(query, conn)
        conn.close()
        
        # Generate predictions and store in DB
        # (Add proper feature engineering here)

    def check_alerts(self):
        """Check for alert conditions"""
        # Implement alert logic

    async def execute_auto_trades(self):
        """Execute automated trades"""
        # Implement trading logic

# Run the bot
if __name__ == '__main__':
    bot = MemecoinBot()
    asyncio.run(bot.run())