"""
Improved PPO for Stock Trading
Phase 2 Implementation
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPROVEMENT 1: Robust Feature Normalizer
# ============================================================================

class RobustFeatureNormalizer:
    """
    Improved normalization with outlier handling and running statistics
    """
    def __init__(self, clip_range=3.0, momentum=0.99):
        self.clip_range = clip_range
        self.momentum = momentum
        self.running_mean = None
        self.running_std = None
        self.epsilon = 1e-8

    def normalize(self, features):
        """Normalize features with robust statistics"""
        if self.running_mean is None:
            self.running_mean = np.mean(features, axis=0)
            self.running_std = np.std(features, axis=0)
        else:
            # Exponential moving average
            self.running_mean = (self.momentum * self.running_mean +
                                 (1 - self.momentum) * np.mean(features, axis=0))
            self.running_std = (self.momentum * self.running_std +
                                (1 - self.momentum) * np.std(features, axis=0))

        # Normalize and clip outliers
        normalized = (features - self.running_mean) / (self.running_std + self.epsilon)
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return normalized

# ============================================================================
# IMPROVEMENT 2: Advanced Technical Indicators
# ============================================================================

class TechnicalIndicators:
    """
    Multi-timeframe technical indicators for stock trading
    """

    @staticmethod
    def compute_rsi(prices, window=14):
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def compute_macd(prices, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    @staticmethod
    def compute_bollinger_bands(prices, window=20, num_std=2):
        """Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return bb_position

    @staticmethod
    def compute_atr(high, low, close, window=14):
        """Average True Range (volatility indicator)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    @staticmethod
    def add_all_indicators(df):
        """Add all technical indicators to dataframe"""
        df = df.copy()

        # Price-based features
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)

        # RSI at different timeframes
        df['rsi_14'] = TechnicalIndicators.compute_rsi(df['close'], 14)
        df['rsi_28'] = TechnicalIndicators.compute_rsi(df['close'], 28)

        # MACD
        df['macd'] = TechnicalIndicators.compute_macd(df['close'])

        # Bollinger Bands
        df['bb_position'] = TechnicalIndicators.compute_bollinger_bands(df['close'])

        # ATR (volatility)
        if 'high' in df.columns and 'low' in df.columns:
            df['atr'] = TechnicalIndicators.compute_atr(
                df['high'], df['low'], df['close']
            )

        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_ratio'] = df['sma_10'] / df['sma_50']

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)

        return df

# ============================================================================
# IMPROVEMENT 3: Enhanced Stock Trading Environment with Risk-Aware Rewards
# ============================================================================

class ImprovedStockTradingEnv(gym.Env):
    """
    Enhanced stock trading environment with:
    - Risk-adjusted reward function
    - Transaction cost modeling
    - Multi-timeframe features
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_amount=100000, transaction_cost_pct=0.001,
                 lookback_window=60, reward_scaling=1e-4):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling

        # Identify stocks in dataset
        self.stock_list = [col.split('_')[0] for col in df.columns
                          if col.endswith('_close')]
        self.num_stocks = len(self.stock_list)

        # Feature normalizer
        self.normalizer = RobustFeatureNormalizer()

        # Define action space: continuous actions for each stock
        # -1 = sell all, 0 = hold, +1 = buy max
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(self.num_stocks,),
            dtype=np.float32
        )

        # Define observation space
        self.feature_columns = [col for col in df.columns if col != 'date']
        state_dim = len(self.feature_columns) + self.num_stocks + 2
        # Features + holdings + cash + portfolio_value

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Initialize state variables
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.num_stocks)
        self.portfolio_value = self.initial_amount
        self.previous_portfolio_value = self.initial_amount

        # Track returns for volatility calculation
        self.returns_history = []

        # Trading statistics
        self.num_trades = 0
        self.total_transaction_cost = 0
        self.last_transaction_cost = 0

        # Episode tracking
        self.portfolio_history = [self.initial_amount]

        return self._get_observation()

    def _get_observation(self):
        """Get current state observation"""
        # Get current row features
        current_row = self.df.iloc[self.current_step]
        features = current_row[self.feature_columns].values.astype(np.float32)

        # Normalize features
        features = self.normalizer.normalize(features.reshape(1, -1)).flatten()

        # Portfolio state
        holdings_value = self._get_holdings_value()
        portfolio_state = np.array([
            self.cash / self.initial_amount,  # Normalized cash
            holdings_value / self.initial_amount,  # Normalized holdings value
        ], dtype=np.float32)

        # Holdings ratio for each stock
        holdings_ratio = self.holdings / (self.holdings.sum() + 1e-8)

        # Concatenate all state components
        observation = np.concatenate([
            features,
            holdings_ratio,
            portfolio_state
        ])

        return observation

    def _get_holdings_value(self):
        """Calculate current value of stock holdings"""
        current_row = self.df.iloc[self.current_step]
        holdings_value = 0

        for i, stock in enumerate(self.stock_list):
            price = current_row[f'{stock}_close']
            holdings_value += self.holdings[i] * price

        return holdings_value

    def _execute_trades(self, actions):
        """Execute trading actions"""
        current_row = self.df.iloc[self.current_step]
        self.last_transaction_cost = 0

        for i, stock in enumerate(self.stock_list):
            action = actions[i]
            price = current_row[f'{stock}_close']

            if action > 0.1:  # Buy
                # Calculate maximum shares we can buy
                max_shares = self.cash / (price * (1 + self.transaction_cost_pct))
                shares_to_buy = max_shares * action

                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    transaction_cost = cost * self.transaction_cost_pct
                    total_cost = cost + transaction_cost

                    if total_cost <= self.cash:
                        self.holdings[i] += shares_to_buy
                        self.cash -= total_cost
                        self.last_transaction_cost += transaction_cost
                        self.num_trades += 1

            elif action < -0.1:  # Sell
                shares_to_sell = self.holdings[i] * abs(action)

                if shares_to_sell > 0:
                    revenue = shares_to_sell * price
                    transaction_cost = revenue * self.transaction_cost_pct
                    net_revenue = revenue - transaction_cost

                    self.holdings[i] -= shares_to_sell
                    self.cash += net_revenue
                    self.last_transaction_cost += transaction_cost
                    self.num_trades += 1

        self.total_transaction_cost += self.last_transaction_cost

    def _calculate_reward(self):
        """
        IMPROVEMENT: Risk-adjusted reward function
        R(t) = alpha * returns - beta * volatility - gamma * transaction_costs
        """
        # Calculate portfolio value
        holdings_value = self._get_holdings_value()
        self.portfolio_value = self.cash + holdings_value

        # Calculate returns
        returns = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value

        # Calculate volatility (risk penalty)
        self.returns_history.append(returns)
        if len(self.returns_history) > 20:
            self.returns_history.pop(0)

        volatility = np.std(self.returns_history) if len(self.returns_history) > 1 else 0

        # Transaction cost component
        transaction_cost_ratio = self.last_transaction_cost / self.previous_portfolio_value

        # Risk-adjusted reward with tuned parameters
        alpha = 1.0  # Return weight
        beta = 0.5   # Risk penalty
        gamma = 0.001  # Transaction cost penalty

        reward = alpha * returns - beta * volatility - gamma * transaction_cost_ratio

        # Scale reward for stable learning
        reward = reward * self.reward_scaling

        # Update for next step
        self.previous_portfolio_value = self.portfolio_value
        self.portfolio_history.append(self.portfolio_value)

        return reward

    def step(self, actions):
        """Execute one time step"""
        # Clip actions to valid range
        actions = np.clip(actions, -1, 1)

        # Execute trades
        self._execute_trades(actions)

        # Calculate reward
        reward = self._calculate_reward()

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= len(self.df) - 1

        # Get new observation
        obs = self._get_observation() if not done else self._get_observation()

        # Info dictionary
        info = self._get_info()

        return obs, reward, done, info

    def _get_info(self):
        """Get episode information"""
        if len(self.portfolio_history) > 1:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            sharpe_ratio = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)

            # Calculate max drawdown
            peak = np.maximum.accumulate(self.portfolio_history)
            drawdown = (self.portfolio_history - peak) / peak
            max_drawdown = np.min(drawdown)
        else:
            sharpe_ratio = 0
            max_drawdown = 0

        total_return = (self.portfolio_value - self.initial_amount) / self.initial_amount

        return {
            'portfolio_value': self.portfolio_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': self.num_trades,
            'total_transaction_cost': self.total_transaction_cost
        }

    def render(self, mode='human'):
        """Render environment state"""
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Cash: ${self.cash:,.2f}")
        print(f"Holdings Value: ${self._get_holdings_value():,.2f}")
        print(f"Total Return: {((self.portfolio_value - self.initial_amount) / self.initial_amount) * 100:.2f}%")

# ============================================================================
# IMPROVEMENT 4: Training Callback with Metrics Logging
# ============================================================================

class TradingMetricsCallback(BaseCallback):
    """
    Callback for logging trading-specific metrics during training
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_returns = []
        self.episode_sharpe = []
        self.episode_drawdown = []

    def _on_step(self):
        """Called at each step"""
        # Check if episode finished
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]

            # Store metrics
            self.episode_returns.append(info.get('total_return', 0))
            self.episode_sharpe.append(info.get('sharpe_ratio', 0))
            self.episode_drawdown.append(info.get('max_drawdown', 0))

            # Print progress every 10 episodes
            if len(self.episode_returns) % 10 == 0:
                avg_return = np.mean(self.episode_returns[-10:])
                avg_sharpe = np.mean(self.episode_sharpe[-10:])

                print(f"\n[Episode {len(self.episode_returns)}]")
                print(f"  Avg Return (last 10): {avg_return*100:.2f}%")
                print(f"  Avg Sharpe (last 10): {avg_sharpe:.3f}")

        return True

# ============================================================================
# IMPROVEMENT 5: Adaptive Clipping and Entropy Functions
# ============================================================================

def adaptive_clip_range(progress_remaining):
    """
    Adaptive clipping range that decreases during training
    progress_remaining: 1.0 at start, 0.0 at end
    """
    epsilon_start = 0.2
    epsilon_min = 0.05
    # Quadratic decay
    epsilon = epsilon_start * max(epsilon_min / epsilon_start, progress_remaining ** 2)
    return epsilon

def adaptive_entropy_coef(progress_remaining):
    """
    Adaptive entropy coefficient for exploration-exploitation tradeoff
    """
    return 0.01 * (progress_remaining ** 2)

# ============================================================================
# Main Training Function
# ============================================================================

def train_improved_ppo(train_df, total_timesteps=1_000_000, n_envs=4):
    """
    Train improved PPO model with all enhancements
    """
    print("Initializing improved PPO training...")
    print(f"Number of parallel environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")

    # Create environment factory
    def make_env(rank, seed=0):
        def _init():
            env = ImprovedStockTradingEnv(train_df)
            env.seed(seed + rank)
            return env
        return _init

    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # Policy network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Actor: 3 layers
            vf=[256, 256, 128]   # Critic: 3 layers
        ),
        activation_fn=nn.Tanh,
        ortho_init=True
    )

    # Create PPO model with improvements
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=adaptive_clip_range,  # Adaptive clipping
        ent_coef=adaptive_entropy_coef,   # Adaptive entropy
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/"
    )

    # Create callback
    callback = TradingMetricsCallback()

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10
    )

    # Save model
    model.save("improved_ppo_stock_trading")
    print("\nModel saved as 'improved_ppo_stock_trading'")

    return model

# ============================================================================
# Evaluation Function
# ============================================================================

def evaluate_model(model, test_df, n_episodes=1):
    """
    Evaluate trained model on test data
    """
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)

    env = ImprovedStockTradingEnv(test_df)

    all_returns = []
    all_sharpe = []
    all_drawdown = []
    all_portfolio_values = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        # Collect metrics
        all_returns.append(info['total_return'])
        all_sharpe.append(info['sharpe_ratio'])
        all_drawdown.append(info['max_drawdown'])
        all_portfolio_values.append(info['portfolio_value'])

    # Calculate statistics
    avg_return = np.mean(all_returns) * 100
    avg_sharpe = np.mean(all_sharpe)
    avg_drawdown = np.mean(all_drawdown) * 100
    avg_portfolio_value = np.mean(all_portfolio_values)

    print(f"\nTest Results (avg over {n_episodes} episode(s)):")
    print(f"  Cumulative Return: {avg_return:.2f}%")
    print(f"  Sharpe Ratio: {avg_sharpe:.3f}")
    print(f"  Max Drawdown: {avg_drawdown:.2f}%")
    print(f"  Final Portfolio Value: ${avg_portfolio_value:,.2f}")
    print(f"  Initial Investment: $100,000")
    print(f"  Profit/Loss: ${avg_portfolio_value - 100000:,.2f}")

    return {
        'return': avg_return,
        'sharpe': avg_sharpe,
        'drawdown': avg_drawdown,
        'portfolio_value': avg_portfolio_value
    }

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Improved PPO Stock Trading - Phase 2")
    print("="*50)

    # Note: You need to load your actual data here
    # This is a placeholder showing the expected format

    """
    # Example data loading:
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')

    # Add technical indicators
    train_df = TechnicalIndicators.add_all_indicators(train_df)
    test_df = TechnicalIndicators.add_all_indicators(test_df)

    # Train model
    model = train_improved_ppo(train_df, total_timesteps=500_000, n_envs=4)

    # Evaluate model
    results = evaluate_model(model, test_df)
    """

    print("\nTo use this code:")
    print("1. Load your stock data (with OHLCV columns)")
    print("2. Add technical indicators using TechnicalIndicators.add_all_indicators()")
    print("3. Call train_improved_ppo() to train")
    print("4. Call evaluate_model() to test")
    print("\nAll improvements are integrated:")
    print("  ✓ Risk-adjusted reward function")
    print("  ✓ Adaptive clipping range")
    print("  ✓ Multi-timeframe technical indicators")
    print("  ✓ Robust feature normalization")
    print("  ✓ Parallel environment training")
    print("  ✓ Enhanced network architecture")
