# Automated Stock Trading using PPO

A deep reinforcement learning project that uses Proximal Policy Optimization (PPO) to automate stock trading decisions. This project improves upon baseline PPO implementation with advanced features and optimizations.

## Project Overview

This project implements an automated stock trading agent using PPO, a state-of-the-art reinforcement learning algorithm. The agent learns to make buy/sell/hold decisions by interacting with historical stock market data.

**Course:** Reinforcement Learning - Phase 2  
**Institution:** FAST School of Computing  
**Team Members:** Maaz Ud Din, Saamer Abbas, Sammar Kaleem, Ali Hassan

### Baseline Work

The project builds upon a standard PPO implementation using Stable-Baselines3. The baseline implementation included:
- Basic PPO algorithm with fixed hyperparameters (ε = 0.2)
- Simple reward function based on portfolio value changes
- Daily OHLCV data with basic technical indicators
- Single environment training
- Standard neural network architecture

**Phase 2 Focus:** We enhanced this baseline with algorithmic improvements, advanced features, and training optimizations to achieve significantly better performance.

## What's Inside

### Main Files
- `improved_ppo_trading.py` - Enhanced PPO implementation with all improvements
- `baseline_vs_improved_comparison.py` - Script to compare baseline vs improved performance
- `gradio_dashboard.py` - Interactive web dashboard for visualizing results
- `comparison_report.txt` - Detailed performance comparison report

### Folders
- `description_markdowns/` - Detailed documentation, reports, and guides
- `graphs_and_images/` - Performance charts and visualizations
- `submission/` - Final submission files including research paper and report

## Key Improvements Over Baseline

### 1. Algorithmic Enhancements
- **Adaptive Clipping Range**: Dynamic epsilon decay (0.2 → 0.05) for better convergence
- **Risk-Adjusted Rewards**: Balances returns, volatility, and transaction costs
- **Multi-Timeframe Features**: Incorporates 5-min, 1-hour, and daily indicators
- **Dynamic Entropy Coefficient**: Improved exploration-exploitation balance

### 2. Technical Improvements
- **Advanced Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Robust Normalization**: Handles outliers with running statistics
- **Parallel Training**: 8 concurrent environments for faster training
- **Enhanced Architecture**: Deeper neural networks [256, 256, 128] layers

### 3. Training Optimizations
- Vectorized environments (SubprocVecEnv)
- Orthogonal weight initialization
- Gradient clipping for stability
- Memory-efficient data pipeline

## Performance Results

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| Cumulative Return | 42.30% | 58.70% | +38.8% |
| Sharpe Ratio | 1.23 | 1.51 | +22.8% |
| Max Drawdown | -18.4% | -12.1% | +34.2% |
| Win Rate | 54.20% | 61.80% | +14.0% |
| Training Time | 14.3h | 2.1h | -85.3% |
| Final Portfolio | $142,300 | $158,700 | +11.5% |

## Getting Started

### Prerequisites
```bash
pip install numpy pandas gym torch stable-baselines3 gradio
```

### Run the Trading Agent
```bash
python improved_ppo_trading.py
```

### View Interactive Dashboard
```bash
python gradio_dashboard.py
```
Then open your browser to `http://localhost:7860`

### Generate Comparison Report
```bash
python baseline_vs_improved_comparison.py
```

## How It Works

1. **Environment**: Custom Gym environment simulates stock trading with realistic constraints
2. **State Space**: Price data, technical indicators, portfolio status, and market features
3. **Action Space**: Buy, Sell, or Hold decisions
4. **Reward Function**: Risk-adjusted returns considering volatility and transaction costs
5. **Training**: PPO algorithm learns optimal trading policy through trial and error

## Project Structure

```
Automated_Stock_using_PPO/
├── improved_ppo_trading.py          # Main implementation
├── gradio_dashboard.py              # Interactive dashboard
├── baseline_vs_improved_comparison.py
├── comparison_report.txt
├── description_markdowns/           # Documentation
│   ├── Phase2_Report.md
│   ├── START_HERE.md
│   └── ...
├── graphs_and_images/               # Performance charts
└── submission/                      # Final deliverables
    ├── Research paper.pdf
    └── Phase2_Report.pdf
```

## Documentation

For detailed information, check out:
- `description_markdowns/START_HERE.md` - Quick start guide
- `description_markdowns/Phase2_Report.md` - Complete technical report
- `comparison_report.txt` - Performance analysis

## Results Visualization

The project includes several performance charts:
- Performance comparison (baseline vs improved)
- Risk analysis and drawdown comparison
- Training efficiency metrics
- Improvement percentages across all metrics

## License

This is an academic project for educational purposes.

## Acknowledgments

Built upon the Stable-Baselines3 PPO implementation and inspired by research in deep reinforcement learning for financial trading.
