# Snake AI: Reinforcement Learning vs. Heuristic Baseline

## 📌 Project Overview
This project implements and compares two different Artificial Intelligence agents playing the classic game of Snake:
1.  **Baseline Agent:** A rule-based heuristic agent that follows simple survival logic.
2.  **DQN Agent:** A Reinforcement Learning agent using a Deep Q-Network (Deep Q-Learning) implemented in PyTorch.

The goal is to demonstrate the difference between hard-coded logic and learned behavior in a grid-world environment.

## 🚀 Features
* **Custom Environment:** A PyGame-based Snake environment optimized for AI interfacing.
* **Deep Q-Learning:** Implementation of a Linear QNet with Experience Replay.
* **Visual Demos:**
    * **Phase 1:** High-speed training loop (generates learning curve).
    * **Phase 2:** Baseline Agent demo (fast speed).
    * **Phase 3:** Trained DQN Agent demo (presentation speed).
* **Data Visualization:** Automatically generates a `final_learning_curve.png` to visualize training performance.

## 🛠️ Installation & Requirements
Ensure you have Python 3.8+ installed. Install the dependencies using pip:

```bash
pip install pygame numpy torch matplotlib