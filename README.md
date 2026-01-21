# HumanED Snake Reinforcement Learning Project

This project implements a custom Snake game environment using Gymnasium for reinforcement learning, along with scripts for training and testing AI agents using Stable Baselines3. It includes both a human-playable Snake game and an RL environment for training models like PPO.

## Features

- **Custom Snake Environment**: [`SnekEnv`](snakeenv.py) class that follows the Gymnasium interface, suitable for RL training.
- **Human-Playable Game**: [`snakegame.py`](snakegame.py) allows manual control of the Snake game using keyboard inputs.
- **Training Scripts**: [`p4-snek-learn.py`](p4-snek-learn.py) for training the Snake agent, and [`p1-intro.py`](p1-intro.py), [`p2-save.py`](p2-save.py), [`p2-load.py`](p2-load.py) for LunarLander examples.
- **Environment Checking**: [`checkenv.py`](checkenv.py) and [`doublecheckenv.py`](doublecheckenv.py) for validating the custom environment.
- **Model Visualization**: [`p5-model-show.py`](p5-model-show.py) for demonstrating trained models.

## Requirements

- Python 3.8+
- Gymnasium
- Stable Baselines3
- OpenCV
- NumPy
- Matplotlib (for logging)

Install dependencies using the provided [`environment.yml`](environment.yml) for Conda:

```bash
conda env create -f environment.yml
conda activate snakeEnv
```

## Installation

1. Clone the repository.
2. Set up the Conda environment as above.
3. Ensure all scripts are executable.

## Usage

### Running the Human-Playable Game

Execute [`snakegame.py`](snakegame.py) to play the Snake game manually:

```bash
python snakegame.py
```

Controls: Use 'a', 'd', 'w', 's' for left, right, up, down. Press 'q' to quit.

### Training the Snake Agent

Run [`p4-snek-learn.py`](p4-snek-learn.py) to train a PPO model on the Snake environment:

```bash
python p4-snek-learn.py
```

Models are saved in the `models/` directory, and logs in `logs/`.

### Checking the Environment

Use [`checkenv.py`](checkenv.py) to validate the [`SnekEnv`](snakeenv.py):

```bash
python checkenv.py
```

### Testing the Environment

Run [`doublecheckenv.py`](doublecheckenv.py) for a quick test with random actions:

```bash
python doublecheckenv.py
```

### LunarLander Examples

- [`p1-intro.py`](p1-intro.py): Basic PPO training on LunarLander.
- [`p2-save.py`](p2-save.py): Train and save models.
- [`p2-load.py`](p2-load.py): Load and run a trained model.

## Project Structure

- [`snakeenv.py`](snakeenv.py): Custom Gym environment for Snake.
- [`snakegame.py`](snakegame.py): Human-playable Snake game.
- [`p1-intro.py`](p1-intro.py): Intro to RL with LunarLander.
- [`p2-save.py`](p2-save.py): Save trained models.
- [`p2-load.py`](p2-load.py): Load and evaluate models.
- [`p4-snek-learn.py`](p4-snek-learn.py): Train Snake agent.
- [`p5-model-show.py`](p5-model-show.py): Show trained Snake model.
- [`checkenv.py`](checkenv.py): Environment checker.
- [`doublecheckenv.py`](doublecheckenv.py): Environment tester.
- [`environment.yml`](environment.yml): Conda environment file.
- `models/`: Directory for saved models.
- `logs/`: Directory for TensorBoard logs.

## Notes

- The Snake environment aims for a snake length goal of 30.
- Rewards are based on proximity to the apple and penalties for collisions.
- Based on tutorials from YouTube (links in code comments).

For more details, refer to the code comments and Gymnasium documentation.

## Source: 

- [Video](https://www.youtube.com/watch?v=XbWhJdQgi7E) by Sentdex 



