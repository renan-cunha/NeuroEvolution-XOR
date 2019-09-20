# NeuroEvolution-XOR

Using genetic algorithms to train neural networks for the XOR gate problem

## Installation

```python3
pip3 install -r requirements.txt
```

## Run

```bash
python3 main.py
```

## Neural Network

* 2 neurons
* 16 neurons
* ReLU Activation
* 1 neuron
* Sigmoid Activation

## AG

* Selection by Tournament(size=3)
* Gaussian Mutation(mean=0, stddev=1, gene_prob=0.3)
* TwoPoint Crossover(prob=0.01)
* iterations=5

