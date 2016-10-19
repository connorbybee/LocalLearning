#  LocalLearning
## Towards local learning and recurrent computation

## Motivation
The purpose of these experiments is to examine local learning rules in recurrent networks. Local learning rules in recurrent networks eliminate the need for a forward propagation (inference) and back propagation (learning) phase during training. Instead, each weight is updated according to a rule based on the activity of connected neurons or neurons within a given layer. Recurrent computation with local learning can also replace the need to perform loop unrolling on time-varying input. Though, these experiments focus on the MNIST data set with stationary input. 

## Networks
* PCANet is a single layer network with lateral connections that performs principle component analysis. 
* ZCANet is a linear network that performs whitening
* ICANet is a non-linear network that performs an independent component analysis
* PhaseNet is a non-linear network that performs supervised learning.

## References Papers
Below are papers that inspired these tests.
[Optimization theory of Hebbian/anti-Hebbian networks for PCA and whitening](https://arxiv.org/abs/1511.09468)
[Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343)
[Early Inference in Energy-Based Models Approximates Back-Propagation](https://arxiv.org/abs/1510.02777)