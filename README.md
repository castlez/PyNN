# PyNN
Python neural network written to learn the mnist handwritten digits

### The Project
The MNIST database includes handwritten digits encoded into arrays represent pixel values. 
This neural network is designed to use these as inputs and produce an answer: 0-9. It does this by
"feeding forward" the inputs throught the layers of the neural net. First it uses  


Sig(w DOT x)  

where Sig is the sigmoid function, w is the weights of that layer, and x is the inputs. Then from the hidden to output layer the 
process is repeated, but with the ouputs from the previous step used instead of the origin inputs. Once an answer has been produced, the
backpropagation algorithm is used to update the weights according to the errors.  

### To Run
In the main directory of the project (where 'Main.py' is located)  
`$ python3 Main.py`

### Links 
Here is a small collection of wikipedia links to explain the subjects involved  
- [Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
- [MNIST](https://en.wikipedia.org/wiki/MNIST_database)
