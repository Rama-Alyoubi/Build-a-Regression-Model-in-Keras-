# Build a Regression Model in Keras

This project aims to build a regression model using Keras, a high-level neural networks API, to predict concrete strength based on various features. The project involves preprocessing the data, splitting it into training and test sets, creating a neural network model, training the model, and evaluating its performance. The code provided serves as an example implementation of this process.

## Prerequisites

- Python 
- NumPy
- Pandas
- scikit-learn
- TensorFlow (including the Keras module)

## Getting Started

To get started with the project, follow these steps:

1. Ensure that the required dependencies are installed.
2. Download the dataset 'concrete_data.csv' and place it in the appropriate directory.
3. Open the 'build-a-regression-model-in-keras.ipynb' file in your preferred Python editor or IDE.

## Code Explanation

The provided code performs the following steps:

1. Imports the necessary libraries and modules.
2. Loads the 'concrete_data.csv' dataset using Pandas.
3. Preprocesses the data by normalizing it.
4. Splits the data into training and test sets using scikit-learn's `train_test_split` function.
5. Initializes a list to store mean squared errors (MSE).
6. Executes the process 100 times, each with a different model.
7. Creates a sequential neural network model with three hidden layers and one output layer.
8. Compiles the model using the Adam optimizer and mean squared error loss function.
9. Trains the model on the training data for 100 epochs.
10. Evaluates the model on the test data and calculates the MSE.
11. Appends the MSE to the list.
12. Calculates the mean and standard deviation of the MSEs.
13. Prints the mean MSE and standard deviation.

## Acknowledgment

This project was completed as part of the assessment for the IBM Introduction to Deep Learning & Neural Networks with Keras course. The code and implementation were done by me.

## References

- [Keras Documentation](https://keras.io/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
