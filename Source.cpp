#include <iostream>
#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
//Global variable set up
const int radius = 3;
const int numSamples = 100;
const double learningRate = 0.15;
const int epochs = 1;
const int numNeurons = 20;
//Set up neurons
double inputLayer[2][numNeurons] = { 0 }; //takes input and weights
double outputLayer[1][numNeurons] = { 0 }; //takes weights and outputs

//Calculates dot product of two arrays from a given pointer and returns a total - must be same size
double dotProduct(double *array1, double *array2, int size) {
	double total = 0;
	for (int i = 0; i < size; i++) {
		total += array1[i] * array2[i];
	}
	return total;
}
//Does an element by element multiplication but keeps the size of the array the same- alters an array at a given pointer to contain this
void elementMultiply(double *array1, double *array2, double *output, int rows) {
	for (int i = 0; i < rows; i++) {
		output[i] = array1[i] * array2[i];
	}
}
//Calculates the sigmoid derivative for every position in an array then alters an array at a pointer to contain this 
void sigmoidDerivativeMatrix(double* inputArray, int rows, double *output) {
	for (int i = 0; i < rows; i++) {
		output[i] = inputArray[i] * (1 - inputArray[i]);
	}
}
//Calculates the sigmoid derivative of a singular value and returns it as a double
double sigmoidDerivativeScalar(double inputVal) {
	double sigmoidValue;
	sigmoidValue = inputVal * (1 - inputVal);
	return sigmoidValue;
}
//Calculates the sigmoid value of a single input and returns it as a double
double sigmoidScalar(double inputVal) {
	double sigmoidValue;
	sigmoidValue = 1 / (1 + exp(-inputVal));
	return sigmoidValue;
}
//Calculates the sigmoid for every position in an array then alters an array at a pointer to contain this 
void sigmoidMatrix(double *inputArray, int rows, double *output) {
	for (int i = 0; i < rows; i++) {
		output[i] = 1 / (1 + exp(-inputArray[i]));
	}
}
//Performs Matrix multiplication returns value into pointer location
void matrixMultiply(double *array1, double *array2, double *output, int arr1_rows, int arr1_cols, int arr2_cols) {
	double result;
	for (int row = 0; row < arr1_rows; row++) {
		for (int col = 0; col < arr2_cols; col++) {
			result = 0;
			for (int i = 0; i < arr1_cols; i++) {
				result = result + array1[row * arr1_cols + i] * array2[i * arr2_cols + col];
			}
			output[row * arr2_cols + col] = result;
		}
	}
}
//Adds two matricies together and returns in pointer of the first matrix
void addMatrix(double *targetArray, double *addArray, int rows) {
	for (int i = 0; i < rows; i++) {
		targetArray[i] = targetArray[i] + addArray[i];
	}
}
//Creates input data and returns a pointer to the data set 
double* createInput(int dataSize, double* inputData) {//Takes in data to ensure inputs match outputs
	int setSize = dataSize * 2;
	double* dataSet = new double[setSize];
	for (int i = 0; i < dataSize; i++) {
		dataSet[i] = radius * cos(inputData[i]); //Uses trig to build data set 
		dataSet[(dataSize + i)] = radius * sin(inputData[i]);//As dataset has two co-ords was more efficient to return in one array of double size
	}
	return dataSet;
}
//Creates an output of random angles between 0 and 1 with 4 decimal places
double* createOutput(int dataSize) {
	double* dataSet = new double[dataSize];
	for (int i = 0; i < dataSize; i++) {
		double theta = rand() % 1000; //randomly seeded number generator
		theta = theta / 1000.0;
		dataSet[i] = theta;
	}
	return dataSet;
}
void trainMultiLayerPerceptron(double* inputData, double* expectedOutputData, int maxiter) {
	for (int j = 0; j < maxiter; j++) {
		double layerOneAdjustment[2][numNeurons] = { 0 };
		double layerTwoAdjustment[1][numNeurons] = { 0 };
		double errorSum = 0.0;
		for (int i = 0; i < numSamples; i++) {
			double layer1output[1][numNeurons] = { 0 };
			double inputDataArray[1][2] = { inputData[i] ,inputData[numSamples + i] };
			double transposeInputData[2][1] = { inputData[i],inputData[numSamples + i] };
			double layerOneAdjustmentTmp[2][numNeurons] = { 0 };
			double layerTwoAdjustmentTmp[1][numNeurons] = { 0 };
			double layer2delta[1][1] = { 0 };
			double layer1error[1][numNeurons] = { 0 };
			double layer1delta[1][numNeurons] = { 0 };
			double layer1outputSigmoid[1][numNeurons] = { 0 };
			matrixMultiply(*inputDataArray, *inputLayer, *layer1output, 1, 2, numNeurons);
			sigmoidMatrix(*layer1output, numNeurons, *layer1output);
			double layer2output = sigmoidScalar(dotProduct(*layer1output, *outputLayer, numNeurons));
			double layer2error = expectedOutputData[i] - layer2output;
			layer2delta[0][0] = layer2error * sigmoidDerivativeScalar(layer2output);
			matrixMultiply(*layer2delta, *outputLayer, *layer1error, 1, 1, numNeurons);
			sigmoidDerivativeMatrix(*layer1output, numNeurons, *layer1outputSigmoid);
			elementMultiply(*layer1error, *layer1outputSigmoid, *layer1delta, numNeurons);
			matrixMultiply(*transposeInputData, *layer1delta, *layerOneAdjustmentTmp, 2, 1, numNeurons);
			matrixMultiply(*layer2delta, *layer1output, *layerTwoAdjustmentTmp, 1, 1, numNeurons);
			for (int ii = 0; ii < numNeurons; ii++) {
				layerOneAdjustment[0][ii] = layerOneAdjustment[0][ii] + layerOneAdjustmentTmp[0][ii];
				layerOneAdjustment[1][ii] = layerOneAdjustment[1][ii] + layerOneAdjustmentTmp[1][ii];
				layerTwoAdjustment[0][ii] = layerOneAdjustment[0][ii] + layerTwoAdjustmentTmp[0][ii];
			}
			errorSum = errorSum + (layer2error * layer2error);
			if (j == (maxiter - 1)) {
				printf("input 1 %lf\n", inputData[i]);
				printf("input 2 %lf\n", inputData[numSamples + i]);
				printf("Expected output %lf\n", expectedOutputData[i]);
				printf("output 1 %lf\n", layer2output);
				printf("error sum %lf\n", errorSum);
			}

			for (int ii = 0; ii < numNeurons; ii++) {//Update layer weights by learning rate times the adjustment 
				inputLayer[0][ii] = inputLayer[0][ii] + learningRate * layerOneAdjustment[0][ii];
				inputLayer[1][ii] = inputLayer[1][ii] + learningRate * layerOneAdjustment[1][ii];
				outputLayer[0][ii] = outputLayer[0][ii] + learningRate * layerTwoAdjustment[0][ii];
			}

		}

	}
}
int main(int argc, char **argv)
{
	srand(time(NULL));
	double* expectedOutputData = createOutput(numSamples);
	double* inputData = createInput(numSamples, expectedOutputData);
	double* testOutput = createOutput(1);
	double* testInput = createInput(1, testOutput);
	for (int i = 0; i < epochs; i++) {
		for (int ii = 0; ii < numNeurons; ii++) {
			inputLayer[0][ii] = { ((rand() % 1000) / 1000.0) - 0.5 };
			inputLayer[1][ii] = { ((rand() % 1000) / 1000.0) - 0.5 };
			outputLayer[0][ii] = { ((rand() % 1000) / 1000.0) - 0.5 };
		}
		trainMultiLayerPerceptron(inputData, expectedOutputData, 1000);
	}
	return 0;
}

