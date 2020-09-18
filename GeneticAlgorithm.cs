using System;
using System.Collections.Generic;
using UnityEngine;

public class GeneticAlgorithm : MonoBehaviour
{
    //public
    public int InputSize;
    public int OutputSize;

    //private
    private int[] neuralNetworkSizes;
    private int networkSize;
    private double trainingRate;

    private List<List<double[]>> weights;
    private List<double[]> biases;
    private List<double[]> outputs;

    private System.Random rand;

    public List<List<double[]>> getWeights
    {
        get { return weights; }
    }

    public List<double[]> getBiases
    {
        get { return biases; }
    }

    // Start is called before the first frame update
    void Awake()
    {
        //CREATE THE SHAPE OF THE NETWORK HERE
        neuralNetworkSizes = new int[4] { 2, 4, 4, 1 };

        trainingRate = 0.7;

        //-----------everything under here shouldn't be changed-----------
        rand = new System.Random();

        networkSize = neuralNetworkSizes.Length;
        InputSize = neuralNetworkSizes[0];
        OutputSize = neuralNetworkSizes[networkSize - 1];

        weights = randomMatrix(0.3f, 0.7f); //layer, neuron, prevNeuron
        biases = randomList(0.3f, 0.7f);

        outputs = new List<double[]>();

        for(int layer=0; layer<networkSize; layer++)
        {
            outputs.Add(new double[neuralNetworkSizes[layer]]);
        }
    }

    //public methods
    public void applyChanges()
    {
        //put movement scripts here
    }

    public void train()
    {
        calculate(getInput());
    }

    public void mutate(List<List<double[]>> weights, List<double[]> biases)
    {
        this.weights = weights;
        this.biases = biases;

        //code to mutate the neural network for the next generation
        for (int layer = 1; layer < networkSize; layer++)
        {
            for (int neuron = 0; neuron < neuralNetworkSizes[layer]; neuron++)
            {
                //mutating the biases
                biases[layer][neuron] += randomDouble(-trainingRate, trainingRate);

                for (int prevneuron = 0; prevneuron < neuralNetworkSizes[layer - 1]; prevneuron++)
                {
                    //mutating the weights
                    weights[layer][neuron][prevneuron] += randomDouble(-trainingRate, trainingRate);
                }
            }
        }
    }

    //helper methods
    private double[] getInput()
    {
        //Here is where you get input to feed into the network
        double[] input = new double[InputSize];

        for(int i=0; i<InputSize; i++)
        {
            input[i] = 0; //set the input here
        }

        return input;
    }

    private void calculate(double[] input)
    {
        if(input.Length != InputSize)
        {
            Debug.Log("Input size is incorrect!");
            return;
        }

        outputs[0] = input;

        for(int layer=1; layer<networkSize; layer++)
        {
            for(int neuron=0; neuron<neuralNetworkSizes[layer]; neuron++)
            {
                double sum = biases[layer][neuron];
                for(int prevneuron=0; prevneuron<neuralNetworkSizes[layer-1]; prevneuron++)
                {
                    sum += outputs[layer-1][prevneuron] * weights[layer][neuron][prevneuron];
                }

                outputs[layer][neuron] = sigmoid(sum);
            }
        }
    }

    private List<double[]> randomList(float min, float max)
    {
        List<double[]> doubleList = new List<double[]>();

        for(int layer=0; layer<networkSize; layer++)
        {
            doubleList.Add(randomArray(min, max, neuralNetworkSizes[layer]));
        }

        return doubleList;
    }

    private List<List<double[]>> randomMatrix(float min, float max)
    {
        List<List<double[]>> matrix = new List<List<double[]>>();

        matrix.Add(new List<double[]>());
        for (int layer = 1; layer < networkSize; layer++)
        {
            matrix.Add(new List<double[]>());
            for (int neuron = 0; neuron < neuralNetworkSizes[layer]; neuron++)
            {
                matrix[layer].Add(randomArray(min, max, neuralNetworkSizes[layer - 1]));
            }
        }

        return matrix;
    }

    private double[] randomArray(float min, float max, int length)
    {
        double[] output = new double[length];

        for(int i=0; i<length; i++)
        {
            output[i] = randomDouble(min, max);
        }

        return output;
    }

    private double randomDouble(double min, double max)
    {
        return (rand.NextDouble() * (max - min)) + min;
    }

    //activation functions
    private double sigmoid(double x)
    {
        return (double)1 / (1 + Math.Exp(-x));
    }
}