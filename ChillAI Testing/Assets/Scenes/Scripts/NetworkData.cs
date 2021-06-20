using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class NetworkData
{
    public List<List<float[]>> weights;
    public List<float[]> biases;
    public int inputSize;
    public int outputSize;
    
    public NetworkData(List<List<float[]>> weights, List<float[]> biases, int inputSize, int outputSize)
    {
        this.weights = weights;
        this.biases = biases;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }
}