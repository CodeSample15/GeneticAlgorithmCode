using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class NetworkData
{
    public List<List<float[]>> weights;
    public List<float[]> biases;

    public NetworkData(List<List<float[]>> weights, List<float[]> biases)
    {
        this.weights = weights;
        this.biases = biases;
    }
}