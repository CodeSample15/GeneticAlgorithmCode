using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class NetworkData
{
    public Network network;

    public NetworkData(Network network)
    {
        this.network = network;
    }
}