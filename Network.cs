using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Network : MonoBehaviour
{
    //Network object, will be what the Core script trains
    private Core core;

    private List<List<float[]>> weights;
    private List<float[]> biases;
    private float[] output;

    public bool Alive;
    public float Network_Fitness;

    //read only access
    public List<List<float[]>> Weights
    {
        get { return weights; }
    }
    public List<float[]> Biases
    {
        get { return biases; }
    }

    //START OF EDIT REGION
    
    //add any additional variables that you need for your network here
    
    //END OF EDIT REGION

    public void Init(List<List<float[]>> weights, List<float[]> biases)
    {
        core = (Core)FindObjectOfType(typeof(Core));

        this.weights = weights;
        this.biases = biases;
        output = new float[core.OutputSize];

        Alive = true; //whether or not the network is alive and 

        Network_Fitness = 0;

        //START OF EDIT REGION

        //put any code here that you want for initialization here

        //END OF EDIT REGION
    }

    //init without setting the weights and biases. you do not need to edit this
    public void Init()
    {
        Init(this.weights, this.biases);
    }

    public List<float> getInput()
    {
        List<float> output = new List<float>();

        //START OF EDIT REGION

        //add the values to the output list that will be provided to the network as input

        //END OF EDIT REGION

        return output;
    }

    public void doAction()
    {
        //START OF EDIT REGION

        /*
         * Take the values of the output layer and tell your network to do something with them
         * 
         * For example:
         * 
         *  setspeed(output[0]);
         *  setrotation(output[1]);
        */

        //END OF EDIT REGION
    }

    public float getFitness()
    {
        float fitness = 0;

        //START OF EDIT REGION

        //set the fitness of the network to how well it's performing
        //for example: set the fitness to the amount of distance that the network traveled from a certain distance
        //you can chose whether a lower or higher fitness is better in the Core script in the editor

        //END OF EDIT REGION

        return fitness;
    }

    public void checkForFail()
    {
        //START OF EDIT REGION

        //set the Alive variable to true if the network is still alive, and set it to false if the network "died" and should no longer run. This makes the training process more efficient.
        //You don't have to put the code here, you can put it somewhere else, as long as the program sets the Alive variable at the appropriate time

        //END OF EDIT REGION
    }

    #region Other network code

    public void setOutput(float[] output)
    {
        this.output = output;
    }

    #endregion

    //START OF EDIT REGION

    //put any additional code you need here for the network to run the way you want it. For example, you could put an on collision enter code block here to sense when a network fails in some way (hits a wall, in the case of the example scene)

    //END OF EDIT REGION
}
