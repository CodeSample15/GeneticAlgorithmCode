using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Core : MonoBehaviour
{
    //Enum variables for the network controller to use
    enum ActivationFunctions { Sigmoid, Tanh, ReLU }

    //Creating variables the user can access in the editor
    [Tooltip("Put the GameObject that will become your network here")]
    [SerializeField] private GameObject Neural_Network;

    [Header("Specifying network sizes and activation functions")]
    [Tooltip("First and last values represent input and output values. All values in between are hidden layer sizes.")]
    [SerializeField] private List<int> Network_Sizes;

    [Tooltip("For telling the network which activation functions to use on each layer (The first one will be ignored since it is the input layer)")]
    [SerializeField] private List<ActivationFunctions> NetworkActivationFunctions;

    [Header("Training varialbes")]
    [Tooltip("How long (in seconds) each generation should last")]
    [SerializeField] private float TimePerGeneration;

    [Tooltip("How many networks should be in each generation")]
    [SerializeField] private int Number_Of_Networks;

    [Tooltip("How many generations the network should train for")]
    [SerializeField] private int Number_Of_Gens;

    [Tooltip("How fast the network will learn / mutate over time")]
    [SerializeField] private float Learning_Rate;

    [Space]
    [Tooltip("Will save the best network to your pc to be reused.")]
    [SerializeField] private bool SaveNetwork = false;

    [Tooltip("Only fill in if you clicked Save Network")]
    [SerializeField] private string NetworkSaveName;

    [Space]
    [Header("Initialization variables")]
    [Space]

    [Tooltip("Where the networks will be initialized to begin training.")]
    [SerializeField] private Vector3 StartPosition;

    [Space]

    [SerializeField] private float Min_Weight_Init;
    [SerializeField] private float Max_Weight_Init;

    [Space]

    [SerializeField] private float Min_Bias_Init;
    [SerializeField] private float Max_Bias_Init;

    private int networkSize;
    private int inputSize;
    private int outputSize;

    //for keeping track of the networks
    private List<GameObject> networks;

    //for error trapping any type of wrong input the user could give the program
    private bool successfulInit;

    //information about training
    private int currentGen;
    private float timeElapsedOnGen;

    //giving the other programs limited access to some of the private data
    #region Read Only Variables
    public int NetworkSize
    {
        get { return networkSize; }
    }

    public int InputSize
    {
        get { return inputSize; }
    }

    public int OutputSize
    {
        get { return outputSize; }
    }

    public List<int> NetworkSizes
    {
        get { return Network_Sizes; }
    }

    public int NumberOfNetworks
    {
        get { return Number_Of_Networks; }
    }

    public float lr
    {
        get { return Learning_Rate; }
    }

    public Vector3 startPosition
    {
        get { return StartPosition; }
    }
    #endregion

    void Awake()
    {
        successfulInit = false;

        currentGen = 0;
        timeElapsedOnGen = 0f;

        //error trapping
        if (Network_Sizes.Count < 2)
            Debug.LogError("Network size is too small!");
        else if (Neural_Network == null)
            Debug.LogError("No network object specified!");
        else if (Number_Of_Networks < 0)
            Debug.LogError("Number of networks is a negative number!");
        else if (Number_Of_Gens < 1)
            Debug.LogError("Number of gens was below 1!");
        else if (Learning_Rate < 0)
            Debug.LogError("Learning rate is negative!");
        else if (Neural_Network.GetComponent<Network>() == null)
            Debug.LogError("Network object does not have a Network script attatched!");
        else if (NetworkActivationFunctions.Count != Network_Sizes.Count)
            Debug.LogError("Network activation functions list is not the same size as the neural network!");
        else if (TimePerGeneration < 0)
            Debug.LogError("Negative time given for each generation!");
        else if (SaveNetwork && NetworkSaveName.Length == 0)
            Debug.LogError("Save button is set to true but there isn't a save name entered!");
        else
        {
            Debug.Log("Initializing...");
            networkSize = Network_Sizes.Count;
            inputSize = Network_Sizes[0];
            outputSize = Network_Sizes[Network_Sizes.Count - 1];

            initNetworks();

            successfulInit = true;
            Debug.Log("Finished!");
        }
    }

    void Update()
    {
        if (currentGen < Number_Of_Gens)
        {
            if (timeElapsedOnGen < TimePerGeneration)
            {


                //update the timer
                timeElapsedOnGen += Time.deltaTime;
            }
            else
            {
                //reset the networks and apply genetic algorithm

                currentGen++;
            }
        }
        else
        {
            //end training and possibly save the best network
        }
    }

    #region Network Management Code

    /// <summary>
    /// Initializing as many networks as the user specifies in the editor
    /// 
    /// Calculates a random set of 
    /// </summary>
    private void initNetworks()
    {
        networks = new List<GameObject>();

        for(int network = 0; network < NumberOfNetworks; network++)
        {
            networks.Insert(0, Instantiate(Neural_Network, StartPosition, Quaternion.identity));

            //initializing the network with random weights and biases
            networks[0].GetComponent<Network>().Init(randomMatrix(Min_Weight_Init, Max_Weight_Init), randomList(Min_Bias_Init, Max_Bias_Init));
        }
    }


    /// <summary>
    /// Iterates through each network, calculates the weights and biases, and calls the networks' "doAction" method
    /// 
    /// This updates each of the networks and is what runs the simulation
    /// </summary>
    private void tickAllNetworks()
    {
        for(int network=0; network<networks.Count; network++)
        {
            //getting the input from the network
            List<float> input = networks[network].GetComponent<Network>().getInput();


            //calculating output
            float[] networkOutput = Calculate(networks[network].GetComponent<Network>(), input);


            //setting the output for the network
            networks[network].GetComponent<Network>().setOutput(networkOutput);


            //running the action code of the network
            networks[network].GetComponent<Network>().doAction();
        }
    }
    #endregion

    #region Calcluations
    private float randomFloat(float min, float max)
    {
        return (float)Random.Range(min, max);
    }

    private float[] randomArray(int size, float min, float max)
    {
        float[] output = new float[size];

        for(int i=0; i<size; i++)
            output[i] = randomFloat(min, max);

        return output;
    }

    private List<float[]> randomList(float min, float max)
    {
        List<float[]> output = new List<float[]>();

        for(int layer=0; layer<networkSize; layer++)
        {
            output.Add(randomArray(Network_Sizes[layer], min, max));
        }

        return output;
    }

    private List<List<float[]>> randomMatrix(float min, float max)
    {
        List<List<float[]>> output = new List<List<float[]>>();

        output.Add(new List<float[]>()); //for the input layer, which won't have a prev neuron to assign a synapse to

        for(int layer=1; layer<networkSize; layer++)
        {
            output.Add(new List<float[]>());

            for (int neuron = 0; neuron < Network_Sizes[layer]; neuron++)
            {
                output[layer].Add(randomArray(Network_Sizes[layer - 1], min, max));
            }
        }

        return output;
    }

    private float[] Calculate(Network network, List<float> input)
    {
        //checking to make sure that input size is correct
        if (input.Count == inputSize)
        {
            //transferring the data in the list to an array
            float[] arrayInput = new float[inputSize];
            for (int i = 0; i < input.Count; i++)
                arrayInput[i] = input[i];

            //defining an array to hold the outputs and assigning the first value in the list to the input array
            List<float[]> outputs = new List<float[]>();
            outputs.Add(arrayInput); //feeding input

            for (int layer = 1; layer < networkSize; layer++)
                outputs.Add(new float[Network_Sizes[layer]]);

            List<float[]> biases = network.Biases;
            List<List<float[]>> weights = network.Weights;

            for (int layer = 1; layer < networkSize; layer++)
            {
                for (int neuron = 0; neuron < Network_Sizes[layer]; neuron++)
                {
                    float sum = biases[layer][neuron];

                    for (int prevneuron = 0; prevneuron < Network_Sizes[layer - 1]; prevneuron++)
                    {
                        sum += outputs[layer - 1][prevneuron] * weights[layer][neuron][prevneuron];
                    }

                    outputs[layer][neuron] = applyActivationFunction(sum, NetworkActivationFunctions[layer]);
                }
            }

            return outputs[outputs.Count - 1];
        }
        else
        {
            //the input the user entered is the wrong size
            Debug.LogError("Wrong size input given to the network!");
            return null;
        }
    }

    private float applyActivationFunction(float x, ActivationFunctions activationFunction)
    {
        switch(activationFunction)
        {
            case ActivationFunctions.Sigmoid:
                x = 1.0f / (1.0f + Mathf.Exp(-x));
                break;

            case ActivationFunctions.Tanh:
                x = 2 / (1 + Mathf.Exp(-(2 * x))) - 1;
                break;

            case ActivationFunctions.ReLU:
                x = Mathf.Max(0, x);
                break;
        }

        return x;
    }
    #endregion
}