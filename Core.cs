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

    [Tooltip("How likely a network will be mutated.")]
    [SerializeField] private float MutationRate;

    [Tooltip("How fast the network will learn / mutate over time")]
    [SerializeField] private float Learning_Rate;

    [Tooltip("Check this box if you want the network with a higher fitness to be treated as better")]
    [SerializeField] private bool Higher_Fitness_Is_Better = true;

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

    [Tooltip("the initial rotation of the networks")]
    [SerializeField] private Vector3 StartRotation;

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
    public int CurrentGen
    {
        get { return currentGen; }
    }

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

    public float mutation_rate
    {
        get { return MutationRate; }
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
        //initializing variables that doesn't require user input first
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
        else if (MutationRate < 0)
            Debug.LogError("Mutation rate is negative!");
        else
        {
            Debug.Log("Initializing...");

            networkSize = Network_Sizes.Count;
            inputSize = Network_Sizes[0];
            outputSize = Network_Sizes[Network_Sizes.Count - 1];

            initNetworks();

            successfulInit = true;
            Debug.Log("Finished!");

            Debug.Log("Starting training for generation 1");
        }
    }

    void Update()
    {
        if (successfulInit) //only running training session if everything was initialized without any errors
        {
            if (currentGen < Number_Of_Gens)
            {
                if (timeElapsedOnGen < TimePerGeneration)
                {
                    //updating the networks
                    tickAllNetworks();

                    //update the timer
                    timeElapsedOnGen += Time.deltaTime;
                }
                else
                {
                    //reset the networks and apply genetic algorithm------------------------------------------------------------------

                    //first get the best networks to the top of the list
                    getTopNetworks();

                    //reuse the top half of the networks and change the bottom half to mutated versions of the top half networks
                    SetLowerNetworks();

                    //reset all of the networks for the next generation
                    resetNetworks();

                    timeElapsedOnGen = 0f; //resetting the time
                    currentGen++; //starting a new generation
                }
            }
            else
            {
                //end training and save the best network
            }
        }
    }

    #region Network Management Code

    /// <summary>
    /// Initializing as many networks as the user specifies in the editor
    /// 
    /// Calculates a random set of weights and biases and assigns it to a newly instantiated network object
    /// </summary>
    private void initNetworks()
    {
        networks = new List<GameObject>();

        for(int network = 0; network < NumberOfNetworks; network++)
        {
            networks.Insert(0, Instantiate(Neural_Network, StartPosition, Quaternion.identity));

            //initializing the network with random weights and biases
            networks[0].GetComponent<Network>().Init(randomMatrix(Min_Weight_Init, Max_Weight_Init), randomList(Min_Bias_Init, Max_Bias_Init));
            networks[0].transform.rotation = Quaternion.Euler(StartRotation);
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

            //calculating the fitness of the network
            networks[network].GetComponent<Network>().Network_Fitness = networks[network].GetComponent<Network>().getFitness();
        }
    }

    /// <summary>
    /// Uses a Bubble Sorting algorithm to move the best networks to the top. Not very efficient, but it works just fine.
    /// 
    /// Depending on what the user selected for better fitness, the algorithm will either move the lower fitness or the higher fitness to the top of the list.
    /// 
    /// This method is called at the end of the generation to get help it select the best networks to clone and reproduce for the next generation.
    /// </summary>
    private void getTopNetworks()
    {
        bool change = false;

        //bubble sort algorithm to get the best networks at the top of the list (0 index)
        do
        {
            change = false;

            for(int i=0; i < networks.Count-1; i++)
            {
                if(networks[i].GetComponent<Network>().Network_Fitness < networks[i + 1].GetComponent<Network>().Network_Fitness && Higher_Fitness_Is_Better)
                {
                    //switch the networks so that the higher fitness network is on top
                    GameObject temp = networks[i];
                    networks[i] = networks[i + 1];
                    networks[i + 1] = temp;

                    change = true; //telling the loop to keep going
                }
                else if(networks[i].GetComponent<Network>().Network_Fitness > networks[i+1].GetComponent<Network>().Network_Fitness && !Higher_Fitness_Is_Better)
                {
                    //switch the networks so that the lower fitness network is on top
                    GameObject temp = networks[i];
                    networks[i] = networks[i + 1];
                    networks[i + 1] = temp;

                    change = true; //telling the loop to keep going
                }
            }
        } while (change);

        //now the networks in the "networks" list are ordered first to last, best to worst. This sets up the top networks to be reused in the next generation, and the lower networks will have their weights and biases set the the best networks and mutated
    }

    /// <summary>
    /// Resets all of the networks in the networks list by setting their position and rotation to what the user specified.
    /// </summary>
    private void resetNetworks()
    {
        foreach(GameObject network in networks)
        {
            network.GetComponent<Network>().Network_Fitness = 0; //resetting the network fitness
            network.transform.position = startPosition; //resetting position
            network.transform.rotation = Quaternion.Euler(StartRotation); //resetting rotation

            network.GetComponent<Network>().Init();
        }
    }

    /// <summary>
    /// Iterate through all of the lower networks in the networks list and assign them mutated brains of the better networks
    /// </summary>
    private void SetLowerNetworks()
    {
        int cutoff = (int)Mathf.Ceil(Number_Of_Networks / 2); //where the brains will start to be duplicated

        for(int i=cutoff; i<Number_Of_Networks; i++)
        {
            //getting one of the good networks and cloning it to the worse network
            Network newNetwork = networks[i - cutoff].GetComponent<Network>();
            newNetwork = Mutate(newNetwork); //mutating the old network for genetic variation 

            //replacing the old network values with the newly mutated one
            networks[i].GetComponent<Network>().Init(newNetwork.Weights, newNetwork.Biases);
        }
    }

    /// <summary>
    /// Returns a mutated version of the network modified as per the parameters the user specifies in the editor (mutation rate and learning rate)
    /// </summary>
    private Network Mutate(Network network)
    {
        //declaring variables for use in the loop and initializing them to prevent errors
        List<float[]> netBiases = network.Biases;
        List<List<float[]>> netWeights = network.Weights;

        bool BiasMutate = false;
        bool WeightMutate = false;

        float MutateAmount = 0f;

        //mutating
        for(int layer=1; layer<networkSize; layer++)
        {
            for(int neuron=0; neuron<Network_Sizes[layer]; neuron++)
            {
                //calculating a random chance of mutation. If the random number generated is less than or equal to the mutation rate, then the value the loop is currently on will be mutated
                BiasMutate = Random.Range(0.0f, 1.0f) <= MutationRate;


                //mutating the bias
                if (BiasMutate)
                {
                    MutateAmount = Random.Range(-Learning_Rate, Learning_Rate);
                    netBiases[layer][neuron] += MutateAmount;
                }


                for (int prevneuron=0; prevneuron<Network_Sizes[layer-1]; prevneuron++)
                {
                    WeightMutate = Random.Range(0.0f, 1.0f) <= MutationRate;

                    //mutate the weight
                    if (WeightMutate)
                    {
                        MutateAmount = Random.Range(-Learning_Rate, Learning_Rate);
                        netWeights[layer][neuron][prevneuron] += MutateAmount;
                    }
                }
            }
        }

        //reinitializing the mutated network and returning
        network.Init(netWeights, netBiases);
        return network;
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