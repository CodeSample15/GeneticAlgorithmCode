using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;
using UnityEditor;

public class Controller : MonoBehaviour
{
    //Enum variables for the network controller to use
    enum ActivationFunctions { Input, Sigmoid, Tanh, ReLU, Binary_Step, Output }
    enum MutationTypes { TopHalf, TopTwo, Top }

    //Creating variables the user can access in the editor
    [Tooltip("Put the GameObject that will become your network here")]
    [SerializeField] private GameObject Neural_Network;



    [Header("Specifying network sizes and activation functions")]

    [Tooltip("First and last values represent input and output values. All values in between are hidden layer sizes.")]
    [SerializeField] private List<int> Network_Sizes;

    [Tooltip("For telling the network which activation functions to use on each layer (The first one will be ignored since it is the input layer)")]
    [SerializeField] private List<ActivationFunctions> NetworkActivationFunctions;

    [Tooltip("Custom output: each activation function represents a different node in the output layer.")]
    [SerializeField] private List<ActivationFunctions> OutputActivationFunctions;



    [Header("Training varialbes")]

    [Tooltip("How the netorks will be mutated")]
    [SerializeField] private MutationTypes mutationType;

    [Tooltip("How long (in seconds) each generation should last")]
    [SerializeField] private float TimePerGeneration;

    [Tooltip("How many networks should be in each generation")]
    [SerializeField] private int Number_Of_Networks;

    [Tooltip("How many generations the network should train for")]
    [SerializeField] private int Number_Of_Gens;

    [Tooltip("How likely a network will be mutated.")]
    [SerializeField] private float MutationRate;

    [Tooltip("How much a network will be mutated by.")]
    [SerializeField] private float Learning_Rate;

    [Tooltip("Check this box if you want the network with a higher fitness to be treated as better")]
    [SerializeField] private bool Higher_Fitness_Is_Better = true;

    [Space]
    [Tooltip("Will save the best network at the end of training to your pc to be reused.")]
    [SerializeField] private bool SaveNetwork = false;

    [Tooltip("Load a previously trained network")]
    [SerializeField] private bool LoadNetwork = false;

    [Tooltip("Only fill in if you clicked Save Network. Works for both saving and loading a network")]
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

    private bool customOutput; //if the user decides to make a custom output configuration rather than just one activation function for the entire layer

    //for keeping track of the networks
    private List<GameObject> networks;

    //for error trapping any type of wrong input the user could give the program
    private bool successfulInit;

    //information about training
    private int currentGen;
    private float timeElapsedOnGen;
    private float bestFitness; //the best fitness of the last generation to train. Does not record the best fitness overall

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

    public float BestFitness
    {
        get { return bestFitness; }
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
        //initializing variables that don't require user input first
        successfulInit = false;

        currentGen = 0;
        timeElapsedOnGen = 0f;
        bestFitness = 0f;

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
        else if (!InputLayerCorrect())
            Debug.LogError("Input layer was not set up correctly (more than one or non in the right place). Input should be selected as the network's first activation function only.");
        else if (TimePerGeneration < 0)
            Debug.LogError("Negative time given for each generation!");
        else if (SaveNetwork && NetworkSaveName.Length == 0)
            Debug.LogError("Save button is set to true but there isn't a save name entered!");
        else if (MutationRate < 0)
            Debug.LogError("Mutation rate is negative!");
        else if (mutationType == MutationTypes.TopTwo && Number_Of_Networks < 3)
            Debug.LogError("TopTwo mutation type was chosen but there are less than 3 networks!");
        else if (!OutputLayerCorrect())
            Debug.LogError("Output layer is in an incorrect position!");
        else if (customOutput && OutputActivationFunctions.Count != Network_Sizes[Network_Sizes.Count - 1])
            Debug.LogError("Cusom output does not fit output layer size.");
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
                if (timeElapsedOnGen < TimePerGeneration && !AllNetworksDead())
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

                    bestFitness = networks[0].GetComponent<Network>().Network_Fitness; //recording the best fitness of the generation

                    //creating the next generation of networks using the method the user picks
                    switch (mutationType)
                    {
                        case MutationTypes.TopHalf:
                            SetLowerNetworks(); //reuse the top half of the networks and change the bottom half to mutated versions of the top half networks
                            break;

                        case MutationTypes.TopTwo:
                            CrossTopTwoNetworks(); //fill the next generation with mutated children of the top two networks
                            break;

                        case MutationTypes.Top:
                            SetLowerNetworks(networks[0].GetComponent<Network>()); //cloning + mutating the best network to the rest of the networks
                            break;
                    }

                    //reset all of the networks for the next generation
                    resetNetworks();

                    timeElapsedOnGen = 0f; //resetting the time
                    currentGen++; //starting a new generation

                    //logging a message saying what generation the program is currently training and how well the last generation performed
                    Debug.Log("Best fitness of generation " + (currentGen).ToString() + ": " + bestFitness.ToString());
                    Debug.Log("Training generation " + (currentGen + 1).ToString() + "...");
                }
            }
            else
            {
                //end training and save the best network
                if (SaveNetwork)
                {
                    Debug.Log("Saving the best network...");
                    getTopNetworks();

                    Save_Network_To_Device(networks[0].GetComponent<Network>().Weights, networks[0].GetComponent<Network>().Biases);
                    Debug.Log("Saved!");

                    SaveNetwork = false;
                }

                Destroy(gameObject); //end the training process
            }
        }
    }

    #region Network Management Code

    ///
    /// Does a simple check to make sure that the network has an input layer defined in the right spot
    ///
    private bool InputLayerCorrect()
    {
        if (NetworkActivationFunctions[0] == ActivationFunctions.Input)
        {
            //user has a activation function in the right place

            for (int i = 1; i < NetworkActivationFunctions.Count; i++)
            {
                if (NetworkActivationFunctions[i] == ActivationFunctions.Input)
                {
                    //user has more than one input layer. return false
                    return false;
                }
            }

            return true; //if everything is set up correctly, return true
        }
        else
        {
            return false;
        }
    }

    /// <summary>
    /// Does the same thing as the function above this one
    /// </summary>
    private bool OutputLayerCorrect()
    {
        for (int i=0; i<NetworkActivationFunctions.Count-1; i++)
        {
            if(NetworkActivationFunctions[i] == ActivationFunctions.Output)
            {
                return false; //returns false if there is an output activation function anywhere but the last layer
            }
        }

        //determine if the user wants to have a custom output
        if(NetworkActivationFunctions[NetworkActivationFunctions.Count-1] == ActivationFunctions.Output)
        {
            customOutput = true;
        }

        return true;
    }

    /// <summary>
    /// Initializing as many networks as the user specifies in the editor
    /// 
    /// Calculates a random set of weights and biases and assigns it to a newly instantiated network object
    /// </summary>
    private void initNetworks()
    {
        networks = new List<GameObject>();

        //if the user wants to load a network, make sure there's a network to load that has the right settings
        if (LoadNetwork)
        {
            if (Load_Network() == null)
            {
                Debug.LogWarning("No network found to load. Using random weights and biases instead");
                LoadNetwork = false;
            }
            else
            {
                Debug.Log("Network found!");

                //load the network and intialize a generation with them
                NetworkData data = Load_Network();

                //if any of the settings are incorrect, change the settings and continue with a warning
                if (data.biases.Count != networkSize || data.inputSize != inputSize || data.outputSize != outputSize)
                {
                    Debug.LogWarning("Loaded network has different input, output, or size than what is required! Changing settings and attempting to train...");

                    /*
                    //changing the settings of the Core script to match that of the loaded model
                    networkSize = data.biases.Count;
                    inputSize = data.biases[0].Length;
                    outputSize = data.biases[data.biases.Count - 1].Length;
                    */
                }
            }
        }

        //creating the networks
        for (int network = 0; network < NumberOfNetworks; network++)
        {
            networks.Insert(0, Instantiate(Neural_Network, StartPosition, Quaternion.identity));

            //initializing the network with random weights and biases
            if (LoadNetwork)
            {
                NetworkData data = Load_Network();

                //initializing the network with the loaded parameters
                networks[0].GetComponent<Network>().Init(data.weights, data.biases);
            }
            else
            {
                //create a randomly initialized network instead
                networks[0].GetComponent<Network>().Init(randomMatrix(Min_Weight_Init, Max_Weight_Init), randomList(Min_Bias_Init, Max_Bias_Init));
            }

            //setting the rotation of the initialized network
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
        for (int network = 0; network < networks.Count; network++)
        {
            //checking to see if the network failed
            networks[network].GetComponent<Network>().checkForFail();


            //only continuing if the network is still "alive"
            if (networks[network].GetComponent<Network>().Alive)
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

            for (int i = 0; i < networks.Count - 1; i++)
            {
                if (networks[i].GetComponent<Network>().Network_Fitness < networks[i + 1].GetComponent<Network>().Network_Fitness && Higher_Fitness_Is_Better)
                {
                    //switch the networks so that the higher fitness network is on top
                    GameObject temp = networks[i];
                    networks[i] = networks[i + 1];
                    networks[i + 1] = temp;

                    change = true; //telling the loop to keep going
                }
                else if (networks[i].GetComponent<Network>().Network_Fitness > networks[i + 1].GetComponent<Network>().Network_Fitness && !Higher_Fitness_Is_Better)
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
        foreach (GameObject network in networks)
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
        int cutoff = Mathf.CeilToInt((float)Number_Of_Networks / 2); //where the top nets will start to be duplicated

        for (int i = cutoff; i < Number_Of_Networks; i++)
        {
            //getting one of the good networks and cloning it to the worse network
            Network oldNetwork = networks[i - cutoff].GetComponent<Network>();

            //mutating the weights and biases of the old network
            List<List<float[]>> newWeights = MutateWeights(oldNetwork.Weights);
            List<float[]> newBiases = MutateBiases(oldNetwork.Biases);

            //reinitializing the network with the new weights and biases
            networks[i].GetComponent<Network>().Init(newWeights, newBiases);
        }
    }

    /// <summary>
    /// Overloaded version of the SetLowerNetworks method to set the lower network to a certain network
    /// </summary>
    private void SetLowerNetworks(Network network)
    {
        network.Init();

        //starting with the second network in the list and cloning + mutating the network fed as an argument
        for (int i = 1; i < Number_Of_Networks; i++)
        {
            //mutating the weights and biases of the old network
            List<List<float[]>> newWeights = MutateWeights(network.Weights);
            List<float[]> newBiases = MutateBiases(network.Biases);

            //reinitializing the network with the new weights and biases
            networks[i].GetComponent<Network>().Init(newWeights, newBiases);
        }
    }

    /// <summary>
    /// Instead of cloning the top half of the networks and mutating them, cross the top two networks over and use that network to create the rest of the offspring with mutations
    /// </summary>
    private void CrossTopTwoNetworks()
    {
        //getting the child of the top two networks
        List<float[]> childBiases = combineBiases(networks[0].GetComponent<Network>().Biases, networks[1].GetComponent<Network>().Biases);
        List<List<float[]>> childWeights = combineWeights(networks[0].GetComponent<Network>().Weights, networks[1].GetComponent<Network>().Weights);

        //variables that will be used inside of the loop
        List<float[]> newBiases;
        List<List<float[]>> newWeights;

        //changing the rest of the networks
        for (int i = 2; i < Number_Of_Networks; i++)
        {
            newBiases = MutateBiases(childBiases);
            newWeights = MutateWeights(childWeights);

            //giving the new weights and biases to the old network
            networks[i].GetComponent<Network>().Init(newWeights, newBiases);
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

        for (int i = 0; i < size; i++)
            output[i] = randomFloat(min, max);

        return output;
    }

    private List<float[]> randomList(float min, float max)
    {
        List<float[]> output = new List<float[]>();

        for (int layer = 0; layer < networkSize; layer++)
        {
            output.Add(randomArray(Network_Sizes[layer], min, max));
        }

        return output;
    }

    private List<List<float[]>> randomMatrix(float min, float max)
    {
        List<List<float[]>> output = new List<List<float[]>>();

        output.Add(new List<float[]>()); //for the input layer, which won't have a prev neuron to assign a synapse to

        for (int layer = 1; layer < networkSize; layer++)
        {
            output.Add(new List<float[]>());

            for (int neuron = 0; neuron < Network_Sizes[layer]; neuron++)
            {
                output[layer].Add(randomArray(Network_Sizes[layer - 1], min, max));
            }
        }

        return output;
    }

    /// <summary>
    /// Calculate the output for a given network. Returns an array containing the output values for the given input
    /// </summary>
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

                    if (customOutput && layer == NetworkSize)
                    {
                        //apply custom output
                        outputs[layer][neuron] = applyActivationFunction(sum, OutputActivationFunctions[neuron]);
                    }
                    else
                    {
                        //apply normal layer activation functions
                        outputs[layer][neuron] = applyActivationFunction(sum, NetworkActivationFunctions[layer]);
                    }
                }
            }

            return outputs[outputs.Count - 1];
        }
        else
        {
            //the input the user entered is the wrong size
            Debug.LogError("Wrong size input given to the network!");
            //Destroy(gameObject); //end the program
            return null;
        }
    }

    /// <summary>
    /// Returns a mutated version of the biases entered into this method.
    /// 
    /// Does this by adding a random amount (determined by learning rate) to one of the biases at a rate set by the MutationRate.
    /// </summary>
    private List<float[]> MutateBiases(List<float[]> bias)
    {
        //declaring variables for use in the loop and initializing them to prevent errors
        bool BiasMutate = false;
        float MutateAmount = 0f;

        //creating a new list to store the data
        List<float[]> output = new List<float[]>();
        output.Add(new float[0]); //adding the first layer, which is for input and doesn't have its own set of biases

        //mutating
        for (int layer = 1; layer < networkSize; layer++)
        {
            output.Add(new float[bias[layer].Length]);

            for (int neuron = 0; neuron < Network_Sizes[layer]; neuron++)
            {
                //calculating a random chance for mutation. If the random number generated is less than or equal to the mutation rate, then the value the loop is currently on will be mutated
                BiasMutate = Random.Range(0.00f, 1.00f) <= MutationRate;

                //mutating the bias
                if (BiasMutate)
                {
                    MutateAmount = Random.Range(-Learning_Rate, Learning_Rate);
                    output[layer][neuron] = bias[layer][neuron] + MutateAmount;
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Returns a mutated version of the weights entered into this method. Same process as the bias mutate code
    /// </summary>
    private List<List<float[]>> MutateWeights(List<List<float[]>> weight)
    {
        //declaring variables for use in the loop and initializing them to prevent errors
        bool WeightMutate = false;
        float MutateAmount = 0f;

        //creating a new list to store the data
        List<List<float[]>> output = new List<List<float[]>>();
        output.Add(new List<float[]>()); //adding the first layer, which is for input and doesn't have its own set of weights

        //mutating
        for (int layer = 1; layer < networkSize; layer++)
        {
            output.Add(new List<float[]>());

            for (int neuron = 0; neuron < Network_Sizes[layer]; neuron++)
            {
                output[layer].Add(new float[Network_Sizes[layer - 1]]);

                for (int prevneuron = 0; prevneuron < Network_Sizes[layer - 1]; prevneuron++)
                {
                    //calculating a random chance for mutation. If the random number generated is less than or equal to the mutation rate, then the value the loop is currently on will be mutated
                    WeightMutate = Random.Range(0.00f, 1.00f) <= MutationRate;

                    //mutating the weight
                    if (WeightMutate)
                    {
                        MutateAmount = Random.Range(-Learning_Rate, Learning_Rate);
                        output[layer][neuron][prevneuron] = weight[layer][neuron][prevneuron] + MutateAmount;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Takes the biases from two parent networks and adds them together to make a child set of network biases
    /// </summary>
    private List<float[]> combineBiases(List<float[]> bias1, List<float[]> bias2)
    {
        //creating a new set of biases
        List<float[]> combined = new List<float[]>();
        combined.Add(new float[0]); //adding the first layer, which is ignored

        for (int layer = 1; layer < networkSize; layer++)
        {
            combined.Add(new float[Network_Sizes[layer]]); //adding a set of neurons for the next layer

            for (int neuron = 0; neuron < Network_Sizes[layer]; neuron++)
            {
                //finding the average between the parents and setting them to the child set of biases
                combined[layer][neuron] = (bias1[layer][neuron] + bias2[layer][neuron]) / 2;
            }
        }

        return combined; //return the child
    }

    /// <summary>
    /// Takes the weights from two parent networks and adds them together to make a child set of network weights
    /// (Same process as the method above this one)
    /// </summary>
    private List<List<float[]>> combineWeights(List<List<float[]>> weight1, List<List<float[]>> weight2)
    {
        //creating a new set of weights
        List<List<float[]>> combined = new List<List<float[]>>();
        combined.Add(new List<float[]>()); //adding the first layer, which is ignored

        for (int layer = 1; layer < networkSize; layer++)
        {
            combined.Add(new List<float[]>()); //adding a new layer

            for (int neuron = 0; neuron < Network_Sizes[layer]; neuron++)
            {
                combined[layer].Add(new float[Network_Sizes[layer - 1]]);

                for (int prevneuron = 0; prevneuron < Network_Sizes[layer - 1]; prevneuron++)
                {
                    //finding the average between the parents and assigning the sum to the child weight
                    combined[layer][neuron][prevneuron] = (weight1[layer][neuron][prevneuron] + weight2[layer][neuron][prevneuron]) / 2;
                }
            }
        }

        return combined; //return the child
    }

    /// <summary>
    /// For testing if all of the networks are dead to finish the current generation
    /// </summary>
    private bool AllNetworksDead()
    {
        foreach (GameObject network in networks)
        {
            if (network.GetComponent<Network>().Alive)
            {
                //network has failed, stop it
                return false;
            }
        }

        return true;
    }

    private float applyActivationFunction(float x, ActivationFunctions activationFunction)
    {
        switch (activationFunction)
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

            case ActivationFunctions.Binary_Step:
                if (x < 0)
                    x = 0;
                else
                    x = 1;
                break;
        }

        return x;
    }
    #endregion

    #region Saving and loading the network
    private void Save_Network_To_Device(List<List<float[]>> weights, List<float[]> biases)
    {
        BinaryFormatter formatter = new BinaryFormatter();
        string path = Application.persistentDataPath + "/" + NetworkSaveName + ".chill";
        FileStream stream = new FileStream(path, FileMode.Create);

        NetworkData data = new NetworkData(weights, biases, InputSize, OutputSize);

        formatter.Serialize(stream, data);
        stream.Close();
    }

    private NetworkData Load_Network()
    {
        string path = Application.persistentDataPath + "/" + NetworkSaveName + ".chill";
        if (File.Exists(path))
        {
            BinaryFormatter formatter = new BinaryFormatter();
            FileStream stream = new FileStream(path, FileMode.Open);

            NetworkData data = formatter.Deserialize(stream) as NetworkData;
            stream.Close();

            return data;
        }
        else
        {
            Debug.Log("File not created yet!");
            return null;
        }
    }
    #endregion
}