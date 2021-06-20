using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Network : MonoBehaviour
{
    //Network object, will be what the Core script trains
    private Controller controller;

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

    //add any additional variables that you need for you for your network here
    public float carLength;
    public float speed;
    public float maxTurnSpeed;

    public float totalTurn;
    public float timeAlive;
    //END OF EDIT REGION

    public void Init(List<List<float[]>> weights, List<float[]> biases)
    {
        controller = (Controller)FindObjectOfType(typeof(Controller));

        this.weights = weights;
        this.biases = biases;
        output = new float[controller.OutputSize];

        Alive = true; //whether or not the network is alive and 

        Network_Fitness = 0;

        //START OF EDIT REGION

        //put any code here that you want for initialization here
        carLength = 1.53f;
        speed = 2f;
        maxTurnSpeed = 1.2f;

        totalTurn = 0f;
        timeAlive = 0f;

        Renderer renderer = GetComponent<Renderer>();
        float r = Random.Range(0.0f, 1.0f);
        float g = Random.Range(0.0f, 1.0f);
        float b = Random.Range(0.0f, 1.0f);

        renderer.material.color = new Color(r, g, b);
        //END OF EDIT REGION
    }

    //init without setting the weights and biases. you do not need to edit this
    public void Init()
    {
        Init(Weights, Biases);
    }

    public List<float> getInput()
    {
        List<float> output = new List<float>();

        //START OF EDIT REGION

        //add the values to the output list that will be provided to the network as input

        //Input size: 5

        RaycastHit hit;
        Vector3 rayStart = new Vector3(transform.position.x + carLength, transform.position.y, transform.position.z);

        //1
        Physics.Raycast(rayStart, transform.forward, out hit);
        output.Add(1 - hit.distance);

        //2
        Physics.Raycast(rayStart, transform.right, out hit);
        output.Add(1 - hit.distance);

        //3
        Physics.Raycast(rayStart, Quaternion.Euler(0, 45, 0) * transform.forward, out hit);
        output.Add(1 - hit.distance);

        //4
        Physics.Raycast(rayStart, Quaternion.Euler(0, -45, 0) * transform.forward, out hit);
        output.Add(1 - hit.distance);

        //5
        Physics.Raycast(rayStart, -transform.right, out hit);
        output.Add(1 - hit.distance);

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

        //move forward
        transform.Translate(Vector3.forward * speed * output[1] * Time.deltaTime);
        
        //turn based off of the output
        float turnAmount = output[0] * maxTurnSpeed;
        transform.Rotate(0, turnAmount * maxTurnSpeed, 0);
        totalTurn += turnAmount * maxTurnSpeed;

        //END OF EDIT REGION
    }

    public float getFitness()
    {
        float fitness = 0;

        //START OF EDIT REGION

        //set the fitness of the network to how well it's performing
        //for example: set the fitness to the amount of distance that the network traveled from a certain distance
        //you can chose whether a lower or higher fitness is better in the Core script in the editor

        //recording fitness based off of how long it's been alive
        timeAlive += Time.deltaTime;
        //fitness = distance(transform.position.x, transform.position.z, core.startPosition.x, core.startPosition.z);
        fitness += timeAlive;

        if (Mathf.Abs(totalTurn) >= 720)
        {
            fitness = 0; //if the network turns too much, fitness is set to zero and kill the network
            Alive = false;
        }
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

    private void OnCollisionEnter(Collision collision)
    {
        //if colliding with something (a wall), the network will stop simulating
        Alive = false;
    }

    private float distance(float x1, float y1, float x2, float y2)
    {
        float differenceX = x2 - x1;
        float differenceY = y2 - y1;

        float squaredX = differenceX * differenceX;
        float squaredY = differenceY * differenceY;

        return Mathf.Sqrt(squaredX + squaredY);
    }

    //END OF EDIT REGION
}