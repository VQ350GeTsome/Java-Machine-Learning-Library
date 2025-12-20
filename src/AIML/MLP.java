package AIML;

public class MLP {

    private Layer[] layers;
    private float learningRate;
    private static java.util.Random rand = new java.util.Random();
    
    /**
     * Constructor that takes layer sizes.
     * For example: {@code new Perceptron(3, 4, 2)} creates:
     *      Layer 1: 3 inputs => 4 outputs
     *      Layer 2: 4 inputs => 2 outputs
     *
     * @param learningRate The rate at which this {@link AIML} will learn.
     * @param sizes The sizes of each layer (inputSize, hiddenSize..., outputSize).
     */
    public MLP(float learningRate, int... sizes) { 
        if (sizes.length < 2)  throw new IllegalArgumentException("Need at least input and output size");
        
        this.learningRate = learningRate;
        
        layers = new Layer[sizes.length - 1];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(sizes[i], sizes[i + 1]);
            layers[i].randomizeWeights();
        }
    }
    public MLP(float learningRate, float maxBiasInitialization, int... sizes) { 
        if (sizes.length < 2)  throw new IllegalArgumentException("Need at least input and output size");
        
        this.learningRate = learningRate;
        
        layers = new Layer[sizes.length - 1];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(sizes[i], sizes[i + 1], maxBiasInitialization);
            layers[i].randomizeWeights();
        }
    }
    
    private MLP(float learningRate, Layer[] layers) {
        this.learningRate = learningRate;
        this.layers = layers;
    }
    
    
    /**
     * Forward pass through all layers.
     * 
     * @param input Input column {@link Matrix.Matrix} ({@link Matrix.Matrix} of shape inputSize × 1).
     * @return Output column {@link Matrix.Matrix} ({@link Matrix.Matrix} of shape outputSize × 1).
     */
    public float[] forward(float[] input) {
        Matrix.Matrix result = input;
        // All the hidden layers
        for (int i = 0; layers.length - 1 > i; i++) result = layers[i].forward(result, hiddenActivation);
        
        // Final layer
        result = layers[layers.length - 1].forward(result, finalActivation);
        return result;
    }
            
    public void train(float[] input, float[] target) {
        Matrix.Matrix[] inputs = new Matrix.Matrix[layers.length];
        Matrix.Matrix result = input;

        // Hidden layers use hiddenActivation
        for (int i = 0; i < layers.length - 1; i++) {
            inputs[i] = result;
            result = layers[i].forward(result, hiddenActivation);
        }

        // Final layer uses finalActivation
        inputs[layers.length - 1] = result;
        result = layers[layers.length - 1].forward(result, finalActivation);
        
        // Compute error at output
        Matrix.Matrix error = target.subtract(result);

        // Start delta with output error
        Matrix.Matrix delta = error;

        // Backprop through layers
        for (int i = layers.length - 1; i >= 0; i--) {
            if (i == layers.length - 1) 
                // Final layer: use finalActivationDerivative
                delta = layers[i].backward(delta, learningRate, inputs[i], finalActivation);
            else 
                // Hidden layers: use hiddenActivationDerivative
                delta = layers[i].backward(delta, learningRate, inputs[i], hiddenActivation);
        }
    }
    public void reward(float[] input, float reward) {
        Matrix.Matrix currentInputs = input;

        for (int i = 0; layers.length - 1 > i; i++) {
            layers[i].reward(currentInputs, reward, learningRate);
            currentInputs = layers[i].forward(currentInputs, hiddenActivation);
        }
    }
    
    public AIML.MLP mutateUniform(float mutation) {
        // What the new layers of the new neural net will be
        Layer[] newLayers = new Layer[this.layers.length];
        
        // Mutate each value of the layers weights & biases 
        // by some value between [-mutation, mutation].
        for (int i = 0; this.layers.length > i; i++) {
            Matrix.Matrix newWeights = this.layers[i].getWeightMatrix();
            Matrix.Matrix newBiases = this.layers[i].getBiasMatrix();
            
            newWeights.mutate(val -> (float)(val + ((Math.random() * 2) - 1) * mutation));
            newBiases.mutate(val  -> (float)(val + ((Math.random() * 2) - 1) * mutation));
            
            newLayers[i] = new Layer(newWeights, newBiases);
        }
        
        return new AIML.MLP(this.learningRate, newLayers);
    }
    public AIML.MLP mutateGaussian(float mutation) {
        // What the new layers of the new neural net will be
        Layer[] newLayers = new Layer[this.layers.length];
        
        // Mutate each value of the layers weights & biases 
        // by some value between [-mutation, mutation].
        for (int i = 0; this.layers.length > i; i++) {
            Matrix.Matrix newWeights = this.layers[i].getWeightMatrix();
            Matrix.Matrix newBiases = this.layers[i].getBiasMatrix();
            
            newWeights.mutate(val -> (float)(val + rand.nextGaussian() * mutation));
            newBiases.mutate(val  -> (float)(val + rand.nextGaussian() * mutation));
            
            newLayers[i] = new Layer(newWeights, newBiases);
        }
        
        return new AIML.MLP(this.learningRate, newLayers);
    }
    
    public int getLayerCount() { return layers.length; }
    public int getNeuronsOfLayer(int i) { return layers[i].getNeuronCount(); }

    private class Layer {
        // Store the input & output sizes.
        private int inputSize;
        private int outputSize;

        // Weight matrix & bias vector.
        private float[][] weights;
        private float[] biases;

        // Store the input, output, & delta.
        private float[] input;
        private float[] output;
        private float[] delta;

        private Activation activation = MLP.LINEAR;

        public Layer(int inputSize, int outputSize, Activation activation) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.activation = activation;

            this.weights = new float[outputSize][inputSize];
            this.biases = new float[outputSize];

            initWeights();
        }

        private void initWeights() {
            java.util.Random rand = new java.util.Random();
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < inputSize; j++) {
                    weights[i][j] = (float) (rand.nextGaussian() * 0.01);
                }
                biases[i] = 0.0f;
            }
        }

        public float[] forward(float[] input) {
            this.input = input;
            this.output = new float[outputSize];

            // Loop through each neuron
            for (int i = 0; i < outputSize; i++) {
                float sum = biases[i];
                for (int j = 0; j < inputSize; j++) sum += weights[i][j] * input[j];
                output[i] = activation.activate.apply(sum);
            }
            return output;
        }

        public float[] backward(float[] gradOutput, float learningRate) {
            float[] gradInput = new float[inputSize];
            delta = new float[outputSize];

            // Get the delta ( derivative ).
            for (int i = 0; i < outputSize; i++) 
                delta[i] = gradOutput[i] * activation.derivative.apply(output[i]);
            

            // Get the gradient.
            for (int i = 0; i < inputSize; i++) for (int j = 0; j < outputSize; j++) 
                gradInput[i] += delta[j] * weights[j][i];
                
            // Update weights and biases.
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < inputSize; j++) weights[i][j] -= learningRate * delta[i] * input[j];
                biases[i] -= learningRate * delta[i];
            }
            return gradInput;
        }
        
        public int getNeuronCount() { return biases.length; }
        
    }
    
    @Override
    public String toString() {
        // Needs to be made. Print weights & biases.
        return "";
    }
     
    //<editor-fold defaultstate="collapsed" desc=" Activation Functions ">
    
    // Activation function data holder.
    public record Activation(java.util.function.Function<Float, Float> activate,
                          java.util.function.Function<Float, Float> derivative) {};

    // The actual activation functions
    public static final Activation 
            LINEAR = new Activation
            (
                (a) -> (a), 
                (a) -> 1.0f
            ),
            RELU = new Activation
            (
                (a) -> a > 0 ? a : 0.0f, 
                (a) -> a > 0 ? 1.0f : 0.0f
            ),
            LEAKYRELU = new Activation
            (
                (a) -> a >= 0 ? a : 0.01f * a, 
                (a) -> a >= 0 ? 1.0f : 0.01f
            ),
            SIGMOID = new Activation
            (
                (a) -> (float)(1.0 / (1.0 + Math.exp(-a))),
                (a) -> {
                    float s = (float)(1.0 / (1.0 + Math.exp(-a)));
                    return s * (1 - s);
                }
            ),
            TANH = new Activation
            (
                (a) -> (float)Math.tanh(a),
                (a) -> {
                    float t = (float)Math.tanh(a);
                    return 1 - t * t;
                }
            );
    //</editor-fold>
}
