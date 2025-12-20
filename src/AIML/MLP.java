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
        for (int i = 0; layers.length > i; i++) layers[i] = new Layer(sizes[i], sizes[i + 1], MLP.LINEAR);
        
    }
    public MLP(float learningRate, float maxBiasInitialization, int... sizes) { 
        if (sizes.length < 2)  throw new IllegalArgumentException("Need at least input and output size");
        
        this.learningRate = learningRate;
        
        layers = new Layer[sizes.length - 1];
        for (int i = 0; layers.length > i; i++) layers[i] = new Layer(sizes[i], sizes[i + 1], MLP.LINEAR);
        
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
        float[] result = input;
        
        // Pass the output of each layer as the input of the next until
        // all layers have been fowarded.
        for (int i = 0; layers.length > i; i++) result = layers[i].forward(result);
        
        return result;
    }
            
    public void train(float[] input, float[] target) {
        // Forward pass.
        float[] result = input;
        for (int i = 0; i < layers.length; i++) result = layers[i].forward(result);

        // Compute output error.
        float[] error = new float[target.length];
        for (int i = 0; i < target.length; i++) error[i] = target[i] - result[i];
        

        // Backprop.
        float[] delta = error;
        for (int i = layers.length - 1; i >= 0; i--) delta = layers[i].backward(delta, learningRate);
    }
    
    public AIML.MLP uniformMutate(float mutation) {
        return this.mutate( (a) -> {
                float mutate = (float) ((Math.random() * 2 - 1) * mutation);
                return a + mutate;
            }
        );
    }
    public AIML.MLP gaussianMutate(float mutation) {
        return this.mutate( (a) -> {
                float mutate = (float) (rand.nextGaussian() * mutation);
                return a + mutate;
            }
        );
    }
    public AIML.MLP mutate(java.util.function.Function<Float, Float> mutate) {
        // What the new layers of the new neural net will be
        Layer[] newLayers = new Layer[this.layers.length];
        
        // Mutate each value of the layers weights
        for (int i = 0; this.layers.length > i; i++) {
            Layer l = this.layers[i];
            float[][] newWeights = new float[l.OUTPUTSIZE][l.INPUTSIZE];
            float[][] currentWeights = l.getWeightMatrix();
            for (int r = 0; newWeights.length > r; r++) for (int c = 0; newWeights[r].length > c; c++)
                newWeights[r][c] = mutate.apply(currentWeights[r][c]);
            
            float[] newBiases = new float[l.getNeuronCount()];
            float[] currentBiases = l.getBiasVector();
            for (int k = 0; newBiases.length > k; k++) 
                newBiases[k] = mutate.apply(currentBiases[k]);

            newLayers[i] = new Layer(newWeights, newBiases, l.activation);
        }
        return new AIML.MLP(this.learningRate, newLayers);
    }
    
    public int getLayerCount() { return layers.length; }
    public int getNeuronsOfLayer(int i) { return layers[i].getNeuronCount(); }

    private class Layer {
        // Store the input & output sizes.
        private final int INPUTSIZE;
        private final int OUTPUTSIZE;

        // Weight matrix & bias vector.
        private final float[][] weights;
        private final float[] biases;

        // Store the input, output, & delta.
        private float[] input;
        private float[] output;
        private float[] delta;

        private Activation activation = MLP.LINEAR;

        public Layer(int inputSize, int outputSize, Activation activation) {
            this.INPUTSIZE = inputSize;
            this.OUTPUTSIZE = outputSize;
            
            this.activation = activation;

            this.weights = new float[outputSize][inputSize];
            this.biases = new float[outputSize];

            initWeights();
        }
        
        public Layer(float[][] weights, float[] biases, Activation activation) {
            if (weights == null || weights.length == 0)
                throw new IllegalArgumentException("Weights cannot be null or empty");  
 
            this.INPUTSIZE = weights[0].length;
            
            for (float[] row : weights) {
                if (row.length != this.INPUTSIZE)
                    throw new IllegalArgumentException("All weight rows must have same length");
            }
            
            this.OUTPUTSIZE = weights.length;
            
            this.activation = activation;
            
            this.weights = new float[weights.length][this.INPUTSIZE];
            for (int i = 0; i < weights.length; i++)
                System.arraycopy(weights[i], 0, this.weights[i], 0, this.INPUTSIZE);

            this.biases = java.util.Arrays.copyOf(biases, biases.length);
        }

        private void initWeights() {
            java.util.Random rand = new java.util.Random();
            for (int i = 0; i < OUTPUTSIZE; i++) {
                for (int j = 0; j < INPUTSIZE; j++) {
                    weights[i][j] = (float) (rand.nextGaussian() * 0.01);
                }
                biases[i] = 0.0f;
            }
        }

        public float[] forward(float[] input) {
            this.input = input;
            this.output = new float[OUTPUTSIZE];

            // Loop through each neuron
            for (int i = 0; i < OUTPUTSIZE; i++) {
                // Start with the bias.
                float sum = biases[i];
        
                // Calculate the weighted sum.
                for (int j = 0; j < INPUTSIZE; j++) sum += weights[i][j] * input[j];
                
                // Apply the activation function and store it in the output array.
                output[i] = activation.activate.apply(sum);
            }
            return output;
        }

        public float[] backward(float[] gradOutput, float learningRate) {
            float[] gradInput = new float[INPUTSIZE];
            delta = new float[OUTPUTSIZE];

            // Get the delta ( derivative ).
            for (int i = 0; i < OUTPUTSIZE; i++) 
                delta[i] = gradOutput[i] * activation.derivative.apply(output[i]);
            

            // Get the gradient.
            for (int i = 0; i < INPUTSIZE; i++) for (int j = 0; j < OUTPUTSIZE; j++) 
                gradInput[i] += delta[j] * weights[j][i];
                
            // Update weights and biases.
            for (int i = 0; i < OUTPUTSIZE; i++) {
                for (int j = 0; j < INPUTSIZE; j++) weights[i][j] -= learningRate * delta[i] * input[j];
                biases[i] -= learningRate * delta[i];
            }
            return gradInput;
        }
        
        public int getNeuronCount() { return this.biases.length; }
        public float[][] getWeightMatrix() { return this.weights; }
        public float[] getBiasVector() { return biases; }
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
