package ML;

public class MLP implements AI {

    private Layer[] layers;
    private final float learningRate;
    
    /**
     * Constructor that takes layer sizes.
     * For example: {@code new Perceptron(3, 4, 2)} creates:
     *      Layer 1: 3 inputs => 4 outputs
     *      Layer 2: 4 inputs => 2 outputs
     *
     * @param learningRate The rate at which this {@link AI} will learn.
     * @param sizes The sizes of each layer (inputSize, hiddenSize..., outputSize).
     */
    public MLP(float learningRate, int... sizes) { 
        if (sizes.length < 2)  throw new IllegalArgumentException("Need at least input and output size");
        
        this.learningRate = learningRate;
        
        layers = new Layer[sizes.length - 1];
        for (int i = 0; layers.length > i; i++) layers[i] = new Layer(sizes[i], sizes[i + 1], AI.Activation.LINEAR);
    }
    
    public MLP(float learningRate, Layer[] layers) {
        this.learningRate = learningRate;
        this.layers = layers;
    }
    
    /**
     * Forward pass through all layers.
     * 
     * @param input Input column {@link Matrix.Matrix} ({@link Matrix.Matrix} of shape inputSize × 1).
     * @return Output column {@link Matrix.Matrix} ({@link Matrix.Matrix} of shape outputSize × 1).
     */
    @Override
    public float[] forward(float[] input) {
        float[] result = input;
        
        // Pass the output of each layer as the input of the next until
        // all layers have been fowarded.
        for (int i = 0; layers.length > i; i++) result = layers[i].forward(result);
        
        return result;
    }

    @Override
    public void train(float[] input, float[] target) { this.train(input, target, (o, t) -> t - o); }
    @Override
    public void train(float[] input, float[] target, BinaryFloatFunc errorFunction) {
        // Forward pass.
        float[] result = input;
        for (Layer layer : layers) result = layer.forward(result);

        // Compute output error.
        float[] error = new float[target.length];
        for (int i = 0; i < target.length; i++) error[i] = errorFunction.apply(target[i], result[i]);
        
        // Backprop.
        float[] delta = error;
        for (int i = layers.length - 1; i >= 0; i--) delta = layers[i].backward(delta, learningRate);
    }
    
    @FunctionalInterface public interface SingleFloatFunc { float apply(float x); }
    @FunctionalInterface public interface BinaryFloatFunc { float apply(float a, float b); }
    
    public void setAllLayersActivation(AI.Activation a) {
        for (int i = 0; this.layers.length > i; i++) this.layers[i].activation = a;
    }
    public void setAllLayersExceptFinalActivation(AI.Activation a) {
        for (int i = 0; this.layers.length - 1 > i; i++) this.layers[i].activation = a;
    }
    public void setFinalLayerActivation(AI.Activation a) { this.layers[this.layers.length - 1].activation = a; }
    public void setLayerActivation(int layer, AI.Activation a) { this.layers[layer].activation = a; }
    
    @Override
    public MLP uniformMutate(float mutation) {
        return this.mutate((w) -> {
                float mutate = (float) ((Math.random() * 2 - 1) * mutation);
                return w + mutate;
            }
        );
    }
    @Override
    public MLP gaussianMutate(float mutation) {
        return this.mutate((w) -> {
                float mutate = (float) (AI.RAND.nextGaussian() * mutation);
                return w + mutate;
            }
        );
    }
    @Override
    public MLP mutate(SingleFloatFunc mutator) {
        // What the new layers of the new neural net will be
        Layer[] newLayers = new Layer[this.layers.length];
        
        // Mutate each value of the layers weights
        for (int i = 0; this.layers.length > i; i++) {
            Layer l = this.layers[i];
            float[][] newWeights = new float[l.outputSize][l.inputSize];
            float[][] currentWeights = l.getWeightMatrix();
            for (int r = 0; newWeights.length > r; r++) for (int c = 0; newWeights[r].length > c; c++)
                newWeights[r][c] = mutator.apply(currentWeights[r][c]);
            
            float[] newBiases = new float[l.getNeuronCount()];
            float[] currentBiases = l.getBiasVector();
            for (int k = 0; newBiases.length > k; k++) 
                newBiases[k] = mutator.apply(currentBiases[k]);

            newLayers[i] = new Layer(newWeights, newBiases, l.activation);
        }
        return new MLP(this.learningRate, newLayers);
    }
    
    public int getLayerCount() { return layers.length; }
    public int getNeuronsOfLayer(int i) { return layers[i].getNeuronCount(); }

    public class Layer {
        // Store the input & output sizes.
        private final int inputSize, outputSize;

        // Weight matrix & bias vector.
        private final float[][] weights;
        private final float[] biases;

        // Store the input, output, delta, & z.
        private float[] input, output, delta, z;

        private AI.Activation activation = AI.Activation.LINEAR;

        public Layer(int inputSize, int outputSize, AI.Activation activation) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            
            this.activation = activation;

            this.weights = new float[outputSize][inputSize];
            this.biases = new float[outputSize];

            initWeightsBiases();
        }
        
        public Layer(float[][] weights, float[] biases, AI.Activation activation) {
            if (weights == null || weights.length == 0)
                throw new IllegalArgumentException("Weights cannot be null or empty");  
 
            this.inputSize = weights[0].length;
            
            for (float[] row : weights) {
                if (row.length != this.inputSize)
                    throw new IllegalArgumentException("All weight rows must have same length");
            }
            
            this.outputSize = weights.length;
            
            this.activation = activation;
            
            this.weights = new float[weights.length][this.inputSize];
            for (int i = 0; i < weights.length; i++)
                System.arraycopy(weights[i], 0, this.weights[i], 0, this.inputSize);

            this.biases = java.util.Arrays.copyOf(biases, biases.length);
        }

        private void initWeightsBiases() {
            for (int i = 0; i < outputSize; i++) {
                for (int j = 0; j < inputSize; j++) {
                    weights[i][j] = (float) (AI.RAND.nextGaussian() * 0.5);
                }
                biases[i] = (float) (AI.RAND.nextGaussian() * 0.05);
            }
        }

        public float[] forward(float[] input) {
            this.input = input;
            this.z = new float[outputSize];
            this.output = new float[outputSize];

            // Loop through each neuron
            for (int i = 0; i < outputSize; i++) {
                // Start with the bias.
                float sum = biases[i];
        
                // Calculate the weighted sum.
                for (int j = 0; j < inputSize; j++) sum += weights[i][j] * input[j];
                
                // Store the sum in the z array, & the activated
                // output in the output array.
                z[i] = sum;
                output[i] = activation.apply(sum);
            }
            return output;
        }

        public float[] backward(float[] gradOutput, float learningRate) {
            float[] gradInput = new float[inputSize];
            delta = new float[outputSize];

            // Get the delta ( derivative ).
            for (int i = outputSize - 1; i >= 0; i--) 
                delta[i] = gradOutput[i] * activation.derive(z[i]);

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
        
        public int getNeuronCount() { return this.biases.length; }
        public float[][] getWeightMatrix() { return this.weights; }
        public float[] getBiasVector() { return biases; }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("{\n");
            sb.append("  Activation Function : ").append(activation.toString()).append("\n");
            sb.append("  weights = [\n");

            for (float[] row : weights) {
                sb.append("    ");
                for (double w : row) {
                    sb.append(String.format("%8.4f ", w));
                }
                sb.append("\n");
            }

            sb.append("  ]\n");
            sb.append("  biases = [ ");
            for (double b : biases) {
                sb.append(String.format("%8.4f ", b));
            }
            sb.append("]\n");
            sb.append("}");

            return sb.toString();
        }
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        for (int i = 0; layers.length > i; i++) {
            sb.append("Layer ").append(i+1).append(" ");
            sb.append(layers[i].toString()).append("\n\n");
        }
        
        return sb.toString();
    }
}
