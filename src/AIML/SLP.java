package AIML;

public class SLP implements AI {
    
    private final int inputSize, outputSize;
    
    private final float[][] weights;
    private final float[] biases;
    
    private AIML.Activation activation = AIML.Activation.HEAVISIDE_STEP;
    
    private final float learningRate;

    public SLP(float learningRate, int inputs, int outputs) {
        this.learningRate = learningRate;
        this.inputSize = inputs;
        this.outputSize = outputs;
        
        this.biases = new float[outputSize];
        this.weights = new float[outputSize][inputSize];
        this.initWeightsBiases();
    }
    
    private SLP(float learningRate, float[][] weights, float[] biases) {
        this.learningRate = learningRate;
        this.weights = weights;
        this.biases = biases;
        
        this.inputSize = weights[0].length;
        this.outputSize = weights.length;
    }
    
    private void initWeightsBiases() {
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; inputSize > j; j++) 
                weights[i][j] = (float) (MLP.RAND.nextGaussian() * 0.50);
            biases[i]  = (float) (MLP.RAND.nextGaussian() * 0.05);
        }
    }
    
    @Override
    public float[] forward(float[] input) {
        float[] out = new float[outputSize];
        for (int i = 0; outputSize > i; i++) {
            float sum = biases[i];
            for (int j = 0; inputSize > j; j++)
                sum += weights[i][j] * input[j];
            out[i] = activation.apply(sum);
        }
        return out;
    }

    @Override
    public void train(float[] input, float[] target) {
        float[] output = forward(input);

        float[] delta = new float[outputSize];

        for (int o = 0; o < outputSize; o++) {
            float error = target[o] - output[o];
            delta[o] = error * activation.derive(output[o]);
        }

        for (int o = 0; o < outputSize; o++) {
            for (int i = 0; i < inputSize; i++) {
                weights[o][i] += learningRate * delta[o] * input[i];
            }
            biases[o] += learningRate * delta[o];
        }
    }
    
    @Override
    public AIML.SLP uniformMutate(float mutation) {
        return this.mutate((a) -> {
                float mutate = (float) ((Math.random() * 2 - 1) * mutation);
                return a + mutate;
            }
        );
    }
    @Override
    public AIML.SLP gaussianMutate(float mutation) {
        return this.mutate((a) -> {
                float mutate = (float) (AIML.MLP.RAND.nextGaussian() * mutation);
                return a + mutate;
            }
        );
    }
    @Override
    public AIML.SLP mutate(AIML.MLP.SingleFloatFunc mutator) {
        float[][] newWeights = new float[outputSize][inputSize];
        float[] newBiases = new float[outputSize];
        
        for (int i = 0; weights.length > i; i++) { 
            for (int j = 0; weights[0].length > j; j++) 
                newWeights[i][j] = mutator.apply(weights[i][j]);
            newBiases[i] = mutator.apply(biases[i]);
        }
        return new AIML.SLP(this.learningRate, newWeights, newBiases);
    }
    
    public void changeActivation(AIML.Activation a) { this.activation = a; }   
}
