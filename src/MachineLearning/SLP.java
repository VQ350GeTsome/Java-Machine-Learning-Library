package MachineLearning;

public class SLP implements AI {
    
    private final int inputSize, outputSize;
    
    private final float[][] weights;
    private final float[] biases;
    
    private AI.Activation activation = AI.Activation.HEAVISIDE_STEP;
    
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
        this.train(input, target, (a, b) -> a - b);
    }
    @Override
    public void train(float[] input, float[] target, MLP.BinaryFloatFunc errorFunction) {
        float[] output = forward(input);

        float[] delta = new float[outputSize];

        for (int o = 0; outputSize > o; o++) {
            float error = errorFunction.apply(target[o], output[o]);
            delta[o] = error * activation.derive(output[o]);
        }

        for (int o = 0; outputSize > o; o++) {
            for (int i = 0; i < inputSize; i++) {
                weights[o][i] += learningRate * delta[o] * input[i];
            }
            biases[o] += learningRate * delta[o];
        }
    }
    
    @Override
    public SLP uniformMutate(float mutation) {
        return this.mutate((a) -> {
                float mutate = (float) ((Math.random() * 2 - 1) * mutation);
                return a + mutate;
            }
        );
    }
    @Override
    public SLP gaussianMutate(float mutation) {
        return this.mutate((a) -> {
                float mutate = (float) (MLP.RAND.nextGaussian() * mutation);
                return a + mutate;
            }
        );
    }
    @Override
    public SLP mutate(MLP.SingleFloatFunc mutator) {
        float[][] newWeights = new float[outputSize][inputSize];
        float[] newBiases = new float[outputSize];
        
        for (int i = 0; weights.length > i; i++) { 
            for (int j = 0; weights[0].length > j; j++) 
                newWeights[i][j] = mutator.apply(weights[i][j]);
            newBiases[i] = mutator.apply(biases[i]);
        }
        return new SLP(this.learningRate, newWeights, newBiases);
    }
    
    public void changeActivation(AI.Activation a) { this.activation = a; }   
}
