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
    
    public void changeActivation(AIML.Activation a) { this.activation = a; }   
}
