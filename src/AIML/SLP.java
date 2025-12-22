package AIML;

public class SLP {
    
    private final int INPUTS, OUTPUTS;
    
    private float[] weights, biases;
    
    private AIML.Activation STEP = AIML.Activation.HEAVISIDE_STEP;

    public SLP(int inputs, int outputs) {
        this.INPUTS = inputs;
        this.OUTPUTS = outputs;
        this.initWeightsBiases();
    }
    
    private void initWeightsBiases() {
        for (int i = 0; i < OUTPUTS; i++) {
            weights[i] = (float) (MLP.rand.nextGaussian() * 0.50);
            biases[i]  = (float) (MLP.rand.nextGaussian() * 0.05);
        }
    }
    
    public float[] decide(float[] inputs) {
        float sum = 0;
        for (int i = 0; INPUTS > i; i++) {
            sum += inputs[i] * weights[i];
        }
    }
    
}
