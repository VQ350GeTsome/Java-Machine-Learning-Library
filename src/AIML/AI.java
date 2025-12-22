package AIML;

public interface AI {
    
    public abstract float[] forward(float[] inputs);
    
    public abstract void train(float[] inputs, float[] targets);
    
}
