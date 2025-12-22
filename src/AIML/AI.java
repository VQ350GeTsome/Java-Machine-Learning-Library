package AIML;

public interface AI {
    
    public abstract float[] forward(float[] inputs);
    
    public abstract void train(float[] inputs, float[] targets);
    
    // Mutations
    public abstract AI uniformMutate(float mutation);
    public abstract AI gaussianMutate(float mutation);
    public abstract AI mutate(AIML.MLP.SingleFloatFunc mutator);
}
