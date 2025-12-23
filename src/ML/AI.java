package ML;

public interface AI {
    
    // Random used for guassian distribution.
    public static final java.util.Random RAND = new java.util.Random();
    
    public abstract float[] forward(float[] inputs);
    
    // Training methods
    public abstract void train(float[] inputs, float[] targets);
    public abstract void train(float[] inputs, float[] targets, MLP.BinaryFloatFunc errorFunc);
    
    // Mutations
    public abstract AI uniformMutate(float mutation);
    public abstract AI gaussianMutate(float mutation);
    public abstract AI mutate(MLP.SingleFloatFunc mutator);
    
    // Activation functions
    public static final class Activation {
        private final String name;
        private final MLP.SingleFloatFunc activation, derivative;

        public Activation(String name, MLP.SingleFloatFunc activation, MLP.SingleFloatFunc derivative) {
            this.name = name;
            this.activation = activation; this.derivative = derivative;
        }

        public float apply(float f) { return (float) this.activation.apply(f); }
        public float derive(float f) { return (float) this.derivative.apply(f); }

        //<editor-fold defaultstate="collapsed" desc=" Common Activation Functions">
        // The actual activation functions
        public static final Activation 
                LINEAR = new Activation(
                    "Linear",
                    (a) -> (a), 
                    (a) -> 1.0f
                ),
                RELU = new Activation(
                    "ReLU",
                    (a) -> a > 0 ? a : 0.0f, 
                    (a) -> a > 0 ? 1.0f : 0.0f
                ),
                LEAKY_RELU = new Activation(
                    "Leaky ReLU",
                    (a) -> a >= 0 ? a : 0.01f * a, 
                    (a) -> a >= 0 ? 1.0f : 0.01f
                ),
                SIGMOID = new Activation(
                    "Sigmoid",
                    (a) -> (float) (1.0 / (1.0 + Math.exp(-a))),
                    (a) -> {
                        float s = (float)(1.0 / (1.0 + Math.exp(-a)));
                        return s * (1 - s);
                    }
                ),
                HARD_SIGMOID = new Activation(
                    "Hard Sigmoid",
                    (a) -> {
                        if (-2.5 > a) return 0.0f;
                        if (a  > 2.5) return 1.0f;
                        return (a * 0.2f) + 0.5f;
                    },
                    (a) -> (2.5 < a || a < -2.5) ? 0.0f : 0.2f
                ),
                TANH = new Activation(
                    "Hyperbolic Tangent",
                    (a) -> (float)Math.tanh(a),
                    (a) -> {
                        float t = (float)Math.tanh(a);
                        return 1 - t * t;
                    }
                ),
                HARD_TANH = new Activation(
                    "Hard tanh",
                    (a) -> {
                        if (a  > 1) return 1.0f;
                        if (-1 > a) return -1.0f;
                        return a;
                    },
                    (a) ->  (1 < a || a < -1) ? 0.0f : 1.0f                    
                ),
                HEAVISIDE_STEP = new Activation(
                    "Heaviside Step",
                    (a) -> (a > 0) ? 1.0f : 0.0f,
                    (a) -> 0.0f
                ),
                SOFT_SIGN = new Activation(
                    "Soft Sign",
                    (a) -> a / (1 + Math.abs(a)),
                    (a) ->  {
                        if (a > 0) return 1.0f / ((1 + a) * (1 + a)); 
                        if (a < 0) return 1.0f / ((1 - a) * (1 - a));
                        return 1.0f; 
                    }
                );
        //</editor-fold>
        
        @Override 
        public String toString() { return this.name; }
    }

}
