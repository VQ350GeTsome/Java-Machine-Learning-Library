package AIML;

import java.util.function.Function;
        
public final class Activation {
    
    private final Function activation, derivative;
    float coeff;
    
    public Activation(Function<Float, Float> activation, Function<Float, Float> derivative) {
        this.activation = activation; this.derivative = derivative;
    }
    
    public Activation(float coeff, Function<Float, Float> activation, Function<Float, Float> derivative) {
        this.activation = activation; this.derivative = derivative;
        this.coeff = coeff;
    }

    public float apply(float f) { return (float) this.activation.apply(f); }
    public float derive(float f) { return (float) this.derivative.apply(f); }

    //<editor-fold defaultstate="collapsed" desc=" Common Activation Functions">
    // The actual activation functions
    public static final Activation 
            LINEAR = new Activation(
                (a) -> (a), 
                (a) -> 1.0f
            ),
            RELU = new Activation(
                (a) -> a > 0 ? a : 0.0f, 
                (a) -> a > 0 ? 1.0f : 0.0f
            ),
            LEAKY_RELU = new Activation(
                (a) -> a >= 0 ? a : 0.01f * a, 
                (a) -> a >= 0 ? 1.0f : 0.01f
            ),
            SIGMOID = new Activation(
                (a) -> (float) (1.0 / (1.0 + Math.exp(-a))),
                (a) -> {
                    float s = (float)(1.0 / (1.0 + Math.exp(-a)));
                    return s * (1 - s);
                }
            ),
            HARD_SIGMOID = new Activation(
                    (a) -> {
                        if (-2.5 > a) return 0.0f;
                        if (a  > 2.5) return 1.0f;
                        return (a * 0.2f) + 0.5f;
                    },
                    (a) -> { 
                        if (2.5 < a || a < -2.5) return 0.0f;
                        else return 0.2f;
                    }
            ),
            TANH = new Activation(
                (a) -> (float)Math.tanh(a),
                (a) -> {
                    float t = (float)Math.tanh(a);
                    return 1 - t * t;
                }
            ),
            HARD_TANH = new Activation(
                    (a) -> {
                        if (a  > 1) return 1.0f;
                        if (-1 > a) return -1.0f;
                        return a;
                    },
                    (a) -> {
                        if (1 < a || a < -1) return 0.0f;
                        return 1.0f;
                    }
            ),
            HEAVISIDE_STEP = new Activation(
                    (a) -> {
                        if (a > 0) return 1.0f; 
                        else return 0.0f;
                    },
                    (a) -> 0.0f
            ),
            SOFT_SIGN = new Activation(
                    (a) -> a / (1 + Math.abs(a)),
                    (a) ->  {
                        if (a > 0) return 1.0f / ((1 + a) * (1 + a)); 
                        if (a < 0) return 1.0f / ((1 - a) * (1 - a));
                        return 1.0f; 
                    }
            );
    //</editor-fold>
}
