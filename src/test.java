
import AIML.Activation;

public class test {

    public static void main(String[] args) {
        
        AIML.MLP mlp = new AIML.MLP(0.05f, 4, 8, 6, 1);

        mlp.setAllLayersActivation(AIML.Activation.HARD_TANH);
        mlp.setFinalLayerActivation(AIML.Activation.HARD_SIGMOID);
        
        AIML.SLP slp = new AIML.SLP(0.05f, 2, 1);
        
        slp.changeActivation(Activation.RELU);
        
        andTest(slp);
    }
    
    private static void train(AIML.AI ai, float[][] inputs, float[][] targets) {
        for (int e = 0; 1000000 > e; e++) {
            int i = e % inputs.length; 
            ai.train(inputs[i], targets[i]);
        }
    }
    private static void display(AIML.AI ai, float[][] inputs, float[][] targets) {
        for (int i = 0; inputs.length > i; i++) {
            float[] out = ai.forward(inputs[i]);
            float output = Math.round(out[0] * 1000) / 1000.0f;
            float target = targets[i][0];
            System.out.println(
                "Input: " + java.util.Arrays.toString(inputs[i]) + "\t" + 
                "Output: " + output +"\t" + 
                "Target: " + target + "\t" +
                "Conclusion? " + ((Math.abs(target - output) < 1.0e-2)?"Correct":"Wrong")
            );
        }
    }
    
    private static void andTest(AIML.AI ai) {
        float[][] inputs = { 
            {0,0},
            {0,1},
            {1,0},
            {1,1}
        };
        
        float[][] targets = {
            {0},
            {0},
            {0},
            {1}
        };
        
        train(ai, inputs, targets);
        display(ai, inputs, targets);
    }
    private static void oddCountTest(AIML.AI ai) {
        float[][] inputs = {
            {0,0,0,0},
            {0,0,0,1},
            {0,0,1,0},
            {0,0,1,1},
            {0,1,0,0},
            {0,1,0,1},
            {0,1,1,0},
            {0,1,1,1},
            {1,0,0,0},
            {1,0,0,1},
            {1,0,1,0},
            {1,0,1,1},
            {1,1,0,0},
            {1,1,0,1},
            {1,1,1,0},
            {1,1,1,1}
        };

        float[][] targets = { 
            {0},{1},{1},{0}, 
            {1},{0},{0},{1}, 
            {1},{0},{0},{1}, 
            {0},{1},{1},{0} 
        };
        
        train(ai, inputs, targets);
        display(ai, inputs, targets);
    }
}
