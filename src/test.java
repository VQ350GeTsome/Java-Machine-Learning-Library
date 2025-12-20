
public class test {

    // XOR test
    public static void main(String[] args) {
        
        AIML.MLP mlp = new AIML.MLP(0.05f, 2, 2, 1);
        
        mlp.setAllLayerActivation(AIML.MLP.SIGMOID);
        
        float [][] inputs = {
            { 0 , 0 },
            { 1 , 0 },
            { 0 , 1 },
            { 1 , 1 }
        };
        
        float[][] targets = { 
            { 1 }, 
            { 0 },
            { 0 },
            { 1 }
        };
        
        // Train
        for (int e = 0; 5000000 > e; e++) 
            mlp.train(inputs[e % 4], targets[e % 4]);
        
        for (int i = 0; i < 4; i++) {
            float[] out = mlp.forward(inputs[i]);
            System.out.println(
                    "Input: " + java.util.Arrays.toString(inputs[i]) + "\tOutput: " + java.util.Arrays.toString(out) + 
                    "\tTarget: " + java.util.Arrays.toString(targets[i])
            );
        }
    }
}
