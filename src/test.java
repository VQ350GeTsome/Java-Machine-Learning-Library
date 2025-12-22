public class test {

    public static void main(String[] args) {
        
        AIML.MLP mlp = new AIML.MLP(0.05f, 4, 8, 6, 1);
        
        mlp.setAllLayersActivation(AIML.Activation.HARD_TANH);
        mlp.setFinalLayerActivation(AIML.Activation.HARD_SIGMOID);
        
        oddCountTest(mlp);
    }
    
    private static void oddCountTest(AIML.MLP mlp) {
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
        
        // Train
        for (int e = 0; 1000000 > e; e++) {
            int i = e % inputs.length; 
            mlp.train(inputs[i], targets[i]);
        }
        
        for (int i = 0; inputs.length > i; i++) {
            float[] out = mlp.forward(inputs[i]);
            float output = Math.round(out[0]);
            float target = targets[i][0];
            System.out.println(
                "Input: " + java.util.Arrays.toString(inputs[i]) + "\t" + 
                "Output: " + output +"\t" + 
                "Target: " + target + "\t" +
                "Conclusion? " + ((target == output)?"Correct":"Wrong")
            );
        }
    }
}
