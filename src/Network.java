/**
 * Created by Michel on 2015-07-20.
 */
public class Network {
    private int width = 90;
    private final int depth = 2;
    private double alpha = 0.8;
    private Neurone[][] brain;
    //private Neurone outputNeurone;
    private double[] outputWeight;

    private double output;

    public Network(double... vars) {
        //width = vars.length;
        brain = new Neurone[width][depth];
        double[][] temp = new double[2][width];
        for (int i = 0; i < width; i++) {
            brain[i][0] = new Neurone(vars);
            brain[i][0].computeAJ();
            temp[0][i] = brain[i][0].getOutput();
        }
        for (int j = 1; j < depth; j++){
            for(int i = 0; i < width; i++){
                brain[i][j] = new Neurone(temp[(j-1)%2]);
                brain[i][j].computeAJ();
                temp[j%2][i] = brain[i][j].getOutput();
            }
        }
        //outputNeurone = new Neurone(temp[(depth - 1)%2]);

        double out =0;
        for(int i = 0; i<width; i++){
            out += temp[(depth - 1)%2][i];
        }

        output = out;
    }

    public void calculate(double... vars){
        double[][] temp = new double[2][width];
        for (int i = 0; i < width; i++) {
            brain[i][0].newValues(vars);
            brain[i][0].computeAJ();
            temp[0][i] = brain[i][0].getOutput();
        }
        for (int j = 1; j < depth; j++){
            for(int i = 0; i< width; i++){
                brain[i][j].newValues(temp[(j-1)%2]);
                brain[i][j].computeAJ();
                temp[j%2][i] = brain[i][j].getOutput();
            }
        }
        //outputNeurone.newValues(temp[(depth - 1)%2]);

        double out =0;
        for(int i = 0; i<width; i++){
            out += temp[(depth - 1)%2][i];
        }
        if(outputWeight != null){
            for (int i = 0; i < width; i++) {
                outputWeight[i] = temp[(depth - 1)%2][i]/out;
            }
        }

        output = out;
    }

    public double getOutput(){
        return output;
    }

    public void learn(double y) {
        if(Math.abs(output - y) < 3){
            alpha = 0.3;
        }
        //double delta = Math.abs(y - output);
        //double a = delta > 10?0.9:delta>5?0.3:0.1;
        //outputNeurone.computeDeltaJ(y);
        //outputNeurone.fixWeights(alpha);
        double[] temp = new double[width];
        for (int i = 0; i < width; i++) {
            if(outputWeight == null) {
                brain[i][depth - 1].computeDeltaJ(y / width);
            }else{
                brain[i][depth - 1].computeDeltaJ(y * outputWeight[i]);
            }
            temp[i] = brain[i][depth - 1].getDeltaJ();
            brain[i][depth - 1].fixWeights(alpha);
        }
        for (int j = depth - 2; j >= 0; j--) {
            for (int i = 0; i < width; i++) {
                brain[i][j].computeDeltaI(temp[i]);
                temp[i]= brain[i][j].getDeltaI();
                brain[i][j].fixWeights(alpha);
            }
        }

        if(outputWeight == null && Math.abs(output - y) < 1){
            outputWeight = new double[width];
            alpha = 0.7;
        }
    }


}
