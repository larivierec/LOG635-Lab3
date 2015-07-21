/**
 * Created by Michel on 2015-07-20.
 */
public class Network {
    private int width;
    private final int depth = 8;
    private final double alpha = 0.5;
    private Neurone[][] brain;

    private double output;

    public Network(double... vars) {
        width = vars.length;
        brain = new Neurone[width][depth];
        double[][] temp = new double[2][vars.length];
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
        double out =0;
        for(int i = 0; i<vars.length; i++){
            out += temp[(depth - 1)%2][i];
        }
        output = out;
    }

    public void calculate(double... vars){
        double[][] temp = new double[2][vars.length];
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
        double out =0;
        for(int i = 0; i<vars.length; i++){
            out += temp[(depth - 1)%2][i];
        }
        output = out;
    }

    public double getOutput(){
        return output;
    }

    public void learn(double y) {
        double[] temp = new double[width];
        for (int i = 0; i < width; i++) {
            brain[i][depth - 1].computeDeltaJ(y/width);
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
    }


}
