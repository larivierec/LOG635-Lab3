import java.util.Random;

/**
 * Created by Michel on 2015-07-13.
 */
public class Neurone {
    private double[] weight;
    private double w0;
    private static Random random = new Random();

    private double[] ai;
    private double dansJ;
    private double aj;
    private double deltaJ;
    private double deltaI;

    private boolean isI = false;

    public Neurone(double... vars){
        w0 = random.nextDouble();
        weight = new double[vars.length];
        for(int i =0; i< vars.length; i++){
            weight[i] = random.nextDouble();
        }
        ai = vars;
    }

    public double getOutput(){
        return aj;
    }

    public void newValues(double... vars){
        ai = vars;
    }

    public void computeAJ(){
        dansJ = getDansJ(ai);
        aj = getG(dansJ);
    }

    public void computeDeltaJ(double y){
        deltaJ = getGPrime(dansJ)*(y-aj);
    }

    public double getDeltaJ(){
        return deltaJ;
    }

    public void computeDeltaI(double deltaJ){
        isI = true;
        deltaI = getGPrime(dansJ)*getDansJ(vectorized(deltaJ));
    }

    public double getDeltaI(){
        return deltaI;
    }

    public void fixWeights(double alpha){
        for (int i = 0; i < weight.length; i++) {
            weight[i] += alpha * ai[i] * (isI?deltaI:deltaJ);
        }
        w0 += isI?deltaI:deltaJ;
        isI = false;
    }

    private double getDansJ(double[] vars){
        double value = 0;
        for(int i = 0; i < weight.length; i++){
            value += (vars[i] * weight[i]);
        }
        value += w0;
        return value;
    }

    private double getG(double val){
        return 1/(1 + Math.exp(val * -1));
    }

    private double getGPrime(double val){
        return (1- val)*val;
    }

    private double[] vectorized(double var){
        double[] vector = new double[ai.length];
        for(double d: vector){
            d = var;
        }
        return vector;
    }

}
