import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class Main {

    private int mTotalLineCount;
    private double[][] mDoubleInputData;
    private double[][] normalizeData;
    private double[] mins;
    private double[] maxs;
    private double[] ys;
    private double[] normalizeYs;
    private Network brain;

    private int succesfulTest = 0;

    public static void main(String[] args) {
        Main m = new Main();
        m.getData("Donnees_sources.csv", true);
        System.out.println(Arrays.toString(m.mins));
        System.out.println(Arrays.toString(m.maxs));
        /*
        for (int i = 0; i < m.mDoubleInputData.length; i++) {
            System.out.println(Arrays.toString(m.mDoubleInputData[i]));
            System.out.println(Arrays.toString(m.normalizeData[i]));
        }
        */
        m.brain = new Network(m.normalizeData[0]);
        m.brain.learn(m.normalizeYs[0]);
        int i = 1;
        while(m.succesfulTest < 50){
            m.brain.calculate(m.normalizeData[i]);
            m.brain.learn(m.normalizeYs[i]);
            if(Math.abs(m.brain.getOutput() - m.normalizeYs[i]) > 0.05) {
                m.succesfulTest = m.succesfulTest <= 0? 0: m.succesfulTest - 1;
            }else{
                m.succesfulTest++;
            }
            System.out.print("Expected: " + m.ys[i] + " final: ");
            System.out.println(((m.brain.getOutput()*m.maxs[m.maxs.length-1])+m.mins[m.mins.length -1]));
            i = (i + 1) % m.normalizeData.length;
        }
        System.out.println("***********starting evaluation******************");


        m.getData("Exemple_evaluation.csv", false);
        for (int k = 0; k < m.normalizeData.length; k++) {
            m.brain.calculate(m.normalizeData[k]);
            System.out.println(((m.brain.getOutput()*m.maxs[m.maxs.length-1])+m.mins[m.mins.length -1]));
        }

    }

    public void getData(String file, boolean withResults) {
        BufferedReader br = null;
        String line = "";
        String splitBy = ";";
        int lineCount = 0;
        int length = 0;
        try {

            String mDataFileName = file;
            br = new BufferedReader(new FileReader(mDataFileName));
            while ((line = br.readLine()) != null) {
                lineCount++;
            }
            this.mTotalLineCount = --lineCount;
            br.close();

            int counter = 0;
            br = new BufferedReader(new FileReader(mDataFileName));
            while ((line = br.readLine()) != null) {
                Object[] array;
                array = line.split(splitBy);
                if (counter == 0) {
                    length = withResults ? array.length - 1 : array.length;
                    this.mDoubleInputData = new double[lineCount][length];
                    this.normalizeData = new double[lineCount][length];
                    if (withResults) {
                        this.mins = new double[array.length];
                        for (int i = 0; i < mins.length; i++) {
                            mins[i] = Double.MAX_VALUE;
                        }
                        this.maxs = new double[array.length];
                        for (int i = 0; i < maxs.length; i++) {
                            maxs[i] = Double.MIN_VALUE;
                        }
                        this.ys = new double[lineCount];
                        this.normalizeYs = new double[lineCount];
                    }
                } else {
                    for (int j = 0; j < array.length; j++) {
                        if (j < length) {
                            this.mDoubleInputData[counter - 1][j] = Double.parseDouble((String) array[j]);
                            if (mins[j] > this.mDoubleInputData[counter - 1][j])
                                mins[j] = this.mDoubleInputData[counter - 1][j];
                            if (maxs[j] < this.mDoubleInputData[counter - 1][j])
                                maxs[j] = this.mDoubleInputData[counter - 1][j];
                        } else if (withResults) {
                            this.ys[counter - 1] = Double.parseDouble((String) array[j]);
                            if (mins[j] > this.ys[counter - 1]) mins[j] = this.ys[counter - 1];
                            if (maxs[j] < this.ys[counter - 1]) maxs[j] = this.ys[counter - 1];
                        }
                    }
                }
                counter++;
            }

            /*
            int outputLinecount = 0;
            String outputLine = "";

            br = new BufferedReader(new FileReader(mTargetOutputFileName));
            while (br.readLine()!= null) {
                outputLinecount++;
            }
            br.close();
            int outputCounter = 0;
            br = new BufferedReader(new FileReader(mTargetOutputFileName));
            while ((outputLine = br.readLine())!= null) {
                Object[] array = outputLine.split(splitBy);
                if(outputCounter == 0){
                    this.mDoubleOutputData = new double[outputLinecount][array.length];
                }
                for(int j = 0; j < array.length; j++){
                    this.mDoubleOutputData[outputCounter][j] = Double.parseDouble((String) array[j]);
                }

                outputCounter++;
            }
            this.mInputMatrix = new Matrix(mDoubleInputData);
            this.mOutputMatrix = new Matrix(mDoubleOutputData);
            */
        } catch (Exception e) {
            e.printStackTrace();
        }

        //normalization
        for (int j = 0; j < length; j++) {
            for (int i = 0; i < mDoubleInputData.length; i++) {
                normalizeData[i][j] = (mDoubleInputData[i][j] - mins[j]) / maxs[j];
            }
        }
        if (withResults) {
            for (int i = 0; i < ys.length; i++) {
                normalizeYs[i] = (ys[i] - mins[mins.length - 1]) / maxs[maxs.length - 1];
            }
        }
    }

    private class myCounter{
        private int cnt;
        public myCounter(){
            cnt = 1;
        }
        public void plus(){
            cnt++;
        }
        public String toString(){
            return Integer.toString(cnt);
        }
    }
}
