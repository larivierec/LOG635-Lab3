import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;

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
        m.getData();
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
        System.out.println(m.brain.getOutput());
        //TODO changer for pour while avec test
        int i = 2;
        while(i < m.normalizeData.length && m.succesfulTest < 10){
            m.brain.calculate(m.normalizeData[i]);
            if(Math.abs(m.brain.getOutput() - m.normalizeYs[i]) > 0.08) {
                m.brain.learn(m.normalizeYs[i]);
                m.succesfulTest = 0;
            }else{
                m.succesfulTest++;
            }
            //System.out.println(m.brain.getOutput());
            System.out.println(((m.brain.getOutput()*m.maxs[m.maxs.length-1])+m.mins[m.mins.length -1]));
            i+=2;
        }
        System.out.println(m.succesfulTest);
    }

    public void getData(){
        BufferedReader br = null;
        String line = "";
        String splitBy = ";";
        int lineCount = 0;
        int length = 0;
        try{

            String mDataFileName = "Donnees_sources.csv";
            br = new BufferedReader(new FileReader(mDataFileName));
            while ((line = br.readLine())!= null) {
                lineCount++;
            }
            this.mTotalLineCount = --lineCount;
            br.close();

            int counter = 0;
            br = new BufferedReader(new FileReader(mDataFileName));
            while ((line = br.readLine())!= null) {
                Object[] array;
                array = line.split(splitBy);
                if(counter == 0){
                    length = array.length - 1;
                    this.mDoubleInputData = new double[lineCount][array.length - 1];
                    this.normalizeData = new double[lineCount][array.length - 1];
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
                }else {
                    for (int j = 0; j < array.length; j++) {
                        if (j < array.length - 1) {
                            this.mDoubleInputData[counter - 1][j] = Double.parseDouble((String) array[j]);
                            if (mins[j] > this.mDoubleInputData[counter - 1][j])
                                mins[j] = this.mDoubleInputData[counter - 1][j];
                            if (maxs[j] < this.mDoubleInputData[counter - 1][j])
                                maxs[j] = this.mDoubleInputData[counter - 1][j];
                        } else {
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
        }catch(Exception e){
            e.printStackTrace();
        }

        //normalization
        for(int j =0;j<length;j++){
            for(int i =0;i<mDoubleInputData.length;i++){
                normalizeData[i][j] = (mDoubleInputData[i][j] - mins[j])/maxs[j];
            }
        }
        for (int i = 0; i < ys.length; i++) {
            normalizeYs[i] = (ys[i] - mins[mins.length - 1])/maxs[maxs.length - 1];
        }
    }
}
