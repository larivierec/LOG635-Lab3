import java.io.BufferedReader;
import java.io.FileReader;

public class Main {

    private int mTotalLineCount;
    private double[][] mDoubleInputData;
    private double[] ys;
    private Network brain;

    public static void main(String[] args) {
        Main m = new Main();
        m.getData();
        m.brain = new Network(m.mDoubleInputData[0]);
        m.brain.learn(m.ys[0]);
        System.out.println(m.brain.getOutput());
        for(int i = 2; i < m.mDoubleInputData.length; i += 2){
            m.brain.calculate(m.mDoubleInputData[i]);
            m.brain.learn(m.ys[i]);
            System.out.println(m.brain.getOutput());
        }
    }

    public void getData(){
        BufferedReader br = null;
        String line = "";
        String splitBy = ";";
        int lineCount = 0;
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
                Object[] array = line.split(splitBy);
                if(counter == 0){
                    this.mDoubleInputData = new double[lineCount][array.length - 1];
                    this.ys = new double[lineCount];
                }else{
                    for(int j = 0; j < array.length; j++){
                        if(j < array.length - 1) {
                            this.mDoubleInputData[counter - 1][j] = Double.parseDouble((String) array[j]);
                        }else{
                            this.ys[counter - 1] = Double.parseDouble((String) array[j]);
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
    }
}
