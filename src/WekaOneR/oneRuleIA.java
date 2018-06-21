/*
Nome: Gabriel Garcia Scanferla - 29880
      Gabriel Alves Moreira - 2017005113
      Gabriel de Faria Castro Moreira - 30388
 */
package WekaOneR;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;


public class oneRuleIA {
        
    public static void main(String[] args) throws Exception {
        
    ConverterUtils.DataSource source = new ConverterUtils.DataSource("treinaN.arff");
    Instances train = source.getDataSet();
    train.setClassIndex(11);

    ConverterUtils.DataSource source2 = new ConverterUtils.DataSource("TesteN.arff");
    Instances test = source2.getDataSet();
    test.setClassIndex(11); 
    
    Classifier oneRule = new OneR();
    
    oneRule.buildClassifier(train);
    
    Evaluation ultimo = new Evaluation(test);
            
    ultimo.evaluateModel(oneRule, test);
        System.out.println(ultimo.toSummaryString());
        System.out.println(ultimo.toClassDetailsString());
        System.out.println(ultimo.toMatrixString());
        System.out.println("Error rate: " + ultimo.errorRate());
    
    
    }
    
    
}
