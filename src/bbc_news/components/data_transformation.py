from bbc_news.entity import DataTransformationConfig
import os
import pandas as pd
from bbc_news import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from bbc_news.utils import read_yaml, create_directories,load_bin,save_bin
import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        tokenizer = Tokenizer(num_words=self.config.params_max_features, split=' ')
        joblib.dump(filename=self.config.tokeniazer_path, value=tokenizer)
        

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    def replace_nan_num(self,dataset):
        numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>0 and dataset[feature].dtypes!='O']

        for feature in numerical_with_nan:
            ## We will replace by using median since there are outliers
            median_value=dataset[feature].median()
            
            ## create a new feature to capture nan values
            #dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
            dataset[feature].fillna(median_value,inplace=True)
        
        logger.info("Replaceed missing dataset with median")
        return dataset
    
    def correlation(self,dataset, threshold):
        col_corr = set()  # Set of all the names of correlated columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
        return col_corr

    def scaling(self,dataset):
        '''Scaling Feature'''
        
        scaling_feature=[feature for feature in dataset.columns if feature not in ['MouseID','cls'] ]
        scaler=MinMaxScaler()
        scaler.fit(dataset[scaling_feature])
        data = pd.concat([dataset[['MouseID','cls']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[scaling_feature]), columns=scaling_feature)],
                    axis=1)
        logger.info("Completed scaling dataset")
        return(data)
        
    def train_test_spliting(self,data):
        #data = pd.read_csv(self.config.data_path)
        
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
    
    
        
        
    def tokenize_and_pad_sequences(self,text_data,max_text_length):
        tokenizer = joblib.load(self.config.tokeniazer_path)
        tokenizer.fit_on_texts(text_data)
        sequences = tokenizer.texts_to_sequences(text_data)
        padded_sequences = pad_sequences(sequences, maxlen=max_text_length)
        return padded_sequences
    
        
    def transformation(self):
        data = pd.read_csv(self.config.train_data_path)
        logger.info("Converted Excel data to DataFrame")
        #encode the target data
        le=LabelEncoder()
        y= le.fit_transform(data['Category'])
        logger.info("Encoded the dependent variable ")
        
        # # drop the independent variable
        # data=data.drop(['Genotype', 'Treatment', 'Behavior', 'class'],axis=1)
        
        # # removing the missing values from numeric features
        # data=self.replace_nan_num(data)
        
        
        # #remove the correlated features
        # corr_features = self.correlation(data.drop(['MouseID','cls'],axis=1), 0.9)
        # data=data.drop(corr_features,axis=1)
        # logger.info("Droped highly correlated features")
        
        # #scaling the dependent variable
        # data=self.scaling(data)
        
        X=self.tokenize_and_pad_sequences(text_data=data['Text'],max_text_length=self.config.params_max_text_length)
        
        
        self.train_test_spliting(pd.concat([pd.DataFrame(X),pd.DataFrame(y,columns=['cat'])],axis=1))
        
        
        
