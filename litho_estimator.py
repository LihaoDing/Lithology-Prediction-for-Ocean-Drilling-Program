import numpy as np
import pandas as pd
from table import table
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
# Add any other imports that you need here

# If you created custom transformers or helper functions, you can also add them to this file.


class LogTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, seed = 1e-5):
        self.seed=seed
        return None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return np.log(X+self.seed)


class ZeroTransform(TransformerMixin, BaseEstimator):
    """This transformer replaces negative values by zeros: elemental concentrations cannot be < zero."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X = np.where(X < 0, 0, X)
        return X
    

class ColorTransform(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        X['munsel_color1'] = X['munsel_color'].apply(lambda x: table[x][0])
        X['munsel_color2'] = X['munsel_color'].apply(lambda x: table[x][1])
        X['munsel_color3'] = X['munsel_color'].apply(lambda x: table[x][2])
        X = X.drop("munsel_color", axis=1)
        return X


class LithoEstimator:
    '''Used to predict lithology in IODP wells. The signature (method name, argument and return types) for the strict minimum number of methods needed are already written for you below.
    Simply complete the methods following your own notebook results. You can also add more methods than provided below in order to keep your code clean.'''

    def __init__(self, path:str='data/log_data.csv') -> None:
        '''The path is a path to the training file. The default is the file I gave you.
        You want to create an X_train, X_test, y_train and y_test following the same principle as in your
        notebook. You also want to define and train your estimator as soon as you create it.
        
        I recommend creatubg the following instance variables in your __init__ method:
        self.X_train, self.X_test, self.y_train, self.y_test
        self.encoder - the label encoder for your categories
        self.model - the entire trained model pipeline

        Note that this class should not handle hyperparameter searching or feature selection - if you did those in your Part B 
        simply use your best estimators.
        
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data2(path)
        self.X_train_trans, self.X_test_trans, self.y_train_trans, self.y_test_trans, self.encoder = self.load_data(path)
        self.model = self.create_pipeline()
        pass

    def load_data2(self, path):
        df = pd.read_csv(path)
        train_frac = int(df.shape[0]*.7)+1
        sorted_df = df.sort_values(by=['DEPTH_WMSF'], ascending=True)
        train_set = sorted_df.iloc[:train_frac]
        test_set = sorted_df.iloc[train_frac:]
        X_train = train_set.drop(["lithology"], axis=1).copy()
        y_train = train_set.lithology
        X_test = test_set.drop(["lithology"], axis=1).copy()
        y_test = test_set.lithology
        return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)

    def load_data(self, path):
        '''
        This function is to split the raw_data to train set and test set for X, y.

        Return the splited X_train, X_test, y_train, y_test
        '''
        df = pd.read_csv(path)
        label_encoder = LabelEncoder()
        label_encoder.fit(df['lithology'])
        df['y'] = label_encoder.transform(df['lithology'])
        train_frac = int(df.shape[0]*.7)+1
        sorted_df = df.sort_values(by=['DEPTH_WMSF'], ascending=True)
        train_set = sorted_df.iloc[:train_frac]
        test_set = sorted_df.iloc[train_frac:]
        X_train = train_set.drop(["lithology"], axis=1).copy()
        y_train = train_set.y
        X_test = test_set.drop(["lithology"], axis=1).copy()
        y_test = test_set.y
        return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test), label_encoder
    
    def create_pipeline(self):
        '''
        This function is used to generated a pipeline model.

        Return the generated pipepline model
        '''
        X_train_color = self.X_train_trans.copy()
        X_train_color['munsel_color1'] = X_train_color['munsel_color'].apply(lambda x: table[x][0])
        X_train_color['munsel_color2'] = X_train_color['munsel_color'].apply(lambda x: table[x][1])
        X_train_color['munsel_color3'] = X_train_color['munsel_color'].apply(lambda x: table[x][2])
        to_log = ['IDPH']
        not_to_log = ['HCGR', 'HURA', 'munsel_color1', 'munsel_color2', 'munsel_color3']
        to_log_pipe = make_pipeline(SimpleImputer(), ZeroTransform(), LogTransformer(seed=6e-3), RobustScaler())
        not_to_log_pipe = make_pipeline(SimpleImputer(), ZeroTransform(), RobustScaler())
        proc_pipe = ColumnTransformer([
            ('to_log_transformer', to_log_pipe, X_train_color[to_log].columns),
            ('not_to_log_transformer', not_to_log_pipe, X_train_color[not_to_log].columns)
        ])
        procession = make_pipeline(ColorTransform(), proc_pipe)
        svc = SVC(kernel='linear')
        svc_pipe = make_pipeline(procession, svc)
        svc_pipe.fit(self.X_train_trans, self.y_train_trans)
        return svc_pipe

    def x_test_score(self) -> np.float:
        '''Returns the F1 macro score of the X_test. This should be of type float.'''
        return f1_score(self.y_test_trans, self.model.predict(self.X_test_trans), average='macro')

    def get_Xs(self) -> (pd.DataFrame, pd.DataFrame):
        '''Returns the X_train and X_test. This method is already written for you.'''
        
        return self.X_train, self.X_test
    
    def get_ys(self) -> (pd.DataFrame, pd.DataFrame):
        '''Returns the y_train and y_test. This method is already written for you.'''

        return self.y_train, self.y_test

    def predict(self, path_to_new_file:str='data/new_data.csv') -> np.array:
        '''Uses the trained algorithm to predict and return the predicted labels on an unseen file.
        The default file is the unknown_data.csv file in your data folder.
        
        Return a numpy array (the default for the "predict()" function of sklearn estimator)'''
        new_samples = pd.read_csv(path_to_new_file)
        return self.encoder.inverse_transform(self.model.predict(new_samples))

    def get_model(self) -> Pipeline:
        '''returns the entire trained pipeline, i.e. your model.
        This will include the data preprocessor and the final estimator.'''
        return self.model