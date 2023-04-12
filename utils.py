import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def perform_preprocessing(options):
    """ Performs full pre-processing pipeline """
    
    def count_missing_percentage(df):
        """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
        # Get Missing Velues per feature (exclude features that are complete)
        feature_missing = train.isnull().sum().sort_values(ascending = False)
        feature_missing = feature_missing[feature_missing > 0]
        percent = round(feature_missing/len(df)*100, 2)
        return pd.concat([feature_missing, percent], axis=1, keys=['Total','Percent'])
    
    def count_outliers(df):
        """ Returns table feature-wise outlier count """
        out = pd.Series(index=df.columns)
        total_num = df.shape[0]
        for column in df:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lowest_val = Q1 - 1.5*IQR
            highest_val = Q3 + 1.5*IQR
            
            flter_df = df[column].copy()
            flter_df = flter_df[flter_df > lowest_val]
            flter_df = flter_df[flter_df < highest_val]

            outlier_percentage = round(100 * (total_num - flter_df.shape[0]) / total_num, 2) 
            out[column] = outlier_percentage

        return out.sort_values(ascending = False)

    def drop_outliers_IQR(df):
        """ Drops samples that include outliers """
        
        def drop_outliers(ddf, field_name):
            """ Drops outliers for specific column """
            df = ddf.copy()
            iqr = 1.5 * (np.percentile(df[field_name], 75) - np.percentile(df[field_name], 25))
            df.drop(df[df[field_name] > (iqr + np.percentile(df[field_name], 75))].index, inplace=True)
            df.drop(df[df[field_name] < (np.percentile(df[field_name], 25) - iqr)].index, inplace=True)
            return df
        
        for column in df:
            if df[column].dtypes != np.int64:
                continue
            df = drop_outliers(df, column)
        return df
    
    def apply_categorical_encoding(dff, categorical_columns):
        """ For given list of categorical features it replaces categorical columns with encoded ste of columns """
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder()

        df = dff.copy().reset_index()
        encoded_dataframe = pd.DataFrame()
        for column in categorical_columns:
            encoded_data = encoder.fit_transform(df[[column]]).toarray()
            encoded_df = pd.DataFrame(encoded_data, columns=['{}_'.format(column) + x for x in encoder.categories_[0]])
            encoded_dataframe = pd.concat([encoded_dataframe, encoded_df], axis=1)
            
        df.drop(categorical_columns, axis=1, inplace=True)
        df = pd.concat([df, encoded_dataframe], axis=1)
        return df
    
    # 1. Load data
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    print("Step #1. train: ", train.shape)
    print("Step #1. test: ", test.shape)
    
    # 2. Remove "bad" features
    ## 2.0 Removes "Id"
    train.drop(["Id"], axis=1, inplace = True)
    test_id = test["Id"]
    test.drop(["Id"], axis=1, inplace = True)
    
    print("Step #2.0. train: ", train.shape)
    print("Step #2.0. test: ", test.shape)
    
    ## 2.1 Removes features with too many Missed Values
    drop_threshold = 20 # Drop features that have more than 20% of missing data
    
    low_data_features = count_missing_percentage(train)
    low_data_features = low_data_features[low_data_features["Percent"] >  drop_threshold]
    drop_cols = low_data_features.T.columns
    print("Features to be dropped: {}".format(list(drop_cols)))
    
    train.drop(drop_cols, axis=1, inplace = True)
    test.drop(drop_cols, axis=1, inplace = True)
    
    print("Step #2.1. train: ", train.shape)
    print("Step #2.1. test: ", test.shape)
    
    ## 2.2 Removes features with too many outliers
    drop_outliers_threshold = 50 # Drop all features that have more than 50% of outliers

    numerical_train = train.loc[:, train.dtypes == np.int64]
    outliers = count_outliers(numerical_train)
    zero_outliers = outliers.index[outliers > drop_outliers_threshold]
    print("Outlier features: {}".format(list(zero_outliers)))

    train.drop(zero_outliers, axis=1, inplace=True)
    test.drop(zero_outliers, axis=1, inplace=True)
    
    print("Step #2.2. train: ", train.shape)
    print("Step #2.2. test: ", test.shape)
    
    # 3. Fill Missing Values
    mode_values = train.mode().iloc[0]
    train.fillna(mode_values, inplace = True)
    test.fillna(mode_values, inplace = True)
    
    print("Step #3. train: ", train.shape)
    print("Step #3. test: ", test.shape)
    
    # 4. Remove Outliers (samples with outliers)
    ## APLLY ONLY TO TRAIN AS IT ONLY HELPS DURING TRAINING!
    ## WE SHALL PROCESS ALL TESTING SAMPLES, SO WE CANT DROP THEM
    train = drop_outliers_IQR(train)
    
    print("Step #4. train: ", train.shape)
    print("Step #4. test: ", test.shape)
    
    # 5. Separate traget and train
    target = train["SalePrice"]
    train.drop(["SalePrice"], axis=1, inplace=True)
    
    print("Step #5. train: ", train.shape)
    print("Step #5. test: ", test.shape)
    
    # 6. Encode categorical features
    categorical_columns = train.columns[train.dtypes == "object"]
    train = apply_categorical_encoding(train, categorical_columns)
    test = apply_categorical_encoding(test, categorical_columns)
    
    print("Step #6.1. train: ", train.shape)
    print("Step #6.1. test: ", test.shape)
    
    # PS: Test dataset has more categorical values than Training causing varience in col number. 
    # Fix was found here: https://www.kaggle.com/code/ekam123/using-categorical-data-with-one-hot-encoding/notebook 
    # and here https://www.kaggle.com/discussions/getting-started/50008#284798 
    train, test = train.align(test, join='left', axis=1)

    print("Step #6.2. train: ", train.shape)
    print("Step #6.2. test: ", test.shape)
    
    # PS: Align causes Missing Values in test dataset
    # TEMPORAL solution is to fill it with modes from train dataset (Don't know what to do with them actually)
    print("Train missing values: ", train.isna().sum().sum())
    print("Test missing values: ", test.isna().sum().sum())
    
    mode_values = train.mode().iloc[0]
    train.fillna(mode_values, inplace = True)
    test.fillna(mode_values, inplace = True)
    
    print("Train missing values: ", train.isna().sum().sum())
    print("Test missing values: ", test.isna().sum().sum())
    
    # 7. Reduce dimentionality with PCA
    ## PS: Data must be scaled before PCA!
    if options["pca"]:
        
        scaler = StandardScaler()
        scaler.fit(train)

        train = pd.DataFrame(scaler.transform(train))
        test = pd.DataFrame(scaler.transform(test))
        
        # pca = PCA(n_components=200)
        # pca.fit_transform(train)

        # # plot scree plot
        # plt.plot(range(1, 201), pca.explained_variance_ratio_, 'ro-', linewidth=2)
        # plt.title('PCA component values. Top-10 are the most important')
        # plt.xlabel('Principal Component')
        # plt.ylabel('Explained Variance Ratio')
        # plt.show()
        
        ## I chose PCA_N = 20
        pca = PCA(n_components=20)
        pca.fit(train)
        
        train = pd.DataFrame(pca.transform(train))
        test = pd.DataFrame(pca.transform(test))
        
        print("Step #6. train: ", train.shape)
        print("Step #6. test: ", test.shape)
    
    # 8. Scale
    scaler = StandardScaler()
    scaler.fit(train)

    train = pd.DataFrame(scaler.transform(train))
    test = pd.DataFrame(scaler.transform(test))
    
    print("Step #7. train: ", train.shape)
    print("Step #7. test: ", test.shape)
    
    return train, test, target, test_id