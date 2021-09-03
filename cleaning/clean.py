import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer


class preprocessing:
    def show_data(self, data):
        return data.head()

    def check_na(self, data):
        """
        Check is na or null value present in dataset
        """
        return data.isna().sum()

    def fill_na(self, data, inplace=False):
        """
        fill all na or null value using SimpleImputer
        """
        col = data.columns
        impute = SimpleImputer()
        data = pd.DataFrame(impute.fit_transform(data), columns=col)
        return data

    def drop_col(self, data, col, inplace=False):
        """
        drop unwanted columns from dataset
        """
        data = data.drop(columns=col, inplace=inplace)
        return data

    def drop_missing(self, data, axis=0, inplace=False):
        """
        drop missing value columns from dataset
        """
        return data.dropna(axis=axis, inplace=inplace)

    def standard_scaler(self, data):
        """perform standardization, mean = 0 and S.D = 1
        fromula: z = (x - x_bar)/S.D
        Where, x is the original feature vector, x_bar is the mean of that feature vector, and S.D is its standard deviation."""
        scaler = StandardScaler()
        arr = scaler.fit_transform(data)
        return arr

    def normalize(self, data):
        """Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 and 1."""
        var = self.check_na(data).tolist()
        if sum(var) == 0:
            norm = Normalizer()
            arr = norm.fit_transform(data)
            return arr
        else:
            return "There is some na value present in the dataset, remove it and try again."
