import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


class autoencoder_kai(BaseEstimator, TransformerMixin):
    """
    This class takes an input df_input (potentially with many columns), 
    and transform it into an output df_output with fewer columns. 

    Inputs:

    """
    def __init__(self):
        return
    

    def calculate_intermediate_layers_units(self):
        """
        Start from n_features_input, 
        Each step: divide by step_size,
        Till we reach n_features_output. 
        """
        layers = []
        
        n_units = self.n_features_input // self.step_size
        while n_units > self.n_features_output:
            # print(n_units)
            layers.append(n_units)
            n_units = n_units // self.step_size
        layers.append(self.n_features_output)
        
        self.layers = layers
        return
    
    def create_encoder_layers(self):
        # encoder
        encoder = Sequential()
        for i, n_units in enumerate(self.layers[1:]):
            if i == 0: # first layer, need to specify input_shape
                encoder.add(Dense(units=n_units, activation='relu', input_shape=(self.n_features_input, )))
            elif i == len(self.layers) - 1: # last layer, use linear activation
                encoder.add(Dense(units=n_units, activation='linear', name='bottleneck'))
            else: # middle layers
                encoder.add(Dense(units=n_units, activation='relu'))
        self.encoder = encoder
        return
    
    def create_decoder_layers(self):
        # decoder
        decoder = Sequential()
        for n_units in self.layers[::-1][1:-1]:
            decoder.add(Dense(units=n_units, activation='relu'))
        decoder.add(Dense(units=self.n_features_input, activation='sigmoid'))
        self.decoder = decoder
        return


    def fit(self, X, n_features_output=2, step_size=4, verbose=True):

        # convert df_input into np.array
        if isinstance(X, pd.DataFrame):
            if is_numeric_dtype(X):
                self.X = np.array(X)
            else:
                raise ValueError('df_input must be all numeric type!')
        elif isinstance(X, (np.ndarray, np.generic)):
            if np.ndim(X) == 2:
                self.X = X
            else:
                raise ValueError('df_input must be 2-dim numpy array!')


        self.n_features_input = self.X.shape[1]
        self.n_features_output = n_features_output
        self.step_size = step_size
        self.verbose=verbose




        # calcualte the layers
        self.calculate_intermediate_layers_units()
        self.create_encoder_layers()
        self.create_decoder_layers()

        # autoencoder
        self.autoencoder = Sequential([self.encoder, self.decoder])
        self.autoencoder.compile(loss='mse', optimizer='adam')
        if self.verbose:
            print(self.autoencoder.summary(expand_nested=True))

        # fit autoencoder
        self.autoencoder.fit(
            x=self.X,
            y=self.X,
            batch_size=128,
            epochs=200,
            callbacks=[EarlyStopping(monitor='loss', patience=5)],
        )


        return self
    

    def transform(self, X):
        x_encoded = self.encoder.predict(X)
        return x_encoded
    






if __name__ == '__main__':

    from tensorflow.keras.datasets.mnist import load_data

    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(60_000, 784) / 256

    ae = autoencoder_kai()
    ae.fit(X=x_train)
    x_encoded = ae.transform(X=x_train)


    import matplotlib.cm as cm
    def plot_transformed(x_transformed, ax, title):
        colors = cm.rainbow(np.linspace(0, 1, 10))
        for i, target_name, color in zip(range(10), range(10), colors):
            ax.scatter(
                x_transformed[y_train==i, 0], 
                x_transformed[y_train==i, 1],
                color=color,
                alpha=1,
                s=0.3,
                label=target_name,
            )
        ax.set_xlabel('feature 1')
        ax.set_ylabel('feature 2')
        ax.set_title(title)
        return

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    plot_transformed(x_encoded, ax, 'autoencoder')
    plt.show()
