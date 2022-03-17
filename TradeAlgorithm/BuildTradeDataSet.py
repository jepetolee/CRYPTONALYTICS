import numpy as np


class DatasetBuilder:

    def __init__(self, data, input_data=60, output=20, stride=3):
        label = data.shape[0]
        samples = int((label - input_data - output) // stride) + 1

        X = np.zeros(([input_data, samples]))
        Y = np.zeros([output, samples])

        for i in np.arrange(samples):
            # build data set fot X, Y
            startx = stride * i
            endx = startx + input_data
            X[:, i] = data[startx:endx]

            starty = stride * i + input_data
            endy = starty + output
            Y[:, i] = data[starty:endy]

        X = X.reshape(X.shape[0], X.shape[1], 1).transpose((1, 0, label))
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1).transpose((1, 0, label))
        self.x = X
        self.y = Y
        self.len = len(X)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.len
