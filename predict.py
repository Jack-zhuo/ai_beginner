import numpy as np

data = np.loadtxt('test.csv', delimiter=',')


def calculate(features):
   model = np.array([
        [-0.1466920023623092],
        [2.536187820507188],
        [34.49488640015812]
    ])
   result = np.dot(features, model)
   print(result)

   return 1





if __name__ == '__main__':
    features = data[:, 1:3]
    label = data[:, 0]

    features = np.append(features,np.ones(shape=(len(features),1)),axis=1)
    calculate(features)
