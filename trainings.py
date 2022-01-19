from keras.layers import Flatten, Dense, Reshape


net_configurations = [
#1-layer MLP with Relu activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'relu'],
      ['Dense', 8, 'softmax']],

#2-layer MLP with Relu activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'relu'],
      ['Dense', 1024, 'relu'],
      ['Dense', 8, 'softmax']],

#3-layer MLP with Relu activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'relu'],
      ['Dense', 1024, 'relu'],
      ['Dense', 512, 'relu'],
      ['Dense', 8, 'softmax']],

#1-layer MLP with tanh activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'tanh'],
      ['Dense', 8, 'softmax']],

#2-layer MLP with tanh activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'tanh'],
      ['Dense', 1024, 'tanh'],
      ['Dense', 8, 'softmax']],

#3-layer MLP with tanh activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'tanh'],
      ['Dense', 1024, 'tanh'],
      ['Dense', 512, 'tanh'],
      ['Dense', 8, 'softmax']],

#1-layer MLP with swish activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'swish'],
      ['Dense', 8, 'softmax']],

#2-layer MLP with swish activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'swish'],
      ['Dense', 1024, 'swish'],
      ['Dense', 8, 'softmax']],

#3-layer MLP with swish activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'swish'],
      ['Dense', 1024, 'swish'],
      ['Dense', 512, 'swish'],
      ['Dense', 8, 'softmax']],

#4-layer MLP with relu and linear end activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'relu'],
      ['Dense', 1024, 'relu'],
      ['Dense', 512, 'relu'],
      ['Dense', 256, 'linear'],
      ['Dense', 8, 'softmax']],


#4-layer MLP with relu and linear end activation
      [['Reshape', 32, 32],
      ['Dense', 2048, 'relu'],
      ['Dense', 1024, 'relu'],
      ['Dense', 512, 'relu'],
      ['Dense', 256, 'sigmoid'],
      ['Dense', 8, 'softmax']]
    ]
test_name = ['Relu-2', 'Relu-3', 'Relu-4', 'Tanh-2', 'Tanh-3', 'Tanh-4', 'Swish-2', 'Swish-3', 'Swish-4', 'ReluLin-5', 'ReluSig-5']
