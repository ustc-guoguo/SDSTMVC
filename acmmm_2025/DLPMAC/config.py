from easydict import EasyDict
config = EasyDict()
'''3Sources'''
# config.input_features1 =3560
# config.input_features2 =3631
# config.enhidden_features = [2000, 320, 50,6]
# config.dehidden_features1 = [50, 320, 2000,3560]
# config.dehidden_features2 = [50, 320, 2000,3631]
# config.classes = 6
'''BBCsports'''
# config.input_features1 =2582
# config.input_features2 =2544
# config.enhidden_features = [1500, 200, 50,5]
# config.dehidden_features1 = [50, 200, 1500,2582]
# config.dehidden_features2 = [50, 200, 1500,2544]
# config.classes = 5
'''Caltech101'''
config.input_features1 =1984
config.input_features2 =512
config.enhidden_features = [500, 320, 50,5]
config.dehidden_features1 = [50, 320, 500,1984]
config.dehidden_features2 = [50, 320, 500,512]
config.classes = 20
'''noisymnist'''
# config.input_features1 =784
# config.input_features2 =784
# config.enhidden_features = [500, 320, 50,10]
# config.dehidden_features1 = [50, 320, 500,784]
# config.dehidden_features2 = [50, 320, 500,784]
# config.classes = 10
'''ORL_mtv'''
# config.input_features1 =400
# config.input_features2 =400
# config.enhidden_features = [300, 150, 50,10]
# config.dehidden_features1 = [50, 150, 300,400]
# config.dehidden_features2 = [50, 150, 300,400]
# config.classes = 40
'''Caltech101_7'''
# config.input_features1 =1984
# config.input_features2 =512
# config.enhidden_features = [500, 320, 50,5]
# config.dehidden_features1 = [50, 320, 500,1984]
# config.dehidden_features2 = [50, 320, 500,512]
# config.classes = 7
'''scene15'''
# config.input_features1 =20
# config.input_features2 =59
# config.enhidden_features = [100, 100, 50,15]
# config.dehidden_features1 = [50, 100, 100,20]
# config.dehidden_features2 = [50, 100, 100,59]
# config.classes = 15
'''BDGP'''
# config.input_features1 = 1750
# config.input_features2 =79
# config.enhidden_features = [500, 200, 200,10]
# config.dehidden_features1 = [200, 200, 500,1750]
# config.dehidden_features2 = [200, 200, 500,79]
# config.classes = 5
'''HandWritten'''
# config.input_features1 =47
# config.input_features2 =240
# config.enhidden_features = [500, 320, 50,10]
# config.dehidden_features1 = [50, 320, 500,47]
# config.dehidden_features2 = [50, 320, 500,240]
# config.classes = 10
'''flower17'''
# config.input_features1 =1360
# config.input_features2 =1360
# config.enhidden_features = [1000, 200, 50,10]
# config.dehidden_features1 = [50, 200, 1000,1360]
# config.dehidden_features2 = [50, 200, 1000,1360]
# config.classes = 17
'''Prokaryotic'''
# config.input_features1 =393
# config.input_features2 =438
# config.enhidden_features = [300, 150, 50,10]
# config.dehidden_features1 = [50, 150, 300,393]
# config.dehidden_features2 = [50, 150, 300,438]
# config.classes = 4
'''yale_mtv'''
# config.input_features1 =4096
# config.input_features2 =3304
# config.enhidden_features = [1500, 200, 50,15]
# config.dehidden_features1 = [50, 200, 1500,4096]
# config.dehidden_features2 = [50, 200, 1500,3304]
# config.classes = 15
'''Reuters_dim10'''
# config.input_features1 =10
# config.input_features2 =10
# config.enhidden_features = [200, 100,50,10]
# config.dehidden_features1 = [50, 100, 200,10]
# config.dehidden_features2 = [50, 100, 200,10]
# config.classes = 6
'''MSRCV1'''
# config.input_features1 =576
# config.input_features2 =512
# config.enhidden_features = [500, 200,50,10]
# config.dehidden_features1 = [50, 200, 500,576]
# config.dehidden_features2 = [50, 200, 500,512]
# config.classes = 7
'''20news'''
# config.input_features1 =2000
# config.input_features2 =2000
# config.enhidden_features = [1500, 1000,200,10]
# config.dehidden_features1 = [200, 1000, 1500,2000]
# config.dehidden_features2 = [200, 1000, 1500,2000]
# config.classes = 5
'''100leaves'''
# config.input_features1 =64
# config.input_features2 =64
# config.enhidden_features = [200, 200, 50,10]
# config.dehidden_features1 = [50, 200, 200,64]
# config.dehidden_features2 = [50, 200, 200,64]
# config.classes = 100
'''BBC4'''
# config.input_features1 =4659
# config.input_features2 =4633
# config.enhidden_features = [2000, 1000, 500,10]
# config.dehidden_features1 = [500, 1000, 2000,4659]
# config.dehidden_features2 = [500, 1000, 2000,4633]
# config.classes = 5
'''NUSWIDE'''
# config.input_features1 =226
# config.input_features2 =145
# config.enhidden_features = [500, 200, 200,15]
# config.dehidden_features1 = [200, 200, 500,226]
# config.dehidden_features2 = [200, 200, 500,145]
# config.classes = 29
'''ALOI'''
# config.input_features1 =77
# config.input_features2 =64
# config.enhidden_features = [500, 200, 200,10]
# config.dehidden_features1 = [200, 200, 500,77]
# config.dehidden_features2 = [200, 200, 500,64]
# config.classes = 100
'''Wikipedia-test'''
# config.input_features1 =128
# config.input_features2 =10
# config.enhidden_features = [500, 200, 200,10]
# config.dehidden_features1 = [200, 200, 500,128]
# config.dehidden_features2 = [200, 200, 500,10]
# config.classes = 10
'''Moives'''
# config.input_features1 =1878
# config.input_features2 =1398
# config.enhidden_features = [1000, 500, 200,15]
# config.dehidden_features1 = [200, 500, 1000,1878]
# config.dehidden_features2 = [200, 500, 1000,1398]
# config.classes = 17


config.prepoch=100
config.lr = 1e-3
config.momentum = 0.9
config.weight_decay = 0
config.w_v = 0

config.print_step = 10
config.tensorboard_step = 100
config.load_iter = 0
config.train_iters = 5000
config.is_train = True
config.use_cuda = True