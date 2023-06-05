from types import SimpleNamespace

cfg = SimpleNamespace(**{})

cfg.exp_name = 'exp106'
cfg.apex = True  # [True, False]

######################
# Globals #
######################
cfg.seed = 42
cfg.epochs = 400
cfg.folds = [3]  # [0, 1, 2, 3, 4]
cfg.external = True  # [True, False]
cfg.use_sampler = True  # [True, False]

######################
# Data #
######################
cfg.train_datadir = ""  # Path("../input/birdclef-2023/train_audio")

######################
# Dataset #
######################
cfg.period = 20  # [5, 10, 20, 30]
cfg.frames = -1  # [-1, 480000, 640000, 960000]

cfg.use_pcen = False
cfg.n_mels = 128  # [64, 128, 224, 256]
cfg.fmin = 20  # [20, 50]
cfg.fmax = 16000  # [14000, 16000]
cfg.n_fft = 2048  # [1024, 2048]
cfg.hop_length = 512  # [320, 512]
cfg.sample_rate = 32000
cfg.secondary_coef = 0.0

cfg.target_columns = [
    'abethr1', 'abhori1', 'abythr1', 'afbfly1', 'afdfly1', 'afecuc1',
    'affeag1', 'afgfly1', 'afghor1', 'afmdov1', 'afpfly1', 'afpkin1',
    'afpwag1', 'afrgos1', 'afrgrp1', 'afrjac1', 'afrthr1', 'amesun2',
    'augbuz1', 'bagwea1', 'barswa', 'bawhor2', 'bawman1', 'bcbeat1',
    'beasun2', 'bkctch1', 'bkfruw1', 'blacra1', 'blacuc1', 'blakit1',
    'blaplo1', 'blbpuf2', 'blcapa2', 'blfbus1', 'blhgon1', 'blhher1',
    'blksaw1', 'blnmou1', 'blnwea1', 'bltapa1', 'bltbar1', 'bltori1',
    'blwlap1', 'brcale1', 'brcsta1', 'brctch1', 'brcwea1', 'brican1',
    'brobab1', 'broman1', 'brosun1', 'brrwhe3', 'brtcha1', 'brubru1',
    'brwwar1', 'bswdov1', 'btweye2', 'bubwar2', 'butapa1', 'cabgre1',
    'carcha1', 'carwoo1', 'categr', 'ccbeat1', 'chespa1', 'chewea1',
    'chibat1', 'chtapa3', 'chucis1', 'cibwar1', 'cohmar1', 'colsun2',
    'combul2', 'combuz1', 'comsan', 'crefra2', 'crheag1', 'crohor1',
    'darbar1', 'darter3', 'didcuc1', 'dotbar1', 'dutdov1', 'easmog1',
    'eaywag1', 'edcsun3', 'egygoo', 'equaka1', 'eswdov1', 'eubeat1',
    'fatrav1', 'fatwid1', 'fislov1', 'fotdro5', 'gabgos2', 'gargan',
    'gbesta1', 'gnbcam2', 'gnhsun1', 'gobbun1', 'gobsta5', 'gobwea1',
    'golher1', 'grbcam1', 'grccra1', 'grecor', 'greegr', 'grewoo2',
    'grwpyt1', 'gryapa1', 'grywrw1', 'gybfis1', 'gycwar3', 'gyhbus1',
    'gyhkin1', 'gyhneg1', 'gyhspa1', 'gytbar1', 'hadibi1', 'hamerk1',
    'hartur1', 'helgui', 'hipbab1', 'hoopoe', 'huncis1', 'hunsun2',
    'joygre1', 'kerspa2', 'klacuc1', 'kvbsun1', 'laudov1', 'lawgol',
    'lesmaw1', 'lessts1', 'libeat1', 'litegr', 'litswi1', 'litwea1',
    'loceag1', 'lotcor1', 'lotlap1', 'luebus1', 'mabeat1', 'macshr1',
    'malkin1', 'marsto1', 'marsun2', 'mcptit1', 'meypar1', 'moccha1',
    'mouwag1', 'ndcsun2', 'nobfly1', 'norbro1', 'norcro1', 'norfis1',
    'norpuf1', 'nubwoo1', 'pabspa1', 'palfly2', 'palpri1', 'piecro1',
    'piekin1', 'pitwhy', 'purgre2', 'pygbat1', 'quailf1', 'ratcis1',
    'raybar1', 'rbsrob1', 'rebfir2', 'rebhor1', 'reboxp1', 'reccor',
    'reccuc1', 'reedov1', 'refbar2', 'refcro1', 'reftin1', 'refwar2',
    'rehblu1', 'rehwea1', 'reisee2', 'rerswa1', 'rewsta1', 'rindov',
    'rocmar2', 'rostur1', 'ruegls1', 'rufcha2', 'sacibi2', 'sccsun2',
    'scrcha1', 'scthon1', 'shesta1', 'sichor1', 'sincis1', 'slbgre1',
    'slcbou1', 'sltnig1', 'sobfly1', 'somgre1', 'somtit4', 'soucit1',
    'soufis1', 'spemou2', 'spepig1', 'spewea1', 'spfbar1', 'spfwea1',
    'spmthr1', 'spwlap1', 'squher1', 'strher', 'strsee1', 'stusta1',
    'subbus1', 'supsta1', 'tacsun1', 'tafpri1', 'tamdov1', 'thrnig1',
    'trobou1', 'varsun2', 'vibsta2', 'vilwea1', 'vimwea1', 'walsta1',
    'wbgbir1', 'wbrcha2', 'wbswea1', 'wfbeat1', 'whbcan1', 'whbcou1',
    'whbcro2', 'whbtit5', 'whbwea1', 'whbwhe3', 'whcpri2', 'whctur2',
    'wheslf1', 'whhsaw1', 'whihel1', 'whrshr1', 'witswa1', 'wlwwar',
    'wookin1', 'woosan', 'wtbeat1', 'yebapa1', 'yebbar1', 'yebduc1',
    'yebere1', 'yebgre1', 'yebsto1', 'yeccan1', 'yefcan', 'yelbis1',
    'yenspu1', 'yertin1', 'yesbar1', 'yespet1', 'yetgre1', 'yewgre1',
    'nocall'
    ]

cfg.bird2id = {b: i for i, b in enumerate(cfg.target_columns)}
cfg.id2bird = {i: b for i, b in enumerate(cfg.target_columns)}

######################
# Loaders #
######################
cfg.loader_params = {
    "train": {
        "batch_size": 32,
        "pin_memory": True,
        "num_workers": 8,
        "drop_last": True,
        "shuffle": True if not cfg.use_sampler else False
    },
    "valid": {
        "batch_size": 64,
        "pin_memory": True,
        "num_workers": 8,
        "shuffle": False
    }
}

######################
# Model #
######################
cfg.backbone = 'eca_nfnet_l0'
cfg.use_imagenet_weights = True
cfg.num_classes = 264
cfg.in_channels = 1
cfg.lr_max = 2.5e-4
cfg.lr_min = 1e-7
cfg.weight_decay = 1e-6
cfg.max_grad_norm = 10
cfg.early_stopping = 20
cfg.mixup_p = 1.0

cfg.pretrained_weights = True
cfg.pretrained_path = '../models/exp089_eca_nfnet_l0/fold_0_model.bin'
cfg.model_output_path = f"../models/{cfg.exp_name}_{cfg.backbone}"
