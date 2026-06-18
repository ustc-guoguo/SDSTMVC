

def set_default_config(data_name):
    if data_name in ['BDGP']:
        return dict(
            dataset=data_name,
            n_samples=2500,
            n_classes=5,
            n_views=2,
            aligned_ratio=0.5,
            batch_size=3000,
            epochs=400, # 500
            lr=0.00005,
            weight_decay=0.01,
            tau=1.0,
            lambda1=1.0,
            lambda2=1.0,
            in_dim=[79, 1750],
            hidden_dim=2000,
            emb_dim=30,
            activation='leakyrelu',
            batchnorm=True,
        )
    elif data_name in ['HandWritten']:
        return dict(
            dataset=data_name,
            n_samples=2000,
            n_classes=10,
            n_views=2,
            aligned_ratio=0.5,
            batch_size=2000,
            epochs=400, # 500
            lr=1e-4,
            weight_decay=0.01,
            tau=1.0,
            lambda1=1.0,
            lambda2=1.0,
            in_dim = [240, 216],
            hidden_dim = 2000,
            emb_dim = 100,
            activation='leakyrelu',
            batchnorm=True,
        )
    elif data_name in ['Wiki']:
        return dict(
            dataset=data_name,
            n_samples=2866,
            n_classes=10,
            n_views=2,
            aligned_ratio=0.5,
            batch_size=3000,
            epochs=250, # 500
            lr=1e-4,
            weight_decay=0.01,
            tau=1.0,
            lambda1=1.0,
            lambda2=1.0,
            in_dim=[10, 128],
            hidden_dim=2000,
            emb_dim=20,
            activation='leakyrelu',
            batchnorm=True,
        )
    elif data_name in ['MNIST-USPS']:
        return dict(
            dataset=data_name,
            n_samples=5000,
            n_classes=10,
            n_views=2,
            aligned_ratio=0.5,
            batch_size=2000,
            epochs=400, # 500
            lr=0.0002,
            weight_decay=0.01,
            tau=0.8,
            lambda1=30.0,
            lambda2=10.0,
            in_dim=[784, 256],
            hidden_dim=2000,
            emb_dim=10,
            activation='leakyrelu',
            batchnorm=True,
        )
    elif data_name in ['NUS-WIDE']:
        return dict(
            dataset=data_name,
            n_samples=9000,
            n_classes=10,
            n_views=2,
            aligned_ratio=0.5,
            batch_size=2000,
            epochs=400, # 500
            lr=0.00006,
            weight_decay=0.01,
            tau=0.3,
            lambda1=50.0,
            lambda2=10.0,
            in_dim=[4096, 300],
            hidden_dim=2000,
            emb_dim=30,
            activation='leakyrelu',
            batchnorm=True,
        )
    elif data_name in ['Reuters_dim10']:
        return dict(
            dataset=data_name,
            n_samples=18758,
            n_classes=6,
            n_views=2,
            aligned_ratio=0.5,
            batch_size=2000,
            epochs=400, # 500
            lr=1e-4,
            weight_decay=0.01,
            tau=1.0,
            lambda1=1.0,
            lambda2=1.0,
            in_dim=[10, 10],
            hidden_dim=2000,
            emb_dim=10,
            activation='leakyrelu',
            batchnorm=True,
        )
    elif data_name in ['Hdigit']:
        return dict(
            dataset=data_name,
            n_samples=10000,
            n_classes=10,
            n_views=2,
            aligned_ratio=0.5,
            batch_size=2000,
            epochs=400, # 500
            lr=1e-4,
            weight_decay=0.01,
            tau=0.6,
            lambda1=10,
            lambda2=1.0,
            in_dim=[256, 784],
            hidden_dim=2000,
            emb_dim=10,
            activation='leakyrelu',
            batchnorm=True,
        )
    elif data_name in ['Deep Animal']:
        return dict(
            dataset=data_name,
            n_samples=10158,
            n_classes=50,
            n_views=2,
            aligned_ratio=0.5,
            batch_size=2000,
            epochs=80, # 100
            lr=1e-6,
            weight_decay=0.01,
            tau=1.0,
            lambda1=30.0,
            lambda2=5.0,
            in_dim=[4096, 4096],
            hidden_dim=2000,
            emb_dim=200,
            activation='leakyrelu',
            batchnorm=True,
        )
    else:
        """The default config."""
        print(f"The parameter configuration for dataset {data_name} has not been implemented. Using default configuration ...")
        return dict(
            dataset=data_name,
            n_samples=2000,
            n_classes=6,
            n_views=2,
            aligned_ratio=0.5,
            batch_size=2000,
            epochs=500,
            lr=1e-4,
            weight_decay=0.01,
            tau=1.0,
            lambda1=10.0,
            lambda2=1.0,
            in_dim=[100, 100],
            hidden_dim=2000,
            emb_dim=30,
            activation='leakyrelu',
            batchnorm=True,
        )

