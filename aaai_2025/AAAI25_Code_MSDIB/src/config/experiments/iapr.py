from config.defaults import Experiment, DDC, Fusion, MLP, Loss, Dataset, MSDIB, Optimizer
iapr = Experiment(
    dataset_config=Dataset(name="iapr"),
    model_config=MSDIB(
        backbone_configs=(
            MLP(input_size=(100,)),
            MLP(input_size=(100,)),
        ),
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        projector_config=None,
        cm_config=DDC(n_clusters=6),
        loss_config=Loss(
            # MASG：multi-aspect self-guidance
            funcs="ddc_1|ddc_2|ddc_3|MASG",
            delta=20.0
        ),
        optimizer_config=Optimizer(scheduler_step_size=50, scheduler_gamma=0.1)
    ),
)
