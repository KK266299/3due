python ue_generate.py \
    dataset=brats19 \
    task.run_name=unet_boundary_noise_0_with_2 \
    method=unet_boundary_noise \
    task=brats19_ue \
    training.epochs=100 \
    training.batch_size=8 \
    training.eval_batch_size=8 \
    training.gpu_ids=[3] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.boundary_width=1 \
    ue.algorithm.params.interior_weight=0 \
    ue.algorithm.params.boundary_weight=1.0 \
    ue.algorithm.params.surrogate_step=10 \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10