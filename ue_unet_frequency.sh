python ue_generate.py \
    dataset=flare21 \
    task=flare21_ue \
    task.run_name=noise_slice_freq_4_255_hard \
    method=noise_slice_frequence \
    training.epochs=100 \
    training.batch_size=8 \
    training.gpu_ids=[0] \
    ue.key.type=samplewise \
    ue.key.from=field \
    ue.key.field=case_id \
    ue.algorithm.params.epsilon=0.0156863 \
    ue.algorithm.params.noise_step=1 \
    ue.algorithm.params.surrogate_step=10 \
    ue.algorithm.params.roi_aware=true \
    ue.algorithm.params.soft_edge=false \
    ue.io.save_from_epoch=50 \
    ue.io.save_every=10 \
    ue.surrogates.s_seg.in_channels=1 \
    ue.surrogates.s_seg.num_classes=5

bash train_flare21_frequency.sh

python ue_generate.py \
  dataset=flare21 \
  task.run_name=noise_coherent_4_255 \
  method=noise_coherent \
  task=flare21_ue \
  training.epochs=100 \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  training.batch_size=8 \
  training.gpu_ids=[0] \
  ue.algorithm.params.epsilon=0.0156863 \
  ue.algorithm.params.surrogate_step=10 \
  ue.io.save_from_epoch=50 \
  ue.io.save_every=10 \
  ue.surrogates.s_seg.in_channels=1 \
  ue.surrogates.s_seg.num_classes=5