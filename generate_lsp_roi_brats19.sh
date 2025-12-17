python ue_generate.py \
  dataset=brats19 \
  task=brats19_ue \
  method=lsp \
  ue.algorithm.kind=training_free \
  ue.algorithm.name=lsp \
  ue.algorithm.params.epsilon=0.0313725 \
  ue.algorithm.params.seed=0 \
  ue.algorithm.params.noise_frame_size=32 \
  ue.algorithm.params.class_sep=10.0 \
  ue.algorithm.params.roi_mode=binary \
  ue.algorithm.params.num_labels=4 \
  ue.key.type=samplewise \
  ue.key.from=field \
  ue.key.field=case_id \
  ue.io.strategy=files \
  ue.io.dtype=int8