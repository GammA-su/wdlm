# WDLM Benchmark Summary

Checkpoint: `runs\toy_state_head_baseline\checkpoint_best.pt`

| split | text_loss | exact_state_accuracy | paraphrase_delta_cosine | example_count |
| --- | ---: | ---: | ---: | ---: |
| val | 2.6399 | 0.0000 | 0.9737 | 7 |
| test_iid | 2.4473 | 0.0000 | 0.9689 | 8 |
| test_lexical_ood | 2.9417 | 0.0000 | 0.9677 | 8 |
| test_paraphrase_ood | 3.2294 | 0.0000 | 0.9696 | 8 |
| test_compositional_ood | 3.1536 | 0.0000 | 0.9477 | 7 |
| test_length_ood | 2.8932 | 0.0000 | 0.9694 | 8 |
