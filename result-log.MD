## Slot by 2 task

### Result on dev set
epoch = 5, 30, 20 for conll, snips, atis
<!-- &#8593;↑ ⬆ -->
| Methods     | metric | wdecay |   conll   | snips     | atis      |
| ----------- | :----- | :----: | :-------: | :-------- | :-------- |
| Bert base   | Sent   |        |           | 90.86     | 93.00     |
| SlotBy2Task |        |        |           | **91.43** | 93.00     |
|             |
| Bert base   | Intent |        |           | 99.00     | 98.00     |
| SlotBy2Task |        |        |           | **99.14** | 98.00     |
|             |
| Bert base   | Slot   |        |   94.78   | 95.93     | 98.28     |
| SlotBy2Task |        |        | **95.09** | **96.41** | **98.42** |

### Result on test set
<!-- &#8593;↑ ⬆ -->
| Methods     | metric | wdecay |   conll   |       snips | atis      |
| ----------- | :----- | :----: | :-------: | ----------: | :-------- |
| Bert base   | Sent   |        |           |       92.00 | 87.57     |
| SlotBy2Task |        |        |           |   **92.43** | **88.24** |
|             |
| Bert base   | Intent |        |           | ! **98.57** | 97.42     |
| SlotBy2Task |        |        |           |       98.29 | **97.98** |
|             |
| Bert base   | Slot   |        |   91.09   |       96.60 | 95.77     |
| SlotBy2Task |        |        | **91.33** |   **96.82** | **95.87** |

snips - libs/JointBERT/snips_models/ep30-Wdec
```log
INFO:trainer:***** Running evaluation on test dataset *****
INFO:trainer:  Num examples = 700
INFO:trainer:  Batch size = 64
INFO:trainer:***** Eval results *****
INFO:trainer:  intent_acc = 0.9857142857142858
INFO:trainer:  loss = 0.3001327548514713
INFO:trainer:  sementic_frame_acc = 0.92
INFO:trainer:  slot_f1 = 0.9659977703455964
INFO:trainer:  slot_precision = 0.9638487208008899
INFO:trainer:  slot_recall = 0.9681564245810056
INFO:root:                            precision    recall  f1-score   support

                     state     0.9667    0.9831    0.9748        59
               object_type     0.9816    0.9877    0.9846       162
               entity_name     0.8438    0.8182    0.8308        33
                  playlist     0.9538    0.9612    0.9575       129
                    artist     0.9626    0.9626    0.9626       107
                   service     1.0000    1.0000    1.0000        24
                 timeRange     0.9720    0.9720    0.9720       107
                      city     0.9831    0.9667    0.9748        60
               best_rating     1.0000    1.0000    1.0000        43
             object_select     1.0000    1.0000    1.0000        40
               object_name     0.9128    0.9252    0.9189       147
              rating_value     1.0000    1.0000    1.0000        80
     condition_description     1.0000    1.0000    1.0000        28
                   country     1.0000    0.9545    0.9767        44
           restaurant_type     1.0000    1.0000    1.0000        65
                movie_type     1.0000    1.0000    1.0000        33
                      year     1.0000    1.0000    1.0000        24
           restaurant_name     0.7778    0.9333    0.8485        15
            playlist_owner     1.0000    1.0000    1.0000        70
                     track     0.6154    0.8889    0.7273         9
          spatial_relation     0.9722    0.9859    0.9790        71
    party_size_description     1.0000    1.0000    1.0000        10
            geographic_poi     1.0000    1.0000    1.0000        11
                music_item     0.9810    0.9904    0.9856       104
      object_location_type     1.0000    1.0000    1.0000        22
     condition_temperature     1.0000    1.0000    1.0000        23
             location_name     1.0000    0.9583    0.9787        24
                movie_name     0.9111    0.8723    0.8913        47
         party_size_number     1.0000    1.0000    1.0000        50
               served_dish     0.9091    0.8333    0.8696        12
               rating_unit     1.0000    1.0000    1.0000        40
                       poi     0.8750    0.8750    0.8750         8
          current_location     1.0000    1.0000    1.0000        14
                   cuisine     0.8667    0.9286    0.8966        14
                      sort     0.9688    0.9688    0.9688        32
                     genre     1.0000    1.0000    1.0000         5
                     album     0.4545    0.5000    0.4762        10
                  facility     1.0000    1.0000    1.0000         3
object_part_of_series_type     1.0000    1.0000    1.0000        11

                 micro avg     0.9638    0.9682    0.9660      1790
                 macro avg     0.9652    0.9682    0.9664      1790
```

snips - libs/JointBERT/snips_models/ep30-dev-slot_by2taskV2Wloss0.2Wdec
```log
INFO:trainer:***** Running evaluation on test dataset *****
INFO:trainer:  Num examples = 700
INFO:trainer:  Batch size = 64
INFO:trainer:***** Eval results *****
INFO:trainer:  intent_acc = 0.9828571428571429
INFO:trainer:  loss = 0.21195497931065885
INFO:trainer:  sementic_frame_acc = 0.9257142857142857
INFO:trainer:  slot_f1 = 0.9696294232376707
INFO:trainer:  slot_precision = 0.9672040022234575
INFO:trainer:  slot_recall = 0.9720670391061452
INFO:root:                            precision    recall  f1-score   support

                     state     0.9831    0.9831    0.9831        59
                      city     1.0000    1.0000    1.0000        60
                  playlist     0.9462    0.9535    0.9498       129
            geographic_poi     1.0000    1.0000    1.0000        11
               rating_unit     1.0000    1.0000    1.0000        40
          spatial_relation     0.9589    0.9859    0.9722        71
               object_name     0.8816    0.9116    0.8963       147
                      sort     0.9688    0.9688    0.9688        32
                    artist     0.9545    0.9813    0.9677       107
             object_select     1.0000    1.0000    1.0000        40
     condition_description     0.9643    0.9643    0.9643        28
              rating_value     1.0000    1.0000    1.0000        80
                   country     1.0000    0.9773    0.9885        44
                       poi     0.8889    1.0000    0.9412         8
            playlist_owner     1.0000    1.0000    1.0000        70
object_part_of_series_type     1.0000    1.0000    1.0000        11
      object_location_type     1.0000    1.0000    1.0000        22
                music_item     0.9811    1.0000    0.9905       104
                     album     0.6000    0.6000    0.6000        10
         party_size_number     1.0000    1.0000    1.0000        50
           restaurant_name     0.8750    0.9333    0.9032        15
                movie_name     0.9070    0.8298    0.8667        47
                 timeRange     0.9906    0.9813    0.9859       107
               object_type     0.9938    0.9938    0.9938       162
             location_name     1.0000    0.9583    0.9787        24
          current_location     1.0000    1.0000    1.0000        14
                     track     0.7500    1.0000    0.8571         9
     condition_temperature     1.0000    1.0000    1.0000        23
               entity_name     0.9032    0.8485    0.8750        33
                movie_type     1.0000    1.0000    1.0000        33
                   service     1.0000    1.0000    1.0000        24
               best_rating     1.0000    1.0000    1.0000        43
           restaurant_type     1.0000    1.0000    1.0000        65
                   cuisine     0.9286    0.9286    0.9286        14
               served_dish     0.9167    0.9167    0.9167        12
                  facility     1.0000    1.0000    1.0000         3
    party_size_description     1.0000    1.0000    1.0000        10
                     genre     1.0000    1.0000    1.0000         5
                      year     1.0000    1.0000    1.0000        24

                 micro avg     0.9672    0.9721    0.9696      1790
                 macro avg     0.9678    0.9721    0.9697      1790
```
