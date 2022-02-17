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
INFO:trainer:***** Running evaluation on dev dataset *****
INFO:trainer:  Num examples = 700
INFO:trainer:  Batch size = 64
INFO:trainer:***** Eval results *****
INFO:trainer:  intent_acc = 0.99
INFO:trainer:  loss = 0.3223934769630432
INFO:trainer:  sementic_frame_acc = 0.9085714285714286
INFO:trainer:  slot_f1 = 0.9592910551093878
INFO:trainer:  slot_precision = 0.9532195927352779
INFO:trainer:  slot_recall = 0.9654403567447045
INFO:root:                            precision    recall  f1-score   support

                 timeRange     0.9821    1.0000    0.9910       110
                      year     1.0000    1.0000    1.0000        25
              rating_value     0.9901    1.0000    0.9950       100
           restaurant_type     0.9375    0.9677    0.9524        62
               object_name     0.9091    0.9272    0.9180       151
                    artist     0.9722    0.9633    0.9677       109
            playlist_owner     1.0000    1.0000    1.0000        54
               object_type     0.9936    0.9936    0.9936       156
                   country     1.0000    0.9773    0.9885        44
                movie_type     1.0000    1.0000    1.0000        24
             object_select     0.9592    0.9592    0.9592        49
                music_item     0.9663    1.0000    0.9829        86
         party_size_number     0.9828    1.0000    0.9913        57
                      city     0.9853    0.9437    0.9640        71
                movie_name     0.9535    0.8367    0.8913        49
               best_rating     1.0000    1.0000    1.0000        51
            geographic_poi     1.0000    1.0000    1.0000        16
object_part_of_series_type     1.0000    1.0000    1.0000        15
               rating_unit     1.0000    1.0000    1.0000        61
     condition_description     0.9565    1.0000    0.9778        22
                   service     1.0000    1.0000    1.0000        39
                  playlist     0.9279    0.9450    0.9364       109
    party_size_description     1.0000    1.0000    1.0000        13
          current_location     1.0000    1.0000    1.0000        17
          spatial_relation     0.9853    0.9853    0.9853        68
                   cuisine     1.0000    0.9091    0.9524        11
             location_name     1.0000    1.0000    1.0000        29
                     state     0.9623    1.0000    0.9808        51
                     album     0.2727    0.2308    0.2500        13
               entity_name     0.6190    0.7222    0.6667        18
                  facility     1.0000    1.0000    1.0000         7
           restaurant_name     0.8261    0.9500    0.8837        20
                       poi     0.4444    0.6667    0.5333         6
                      sort     1.0000    1.0000    1.0000        26
     condition_temperature     1.0000    1.0000    1.0000        21
               served_dish     0.7143    1.0000    0.8333         5
      object_location_type     1.0000    1.0000    1.0000        20
                     track     0.3077    0.6667    0.4211         6
                     genre     0.4000    0.6667    0.5000         3

                 micro avg     0.9532    0.9654    0.9593      1794
                 macro avg     0.9576    0.9654    0.9608      1794
```

snips - libs/JointBERT/snips_models/ep30-dev-slot_by2taskV2Wdec
```log
INFO:trainer:***** Model Loaded *****
INFO:trainer:***** Running evaluation on dev dataset *****
INFO:trainer:  Num examples = 700
INFO:trainer:  Batch size = 64
INFO:trainer:***** Eval results *****
INFO:trainer:  intent_acc = 0.9914285714285714
INFO:trainer:  loss = 0.3689329827373678
INFO:trainer:  sementic_frame_acc = 0.9142857142857143
INFO:trainer:  slot_f1 = 0.9641367806505421
INFO:trainer:  slot_precision = 0.961730449251248
INFO:trainer:  slot_recall = 0.9665551839464883
INFO:root:                            precision    recall  f1-score   support

                     state     0.9273    1.0000    0.9623        51
                      year     1.0000    1.0000    1.0000        25
               object_type     0.9936    0.9936    0.9936       156
     condition_description     1.0000    1.0000    1.0000        22
                movie_name     0.9333    0.8571    0.8936        49
                      city     0.9710    0.9437    0.9571        71
                 timeRange     0.9821    1.0000    0.9910       110
               object_name     0.8790    0.9139    0.8961       151
                   cuisine     1.0000    0.9091    0.9524        11
                    artist     0.9722    0.9633    0.9677       109
           restaurant_type     0.9365    0.9516    0.9440        62
               rating_unit     1.0000    1.0000    1.0000        61
           restaurant_name     0.9048    0.9500    0.9268        20
         party_size_number     1.0000    1.0000    1.0000        57
              rating_value     1.0000    1.0000    1.0000       100
            playlist_owner     1.0000    1.0000    1.0000        54
          current_location     1.0000    1.0000    1.0000        17
          spatial_relation     0.9565    0.9706    0.9635        68
                music_item     0.9770    0.9884    0.9827        86
                  playlist     0.9720    0.9541    0.9630       109
      object_location_type     1.0000    1.0000    1.0000        20
               best_rating     1.0000    1.0000    1.0000        51
             object_select     0.9796    0.9796    0.9796        49
object_part_of_series_type     1.0000    1.0000    1.0000        15
     condition_temperature     1.0000    1.0000    1.0000        21
                   service     0.9750    1.0000    0.9873        39
                      sort     1.0000    1.0000    1.0000        26
    party_size_description     1.0000    1.0000    1.0000        13
                     album     0.6250    0.3846    0.4762        13
             location_name     1.0000    1.0000    1.0000        29
                       poi     0.7500    0.5000    0.6000         6
                     genre     0.6000    1.0000    0.7500         3
               served_dish     0.7143    1.0000    0.8333         5
               entity_name     0.8333    0.8333    0.8333        18
                     track     0.4545    0.8333    0.5882         6
                   country     0.9773    0.9773    0.9773        44
            geographic_poi     0.9375    0.9375    0.9375        16
                movie_type     1.0000    1.0000    1.0000        24
                  facility     1.0000    1.0000    1.0000         7

                 micro avg     0.9617    0.9666    0.9641      1794
                 macro avg     0.9630    0.9666    0.9640      1794
```
