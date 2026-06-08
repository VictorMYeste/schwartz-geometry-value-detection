# Selected Qualitative Decoder Examples

Selected from cases where the Schwartz decoder changed the final label set and improved sample-level F1.

| rank | selection_reason | excerpt | gold_labels | standard_labels | decoded_labels | added_labels | removed_labels | sample_f1_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | adds_two_nearby_true_values | [removed: dataset licence] | Security: societal \| Conformity: rules | Security: personal | Security: personal \| Security: societal \| Conformity: rules | Security: societal \| Conformity: rules |  | 0.8 |
| 2 | adds_true_value_and_removes_opposite_false_positive | [removed: dataset licence] | Conformity: interpersonal | Self-direction: action \| Conformity: rules | Conformity: rules \| Conformity: interpersonal | Conformity: interpersonal | Self-direction: action | 0.6666666666666666 |
| 3 | removes_two_opposite_false_positives | [removed: dataset licence] | Power: dominance | Power: dominance \| Benevolence: dependability \| Universalism: concern | Power: dominance |  | Benevolence: dependability \| Universalism: concern | 0.5 |
| 4 | removes_opposite_false_positive | [removed: dataset licence] | Universalism: nature | Security: societal \| Universalism: nature | Universalism: nature |  | Security: societal | 0.3333333333333333 |
| 5 | removes_opposite_false_positive | [removed: dataset licence] | Achievement | Achievement \| Humility | Achievement |  | Humility | 0.3333333333333333 |
| 6 | adds_nearby_true_value | [removed: dataset licence] | Hedonism | Stimulation | Stimulation \| Hedonism | Hedonism |  | 0.6666666666666666 |
| 7 | adds_nearby_true_value | [removed: dataset licence] | Self-direction: thought | Self-direction: action | Self-direction: thought \| Self-direction: action | Self-direction: thought |  | 0.6666666666666666 |
| 8 | adds_nearby_true_value | [removed: dataset licence] | Universalism: concern | Benevolence: caring | Benevolence: caring \| Universalism: concern | Universalism: concern |  | 0.6666666666666666 |
| 9 | adds_nearby_true_value | [removed: dataset licence] | Power: resources | Power: dominance | Power: dominance \| Power: resources | Power: resources |  | 0.6666666666666666 |
| 10 | adds_nearby_true_value | [removed: dataset licence] | Security: personal | Security: societal | Security: personal \| Security: societal | Security: personal |  | 0.6666666666666666 |
