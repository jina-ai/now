When there are major changes in the test cases (execution time) compared to what's stored
in the `.test_durations` file, it's recommended to update the duration information with `--store-durations`
to ensure that the splitting is in balance:

```
pytest --store-durations
```
