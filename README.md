# patent-kg-tf
## patent-kg

## execution
For Ubuntu environment, install the runtime environment according to the requirements; conda/venv is recommended.

Run the extract.py file directly.

Default values for argparse have already been provided.

Adjust input_filename and output_filename to your personal file locations.

You can also use:
```
python extract.py examples/example.json examples/example_result.json  --use_cuda true

```

## note
The TensorFlow version is 1.4.

Batch processing has not been added (slow).

The final filtering for adverb_list and confidence settings are not in their final versions.


