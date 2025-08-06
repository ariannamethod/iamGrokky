# Datasets

This directory holds data used for model fine tuning.

## Format

The example dataset `pulp_fiction_dialogue.csv` contains dialogue from the
*Pulp Fiction* screenplay. It is a comma separated file with the following
columns:

1. `Line number`
2. `Character (in script)`
3. `Character (actual)`
4. `Off screen`
5. `Voice-over`
6. `Place`
7. `Time`
8. `Line` – the dialogue text
9. `Word count`

Only the `Line` column is needed for language modeling.

## Preprocessing

Use `preprocess.py` to convert the CSV into a plain-text corpus suitable for
training:

```bash
python preprocess.py --input pulp_fiction_dialogue.csv --output pulp_fiction.txt
```

The script extracts the `Line` column and writes one line of text per row to
the specified output file.
