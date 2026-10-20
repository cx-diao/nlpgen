# Sample NLPGens runs

Each directory here is a sample of one NLPGens run: the annotations of one paper
(the sample paper in [`../data`](../data)) taken from the full run for that venue.
The files are exactly as in the full dataset. See the
[NLPGens section of the main README](../README.md#nlpgens) for the layout, the
column reference and how to load the data.

```bash
python -m nlpgens summary output/*/
```

The annotations are licensed under CC BY-NC-SA 4.0, like the rest of NLPGens.
They contain sentences from NLPeer papers; see [`../data/README.md`](../data/README.md)
for the source licences.
