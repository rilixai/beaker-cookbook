# Attribution

BANKING77 is from PolyAI's task-specific datasets repository:
https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data

Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić.
2020. *Efficient Intent Detection with Dual Sentence Encoders*.
https://arxiv.org/abs/2003.04807

The upstream data is licensed under CC BY 4.0; see [DATA_LICENSE](DATA_LICENSE).
The 77 category IDs are copied unchanged into `src/jev_banking77/categories.json`.
The initial editable descriptions replace underscores in those IDs with spaces.
One real training message is reproduced in the offline client tests.
The preparation command downloads the original CSV files and converts them to
JSONL. It creates a validation split from the original training data and retains
the complete official test split separately. It does not change the labels.

This recipe's harness and Beaker integration are original code. It does not copy
JevBench code or claim to reproduce its published scores. JevBench is a useful
comparison: https://github.com/dhruvmehra/jevbench.
