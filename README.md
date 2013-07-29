sundry
======

Miscellaneous scripts and bits and pieces that don't fit anywhere else


# disfl/clean_swbd.py #

This script pre-processes the Switchboard corpus. It:

* removes disfluent text from the trees. Currently *both* words under EDITED nodes *and* words in the 'reparandum' portion of
the \[ reparandum + repair \] .dff/.dps annotation format. Note that this is a departure from Mark's work, where
only the reparandum text is removed. A switch may need to be built in to turn off the EDITED node cleaning, for consistency
with previous work.

* removes punctuation
* removes trace nodes
* removes fillers (um, uh etc) --- also not done by Mark's previous work, but necessary for LTH converter to perform well.
* removes disfluency annotations
* removes resulting empty categories
* removes resulting empty sentences

The script writes the resulting .mrg files to the output directory, and assumes it will find SWBD on /usr/local/data/Penn3.
It also writes out .dps files with updated reparanda, including words under EDITED nodes, for use with the repairs
pipeline. These .dps files have not yet been tested.
