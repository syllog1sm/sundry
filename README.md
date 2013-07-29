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

# LTH Conversion Scripts #

I haven't put tonnes of thought into these, but here are four LTH conversion configurations I'm curious about.

* lth_prague_coord.sh

Set coordination structure to "prague". I don't even know what this structure means atm. This has the
"base" settings --- the other schemes are described as diffs from this one. (That's not a particularly well motivated
decision.)

* lth_oldlth_coord.sh

Set coordination structure to "oldLTH". I haven't looked at this structure either. NB: "melchuk" seems to produce lots of
non-projective trees, so the third coordination setting is apparently out of contention. I don't know why it does that.

* lth_fancy_np.sh

This sets the various NP tweak flags to "on", e.g. it distinguishes titles, names, suffixes, apposition, etc. I figure
these flags won't make much difference alone, but may be worth looking at as a batch, and it should be nicely
orthogonal to the other things we're looking at.

* lth_stanprep.sh

This sets the prepositional phrase and subordinating conjunction headedness to be more like the structures that
we find puzzling in Stanford. The hypothesis here is that these structures will perform poorly, suggesting we
could get better Stanford results by reversing these decisions.

* lth_roots.sh

This enables the setting rootLabels, which "use separate root labels such as ROOT-S, ROOT-FRAG".
