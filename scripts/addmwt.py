"""
Block AddMwt for detection of segmented words (morphological trees).
"""

import udapi.block.ud.addmwt

MWTS = {}

mwt_root = 0 # empty variable that will be overwritten each time

# shared values for all entries in MWTS
for v in MWTS.values():
    v['feats'] = '_'
    v['main'] = mwt_root  # which of the two words will inherit the original children (if any)


class AddMwt(udapi.block.ud.addmwt.AddMwt):
    """Detect and mark MWTs (split them into words and add the words to the tree)."""

    def multiword_analysis(self, node):
        """Return a dict with MWT info or None if `node` does not represent a multiword token."""
        analysis = MWTS.get(node.form.lower(), None)
        if analysis is not None:
            return analysis
