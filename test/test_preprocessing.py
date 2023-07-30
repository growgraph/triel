import logging
import sys
import unittest

from lm_service.preprocessing import normalize_input_text

logger = logging.getLogger(__name__)


class TestPreprocessing(unittest.TestCase):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    phrase = (
        "This corresponds to the transit of an Earth-sized planet orbiting a"
        " star of 0.9 R☉ in 60 days detected with a S/Ntransit >10 (100 ppm"
        " transit depth). For example, an Earth-size transit across a G star"
        " creates an 80 ppm depth. The different science objectives require"
        " 500 separate target pointings. Assuming 1 hour per pointing the"
        " mission duration is estimated at 1175 days or 3.2 years."
    )

    def test_consecutive_candidates(self):
        out = normalize_input_text(self.phrase)
        self.assertEqual(
            out,
            [
                (
                    "This corresponds to the transit of an Earth-sized planet"
                    " orbiting a star of 0.9 R in 60 days detected with a"
                    " S/Ntransit >10 (100 ppm transit depth)."
                ),
                (
                    "For example, an Earth-size transit across a G star"
                    " creates an 80 ppm depth."
                ),
                (
                    "The different science objectives require 500 separate"
                    " target pointings."
                ),
                (
                    "Assuming 1 hour per pointing the mission duration is"
                    " estimated at 1175 days or 3.2 years."
                ),
            ],
        )

    def test_dash(self):
        phrase = (
            "Launched on 18 December 2019, "
            "it is the first Small-class mission in "
            "ESA's Cosmic Vision science programme."
        )
        out = normalize_input_text(phrase)
        self.assertEqual(
            out,
            [
                "Launched on 18 December 2019, it is the first Small-class"
                " mission in ESA's Cosmic Vision science programme."
            ],
        )

    def test_newline(self):
        phrase = """All discovered tRFs from TCGA can be downloaded from https://cm.jefferson.edu/tcga-mintmap-profiles or studied interactively through the newly-designed version 2.0 of MINTbase at https://cm.jefferson.edu/MINTbase.\\n\\nNOTE: while the manuscript is under review, the content on the page https://cm.jefferson.edu/tcgamintmap-profiles is password protected and available only to Reviewers.\\n\\nKey PointsO_LIComplexity: tRNAs exhibit a complex fragmentation pattern into a multitude of tRFs that are conserved within the samples of a given cancer but differ across cancers.\\nC_LIO_LIVery extensive mitochondrial contributions: the 22 tRNAs of the mitochondrion (MT) contribute 1/3rd of all tRFs found across cancers, a disproportionately high number compared to the tRFs from the 610 nuclear tRNAs.\\nC_LIO_LIUridylated (not guanylated) 5{acute}-His tRFs: in all human tissues analyzed, tRNAHisGTG produces many abundant modified 5{acute}-tRFs with a U at their \\"-1\\" position (-1U 5{acute}-tRFs), instead of a G.\\nC_LIO_LILikely central roles for tRNAHisGTG: the relative abundances of the -1U 5{acute}-tRFs from tRNAHisGTG remain strikingly conserved across the 32 cancers, a property that makes tRNAHisGTG unique among all tRNAs and isoacceptors.\\nC_LIO_LISelective tRF-mRNA networks: tRFs are negatively correlated with mRNAs that differ characteristically from cancer to cancer.\\nC_LIO_LIMitochondrion-encoded tRFs are associated with nuclear proteins: in nearly all cancers, and in a cancer-specific manner, tRFs produced by the 22 mitochondrial tRNAs are negatively correlated with mRNAs whose protein products localize to the nucleus.\\nC_LIO_LItRFs are associated with membrane proteins: in all cancers, and in a cancer-specific manner, nucleus-encoded and MT-encoded tRFs are negatively correlated with mRNAs whose protein products localize to the cells membrane.\\nC_LIO_LItRFs are associated with secreted proteins: in all cancers, and in a cancer-specific manner, nucleusencoded and MT-encoded tRFs are negatively correlated with mRNAs whose protein products are secreted from the cell.\\nC_LIO_LItRFs are associated with numerous mRNAs through repeat elements: in all cancers, and in a cancerspecific manner, the genomic span of mRNAs that are negatively correlated with tRFs are enriched in specific categories of repeat elements.\\nC_LIO_LIintra-cancer tRF networks can depend on sex and population origin: within a cancer, positive and negative tRF-tRF correlations can be modulated by patient attributes such as sex and population origin.\\nC_LIO_LIweb-enabled exploration of an \\"Atlas for tRFs\\": we released a new version of MINTbase to provide users with the ability to study 26,531 tRFs compiled by mining 11,719 public datasets (TCGA and other sources).\\nC_LI"""

        out = normalize_input_text(phrase)
        self.assertEqual(len(out), 13)


if __name__ == "__main__":
    unittest.main()
