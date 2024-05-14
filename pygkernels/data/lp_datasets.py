import os

from pygkernels.data._dataset import Datasets

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


class LPDatasets(Datasets):
    def __init__(self, datasets_root=None):
        paths = {
            "Karate": "konect/ucidata-zachary.gml",
            "Bombing": "konect/moreno_train.gml",
            "Iceland": "konect/iceland.gml",
            "LesMis": "konect/moreno_lesmis.gml",
            "Neurons": "konect/dimacs10-celegansneural.gml",
            # "E.coli": "",
            "Netscience": "konect/dimacs10-netscience.gml",
            "Infectious": "konect/sociopatterns-infectious.gml",
            "Metabolic": "konect/dimacs10-celegans_metabolic.gml",
            "US Air": "konect/opsahl-usairport.gml",
            "ASoIaF": "konect/asoiaf.gml",
            "Email": "konect/arenas-email.gml",
            "Bible": "konect/moreno_names.gml",
            "Yeast": "konect/moreno_propro.gml",
            "Vidal": "konect/maayan-vidal.gml",
            "Household": "konect/petster-hamster-household.gml",
            "Friendships": "konect/petster-friendships-hamster.gml",
            "Hamster": "konect/petster-hamster.gml",
        }
        super().__init__(paths, datasets_root=datasets_root)


if __name__ == "__main__":
    datasets = LPDatasets()
    G = datasets["ASoIaF"]
    print(G.edges)
