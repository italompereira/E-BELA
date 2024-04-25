from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import Hop
from typing import List, Set
import attr
import logging

logging.basicConfig(level=logging.INFO, filename="program.log", format="%(asctime)s - %(levelname)s - %(message)s")


@attr.s
class KG(KG):

    include_predicates = attr.ib(
        factory=set,
        type=Set[str],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(str)
        ),
    )

    def get_from_kg(self, query):
        while True:
            try:
                return self.connector.fetch(query)
                break
            except Exception as e:
                print(str(e))
                logging.info(f"Message: {e}")

    def _res2hops(self, vertex: Vertex, res) -> List[Hop]:

        hops = []
        for value in res:
            obj = Vertex(value["o"]["value"])
            pred = Vertex(
                value["p"]["value"],
                predicate=True,
                vprev=vertex,
                vnext=obj,
            )
            # if pred.name not in self.skip_predicates:
            #     hops.append((pred, obj))

            if pred.name in self.include_predicates:
                hops.append((pred, obj))

        return hops