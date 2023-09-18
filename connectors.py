from pyrdf2vec.connectors import Connector
import attr
from cachetools import Cache, TTLCache, cachedmethod
from cachetools.keys import hashkey
import requests
import operator


@attr.s
class Connector(Connector):
    
    @cachedmethod(operator.attrgetter("cache"), key=partial(hashkey, "fetch"))
    def fetch(self, query: str) -> Response:
        """Fetchs the result of a SPARQL query.

        Args:
            query: The query to fetch the result.

        Returns:
            The response of the query in a JSON format.

        """
        url = f"{self.endpoint}/query?query={parse.quote(query)}"
        url = f"{self.endpoint}?default-graph-uri=http%3A%2F%2Fdbpedia.org&query={parse.quote(query)}"
        with requests.get(url, headers=self._headers) as res:
            return res.json()