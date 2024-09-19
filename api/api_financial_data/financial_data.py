import json
from eodhd import APIClient

# https://eodhd.medium.com/how-to-build-a-stocks-market-web-app-with-python-af46ce177562

class EODHDAPIsDataFetcher:
    def __init__(self, api_token):
        self._api_token = api_token

    def fetch_exchanges(self):
        try:
            api = APIClient(self._api_token)
            df = api.get_exchanges()
            return json.loads(df.to_json(orient="records"))

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

