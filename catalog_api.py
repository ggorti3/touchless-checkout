import requests
import base64
import http.client


def login(username, password):
    url = "https://gateway-staging.ncrcloud.com/security/authentication/login"
    decoded = f'{username}:{password}'
    encoded = base64.b64encode(decoded.encode()).decode("ISO-8859-1")

    headers = {
        'Authorization': f'Basic {encoded}'
    }

    response = requests.request("POST", url, headers=headers, data={})
    token = response.json()['token']

    return f'AccessToken {token}'

# conn = http.client.HTTPSConnection("api.ncr.com")

# headers = {
#     'nep-enterprise-unit': "320863da017645489c53ee8a7d02233f",
#     'nep-correlation-id': "10-23-2021-21-10-35",
#     'nep-organization': "test-drive-0c2e197c210f4ef2bf181",
#     }

# conn.request("GET", "/catalog/v2/item-details?pageNumber=0&pageSize=200&filterDate=2022-08-17T05%3A18%3A25.695Z", headers=headers)

# res = conn.getresponse()
# data = res.read()

# print(data.decode("utf-8"))