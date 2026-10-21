# netpattern (Python SDK)

A small client for calling NetPattern deployments. Standard library only.

```bash
pip install ./sdk/python
export NETPATTERN_API_KEY=np_…            # an inference key from the "API keys" page
```

```python
from netpattern import Client

client = Client(base_url="http://localhost:8000")
print(client.schema("dep_…")["classes"])
print(client.predict("dep_…", ["image.png"], top_k=3)["outputs"])
```

Errors raise `netpattern.ApiError` with `status`, `code` and `detail`.
