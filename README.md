<p align="center">

<img src="https://github.com/jina-ai/now/blob/main/docs/_static/logo-light.svg?raw=true" alt="Jina NOW logo: The data structure for unstructured data" width="300px">  


<br>
One line to host them all. Bootstrap your image search case in minutes.
</p>

<p align=center>
<a href="https://pypi.org/project/jina-now/"><img src="https://github.com/jina-ai/jina/blob/master/.github/badges/python-badge.svg?raw=true" alt="Python 3.7 3.8 3.9 3.10" title="Jina NOW supports Python 3.7 and above"></a>
<a href="https://pypi.org/project/jina-now/"><img src="https://img.shields.io/pypi/v/jina-now?color=%23099cec&amp;label=PyPI&amp;logo=pypi&amp;logoColor=white" alt="PyPI"></a>
</p>

<!-- start elevator-pitch -->

<p align="center">
<img src="https://user-images.githubusercontent.com/11627845/164569398-5ef22a41-e2e1-438a-88a5-2ac43ad9426d.gif" alt="Jina NOW logo: The data structure for unstructured data" width="600px">


NOW gives the world access to customized neural image search in just one line of code.
Main features
- 🐥 **Easy**: Minimal effort required to set up your search case
- 🐎 **Fast**: Set up your search case within minutes
- 🌈 **Quality**: If you provide labels to your documents, Jina NOW fine-tunes a model for you
- 🌳 **Reliable**: We take care of the deployment and maintenance (coming soon)
- ✨ **Nocode**: Deployment can be done by non-technical people




### Installation

```bash
pip install jina-now
```


In case you need sudo for running Docker, install and use jina-now using sudo as well.

#### Mac M1

For the Mac M1 it is generally recommended using a conda environment as outlined in the [Jina documentation](https://docs.jina.ai/get-started/install/troubleshooting/#on-mac-m1).
In a new conda environment first execute `conda install grpcio tokenizers protobuf`. Then run `pip install jina-now`.

### Usage
```bash
jina now [start | stop] --data [<pushpullid> | <localpath> | <url>] --quality [medium | good | excellent] --cluster <k8s-cluster-name>
```

### Quick Start
```bash
jina now start
```
  
### Use CLI Parameters
```bash
jina now start --quality medium --data /local/img/folder
```
  
### Use API
You can now send requests to the API using the jina client. This case shows a local deployment.
```bash
from jina import Client    
client = Client(
        host='localhost',
        protocol="grpc",
        port=31080,
) 
response = client.search(
        Document(text=search_text), # or in case you send an image: Document(url=image_path),
        parameters={"limit": 9, "filter": {}},
)
```
  
### Cleanup
```bash
jina now stop
```

### Requirements
- `Linux` or `Mac`
- `Python 3.7`, `3.8`, `3.9` or `3.10`
#### Local execution
- `Docker` installation
- 10 GB assigned to docker
- User must be permitted to run docker containers
#### Google Cloud deployment
- Billing account enabled
#### Jina Flow as a service
- No further requirements (coming soon)

## Supported Modalities (more will be added)

- [x] Text
- [x] Image
- [ ] Audio
- [ ] Video
- [ ] 3D mesh
- [ ] PDF 
- [ ] ...

[![IMAGE ALT TEXT HERE](https://user-images.githubusercontent.com/11627845/164571632-0e6a6c39-0137-413b-8287-21fc34785665.png)](https://www.youtube.com/watch?v=fdIaLP0ctpo)
</p>
<br>
  
## Examples

<details><summary>👕 Fashion</summary>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/11627845/157079335-8f36fc73-d826-4c0a-b1f3-ed5d650a1af1.png">
</details>

<details><summary>☢️ Chest X-Ray</summary>
<img src="https://user-images.githubusercontent.com/11627845/157067695-59851a77-5c43-4f68-80c4-403fec850776.png" width="400">
</details>
  
<details><summary>💰 NFT - bored apes</summary>
<img src="https://user-images.githubusercontent.com/11627845/157019002-573cc101-e23b-4020-825c-f37ec66c6ccf.jpeg" width="400">
</details>
  
<details><summary>🖼 Art</summary>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/11627845/157074453-721c0f2d-3f7d-4839-b6ff-bbccbdba2e5f.png">
</details>
  
<details><summary>🚗 Cars</summary>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/11627845/157081047-792df6bd-544d-420c-b180-df824c802e73.png">
</details>
  
<details><summary>🏞 Street view</summary>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/11627845/157087532-46ae36a2-c97f-45d7-9c3e-c624dcf6dc46.png">
</details>

<details><summary>🦆 Birds</summary>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/11627845/157069954-615a5cb6-dda0-4a2f-9442-ea807ad4a8d5.png">
</details>


### Now use your custom data :)
<!-- end elevator-pitch -->
