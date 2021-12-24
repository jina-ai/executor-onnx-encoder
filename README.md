# ONNXEncoder

An executor that loads ONNX models and embeds documents using the ONNX runtime.


## Usage

#### via Docker image (recommended)

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://ONNXEncoder')
```

#### via source code

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://ONNXEncoder')
```

- To override `__init__` args & kwargs, use `.add(..., uses_with: {'key': 'value'})`
- To override class metas, use `.add(..., uses_metas: {'key': 'value})`
