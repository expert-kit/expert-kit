# Torch ExpertMesh Doc

ExpertMesh Integration for deepseek torch version.

## Installation

#TODO

## Usage

### For Demo Testing of mini-deekseek-v3

First, you should generate a mini deepseekv3 model. Use the script `save_mini_model.sh` in `examples`, where the config.json file controls the model size.
```
./examples/save_mini_model.sh
```

Second, use the script `run_local.sh` in directory `examples` to test the saved model. The input text can be writen in file `local_test.txt`.

## Requirements

```
pip install -i requirements.txt
```

## Deployment Example

To test the Demo, deploy mock server first:
```bash
./mock_server --expert-dim [dim]
```
Then, run the bash script:
```bash
./examples/run_local.sh
```

## References

- [DeepSeek-V3 Documentation](https://github.com/deepseek-ai/DeepSeek-V3)
- [gRPC Documentation](https://grpc.io/docs/languages/python/)
- [DeepSeek-R1 Documentation](https://github.com/deepseek-ai/DeepSeek-R1)