# Expert Kit

**Expert Kit** is a framework for MoE (Mixture of Experts) model inference. It is designed to be a scalable and easy-to-use framework for deploying EP(Expert Parallelism) clusters to serve MoE inference.

- [ek-computation](./ek-agent): runs in hardwares and performs computation task.
- [ek-exchange](./ek-exchange): experimental high-performance communication of expert.
- [ek-edb](./ek-edb): supports registering and loading experts' weight in fine-grained granularity.
- [ek-apm](./ek-exchange): is APM(Application Performance Monitoring) tools for collect performance data in EP cluster.
- [ek-benchmark](./ek-benchmark): contains several micro-benchmarks help you know the performance detail of EP cluster.
- [ek-solution](./ek-solution): contains several recipes to quickly setup a running EP cluster for MoE inference.
