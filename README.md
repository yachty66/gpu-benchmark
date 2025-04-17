# gpu-benchmark

## what

This is GPU benchmark tool

1. Performance measurement - How fast can the GPU complete specific tasks?
2. Health/stability testing - Can the GPU operate reliably under sustained load?

Things which are important for the project:

1. needs to be CLI based only because GPU users often dont have a monitor available for their GPU
2. super easy to run - literally just pip install and then running the package
3. Frontend with leaderboard to compare results in the long run

## current most popular benchmarks

- futuremark/3dmark
- furemark
- FurMark for stress test
- 3dMark for rendering
- GPU-Z to see if it's legit
- MSI Afterburner – Overclock, benchmark, monitor tool
- Unigine Heaven – GPU Benchmark/stress test
- Unigine Superposition – GPU Benchmark/stress test
- Blender – Rendering benchmark
- 3DMark Time Spy - DirectX 12 benchmark
- 3DMark Fire Strike - DirectX 11 benchmark
- Furmark – OpenGL benchmark/stress test
- Passmark – Comprehensive benchmark
- PCMark– Comprehensive benchmark
- Novabench – Comprehensive benchmark
- SiSoft Sandra – Comprehensive benchmark
- gpu-z

## notes

now i need to either make a package which is a wrapper around all of this exisintg benchmarks and runs them or i make a separate package which runs different llms and image gen models