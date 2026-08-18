# Observed-model evidence sweep — raw output

Re-run of the evidence table in `docs/OPEN_VULKAN_OBSERVED_MODEL.md`, to confirm
or refute the fix in `6c1589a` ("every observed node summed the WHOLE obs
buffer") across more than the single seed / single model it was first measured on.

Script: `bench/observed_model_evidence.exs`, at commit 6c1589a (+ the uncommitted
gate1/reconcile-core tree; nothing in the sampler or synth path is modified by it).

    COMPILER=none   SEEDS=42,1,2,3 mix run --no-deps-check bench/observed_model_evidence.exs
    COMPILER=vulkan SEEDS=42,1,2,3 mix run --no-deps-check bench/observed_model_evidence.exs

Host: super-io — Intel Xeon E5-2699 v4 (88 threads), NVIDIA GeForce RTX 3060 Ti
(DiscreteGpu), Linux 6.8.0-137-generic x86_64, Erlang/OTP 27 (erts 15.2.7.2),
Elixir 1.18.3. Nx backend `Nx.BinaryBackend` in both arms; the `vulkan` arm
differs only in that the leapfrog chain runs on the GPU.
Date: 2026-08-17.

Every row is measured against the closed-form conjugate posterior, not against
the other arm. The `gpu` column counts `Exmc.NUTS.Vulkan.Dispatch.chain/8` calls
made while that row sampled — it is the vacuity guard: a `vulkan` row reading 0
there fell back to the host path and proves nothing.

Result: **72 rows (9 variants x 4 seeds x 2 arms), all within tolerance of the
closed form.** Worst mean error 0.091, worst sd error 8.1%, fewest distinct
draws 447/500. No frozen chain on any variant, on either arm.

---

## COMPILER=none

```
=== observed-model evidence — docs/OPEN_VULKAN_OBSERVED_MODEL.md ===
compiler       : none
warmup/samples : 300/500
seeds          : 42,1,2,3 (each reported separately)
backend        : {Nx.BinaryBackend, []}
variant                     seed  mean      truth     sd        truth     distinct  gpu       eps         verdict
-----------------------------------------------------------------------------------------------------------------
scalar 3 obs                42    3.9696    3.9867    0.5530    0.5764    472/500   0         1.083809    ok
scalar 3 obs                1     4.0043    3.9867    0.5525    0.5764    467/500   0         1.197251    ok
scalar 3 obs                2     3.9648    3.9867    0.5949    0.5764    465/500   0         1.272550    ok
scalar 3 obs                3     3.9465    3.9867    0.5879    0.5764    459/500   0         1.016601    ok
vector 3 obs                42    4.0419    3.9867    0.5312    0.5764    484/500   0         0.977807    ok
vector 3 obs                1     4.0043    3.9867    0.5525    0.5764    467/500   0         1.197251    ok
vector 3 obs                2     3.9648    3.9867    0.5949    0.5764    465/500   0         1.272550    ok
vector 3 obs                3     3.9465    3.9867    0.5879    0.5764    459/500   0         1.016601    ok
scalar 3 obs, sigmas 1/2/3  42    3.9991    3.9506    0.7856    0.8540    481/500   0         1.127945    ok
scalar 3 obs, sigmas 1/2/3  1     3.9408    3.9506    0.8274    0.8540    467/500   0         1.051368    ok
scalar 3 obs, sigmas 1/2/3  2     3.9180    3.9506    0.8814    0.8540    465/500   0         1.265778    ok
scalar 3 obs, sigmas 1/2/3  3     3.9007    3.9506    0.8880    0.8540    469/500   0         0.939060    ok
scalar 1 obs                42    3.9028    3.9604    0.9690    0.9950    474/500   0         1.192544    ok
scalar 1 obs                1     3.9907    3.9604    0.9537    0.9950    467/500   0         1.192196    ok
scalar 1 obs                2     3.8788    3.9604    0.9691    0.9950    491/500   0         0.928519    ok
scalar 1 obs                3     3.9064    3.9604    1.0367    0.9950    469/500   0         1.070138    ok
scalar 2 obs                42    4.4675    4.4776    0.6847    0.7053    478/500   0         1.357832    ok
scalar 2 obs                1     4.4917    4.4776    0.6740    0.7053    467/500   0         1.016123    ok
scalar 2 obs                2     4.4525    4.4776    0.7284    0.7053    465/500   0         1.262383    ok
scalar 2 obs                3     4.4396    4.4776    0.7465    0.7053    461/500   0         1.302051    ok
scalar 5 obs                42    2.9526    2.9762    0.4348    0.4454    476/500   0         1.307798    ok
scalar 5 obs                1     2.9898    2.9762    0.4269    0.4454    467/500   0         1.227873    ok
scalar 5 obs                2     2.9804    2.9762    0.4665    0.4454    471/500   0         1.402421    ok
scalar 5 obs                3     2.9539    2.9762    0.4642    0.4454    475/500   0         0.903267    ok
vector 5 obs                42    2.9565    2.9762    0.4359    0.4454    475/500   0         1.310263    ok
vector 5 obs                1     2.9898    2.9762    0.4269    0.4454    467/500   0         1.227873    ok
vector 5 obs                2     2.9804    2.9762    0.4665    0.4454    471/500   0         1.402421    ok
vector 5 obs                3     2.9539    2.9762    0.4642    0.4454    475/500   0         0.903273    ok
scalar 5 obs, sigmas 1..5   42    1.4945    1.5186    0.7680    0.8155    467/500   0         1.351082    ok
scalar 5 obs, sigmas 1..5   1     1.5349    1.5186    0.7793    0.8155    467/500   0         1.123608    ok
scalar 5 obs, sigmas 1..5   2     1.5305    1.5186    0.8539    0.8155    471/500   0         1.357870    ok
scalar 5 obs, sigmas 1..5   3     1.4787    1.5186    0.8458    0.8155    469/500   0         1.036038    ok
scalar 4 obs, sigmas .5/1/2/442    3.9588    3.9584    0.4182    0.4335    480/500   0         1.113172    ok
scalar 4 obs, sigmas .5/1/2/41     3.9751    3.9584    0.4078    0.4335    477/500   0         0.929142    ok
scalar 4 obs, sigmas .5/1/2/42     3.9138    3.9584    0.4366    0.4335    481/500   0         1.074872    ok
scalar 4 obs, sigmas .5/1/2/43     3.9257    3.9584    0.4519    0.4335    472/500   0         1.001788    ok
36 rows, all within tolerance of the closed form
```

## COMPILER=vulkan

```
=== observed-model evidence — docs/OPEN_VULKAN_OBSERVED_MODEL.md ===
compiler       : vulkan
warmup/samples : 300/500
seeds          : 42,1,2,3 (each reported separately)
backend        : {Nx.BinaryBackend, []}
variant                     seed  mean      truth     sd        truth     distinct  gpu       eps         verdict
-----------------------------------------------------------------------------------------------------------------
[nx_vulkan_vulkano] device: NVIDIA GeForce RTX 3060 Ti (DiscreteGpu)
scalar 3 obs                42    3.9864    3.9867    0.5542    0.5764    479/500   692       1.139122    ok
scalar 3 obs                1     3.9732    3.9867    0.5337    0.5764    466/500   651       1.171770    ok
scalar 3 obs                2     3.9744    3.9867    0.5999    0.5764    470/500   657       1.290524    ok
scalar 3 obs                3     3.9900    3.9867    0.6015    0.5764    483/500   687       0.955862    ok
vector 3 obs                42    3.9716    3.9867    0.5522    0.5764    472/500   678       1.064154    ok
vector 3 obs                1     3.9732    3.9867    0.5337    0.5764    466/500   651       1.171770    ok
vector 3 obs                2     3.9744    3.9867    0.5999    0.5764    470/500   657       1.290524    ok
vector 3 obs                3     3.9900    3.9867    0.6015    0.5764    483/500   687       0.955862    ok
scalar 3 obs, sigmas 1/2/3  42    3.9200    3.9506    0.8107    0.8540    470/500   681       1.192643    ok
scalar 3 obs, sigmas 1/2/3  1     3.9495    3.9506    0.7847    0.8540    462/500   649       1.170199    ok
scalar 3 obs, sigmas 1/2/3  2     3.9708    3.9506    0.8631    0.8540    474/500   659       1.308949    ok
scalar 3 obs, sigmas 1/2/3  3     3.9055    3.9506    0.8877    0.8540    469/500   661       0.909354    ok
scalar 1 obs                42    3.9407    3.9604    0.9234    0.9950    467/500   671       1.328150    ok
scalar 1 obs                1     4.0084    3.9604    0.9602    0.9950    447/500   629       1.740794    ok
scalar 1 obs                2     3.8836    3.9604    1.0344    0.9950    462/500   647       1.283399    ok
scalar 1 obs                3     3.9085    3.9604    1.0113    0.9950    465/500   661       1.094599    ok
scalar 2 obs                42    4.4662    4.4776    0.6697    0.7053    471/500   674       1.148538    ok
scalar 2 obs                1     4.4468    4.4776    0.6648    0.7053    461/500   643       1.331390    ok
scalar 2 obs                2     4.4803    4.4776    0.7326    0.7053    472/500   661       1.186112    ok
scalar 2 obs                3     4.4154    4.4776    0.7498    0.7053    458/500   643       1.084425    ok
scalar 5 obs                42    2.9720    2.9762    0.4328    0.4454    479/500   689       1.126011    ok
scalar 5 obs                1     2.9598    2.9762    0.4278    0.4454    460/500   642       1.362714    ok
scalar 5 obs                2     2.9467    2.9762    0.4310    0.4454    486/500   684       1.270513    ok
scalar 5 obs                3     2.9817    2.9762    0.4760    0.4454    479/500   676       0.930254    ok
vector 5 obs                42    2.9720    2.9762    0.4328    0.4454    479/500   689       1.126011    ok
vector 5 obs                1     2.9598    2.9762    0.4278    0.4454    460/500   642       1.362712    ok
vector 5 obs                2     2.9467    2.9762    0.4310    0.4454    486/500   684       1.270513    ok
vector 5 obs                3     2.9817    2.9762    0.4760    0.4454    479/500   676       0.930254    ok
scalar 5 obs, sigmas 1..5   42    1.4954    1.5186    0.7916    0.8155    479/500   688       1.082300    ok
scalar 5 obs, sigmas 1..5   1     1.5172    1.5186    0.7574    0.8155    466/500   650       1.159494    ok
scalar 5 obs, sigmas 1..5   2     1.4445    1.5186    0.8475    0.8155    463/500   645       1.565374    ok
scalar 5 obs, sigmas 1..5   3     1.4273    1.5186    0.8623    0.8155    461/500   648       1.353551    ok
scalar 4 obs, sigmas .5/1/2/442    3.9358    3.9584    0.4224    0.4335    474/500   674       1.238148    ok
scalar 4 obs, sigmas .5/1/2/41     3.9397    3.9584    0.4079    0.4335    460/500   643       1.338926    ok
scalar 4 obs, sigmas .5/1/2/42     3.9250    3.9584    0.4506    0.4335    462/500   647       1.296058    ok
scalar 4 obs, sigmas .5/1/2/43     3.9639    3.9584    0.4632    0.4335    479/500   676       0.968255    ok
36 rows, all within tolerance of the closed form
```
