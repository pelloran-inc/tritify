# Flagship run comparison report — GLM-4.5-Air across three machines

This is the record of one run. GLM-4.5-Air generated its answer across three machines, none of which holds
the model in RAM, and the run was compared with a single-host reference layer by layer. Every hash below is
the SHA-256 of integers the engine computed.

It is a comparison report published by Pelloran, with matching hashes. Independent verification requires
recomputation or additional accessible artifacts.

This file shows the result, not how the machines were coordinated:
- The orchestration is Pelloran's separate, closed product.
- Pelloran keeps the run's full telemetry.
- The run cannot be re-run from this repository ([REPRODUCE.md](REPRODUCE.md)). What a reader can check here is
  that at every layer the distributed run's hash equals the single-host reference's.

The same data, machine-readable: [`flagship_run.json`](flagship_run.json).

## The model

| | |
|---|---|
| Model | GLM-4.5-Air |
| Parameters | 106.9B |
| Layers | 46 |
| Experts per MoE layer | 128 routed (top 8) and 1 shared |
| The engine's integer codes | 427.4 GB |

## The machines

| Name | Role | Device | Architecture | RAM | Holds | Routed experts held |
|---|---|---|---|---|---|---|
| M3 | coordinator | Apple M3 Max | aarch64 | 68.7 GB | 28.8 GB | — |
| EPYC | worker | AMD EPYC 9454P | x86_64 | 337.5 GB | 233.6 GB | 0–74 |
| host B | worker | Intel Xeon Platinum 8358 | x86_64 | 238.5 GB | 165.1 GB | 75–127 |

The largest machine has 337.5 GB of RAM; the model's codes are 427.4 GB.

## The result

- **The answer:** "The capital of France is Paris." (tokens `[10017, 6722, 315, 9621, 374, 12089, 1189]`).
- **The generation's hash:** distributed `564f495cea735728c220a3c2a6bd64ae4937fb105e2625fb155970876b47f741`, single-host reference `564f495cea735728c220a3c2a6bd64ae4937fb105e2625fb155970876b47f741`. Equal.
- **The output's hash:** distributed `67bd34b30d2fae8198897a04449d14cf5b1b2dda8bfa78e3efe2524cb9f2b7bc`, single host `67bd34b30d2fae8198897a04449d14cf5b1b2dda8bfa78e3efe2524cb9f2b7bc`. Equal.
- **Over the whole run:** 2,504 round trips and 10,080 expert fetches.

## The chain: every layer, distributed against single host

For each layer, the hash of the residual stream after it, in the distributed run and in the single-host
reference. All 46 are equal.

| Layer | Distributed | Single-host reference | Equal |
|---|---|---|---|
| 0 | `48eb94850ed5840f2b2f0edd9c242de70f0f403325121f0e8a2182ebed4af5a6` | `48eb94850ed5840f2b2f0edd9c242de70f0f403325121f0e8a2182ebed4af5a6` | yes |
| 1 | `12b4bda185113ca4373290a7b0a406cbb9e6500da8afa4f84d82f01d88863a19` | `12b4bda185113ca4373290a7b0a406cbb9e6500da8afa4f84d82f01d88863a19` | yes |
| 2 | `4826cf52c55a05a4f98fda475c1bf3743c5664f945f590e0436df35d62761db6` | `4826cf52c55a05a4f98fda475c1bf3743c5664f945f590e0436df35d62761db6` | yes |
| 3 | `c94b434092efa659fff8b6f43cd8510358a2ac6aabc63cae2f603156f305c948` | `c94b434092efa659fff8b6f43cd8510358a2ac6aabc63cae2f603156f305c948` | yes |
| 4 | `cfbde97611519d297270a22d3d0918eae6ffbb1cc8246ddb691f86c7bf319586` | `cfbde97611519d297270a22d3d0918eae6ffbb1cc8246ddb691f86c7bf319586` | yes |
| 5 | `77430b6bfb33f53e4b608060ab9b733774d5d767da978309c4097571c9a51d7d` | `77430b6bfb33f53e4b608060ab9b733774d5d767da978309c4097571c9a51d7d` | yes |
| 6 | `73eaa6d2e2320466f7a7ba2cd87ba4ec9e437bae20ba88e9f63938eaf810d83e` | `73eaa6d2e2320466f7a7ba2cd87ba4ec9e437bae20ba88e9f63938eaf810d83e` | yes |
| 7 | `93e714925fd5e244427eff69340dd2ba66e1ec1a2b97d50ff825fec8c3375610` | `93e714925fd5e244427eff69340dd2ba66e1ec1a2b97d50ff825fec8c3375610` | yes |
| 8 | `17c1ca608b0978a0b36fa7b627110266fda5ac2b644fe2af91957b4be71534eb` | `17c1ca608b0978a0b36fa7b627110266fda5ac2b644fe2af91957b4be71534eb` | yes |
| 9 | `bfcf9ad6142e226b6ded3e29155ffb68c335d6decbe56ebb092805251988176a` | `bfcf9ad6142e226b6ded3e29155ffb68c335d6decbe56ebb092805251988176a` | yes |
| 10 | `afd21995d0392b44a040ebe99da3355130b3c700ba5100ae3d9a2208f5d4336a` | `afd21995d0392b44a040ebe99da3355130b3c700ba5100ae3d9a2208f5d4336a` | yes |
| 11 | `6bb518b4222d5b82d473648b74b315f58e36f055cdebe676944bc27118cfb092` | `6bb518b4222d5b82d473648b74b315f58e36f055cdebe676944bc27118cfb092` | yes |
| 12 | `bc4066fa88ff9a60db436692bf38ef237ce0379c426e1e39fe0a4401c1d34802` | `bc4066fa88ff9a60db436692bf38ef237ce0379c426e1e39fe0a4401c1d34802` | yes |
| 13 | `f900f6fb2d4efd7fe37b469d38e53548a765c39afb270fa2eaa8774b7f5a0d9c` | `f900f6fb2d4efd7fe37b469d38e53548a765c39afb270fa2eaa8774b7f5a0d9c` | yes |
| 14 | `056cd98f59133808148a56190c903cb8eae2701976832ed1b3d62f0a8c7d9233` | `056cd98f59133808148a56190c903cb8eae2701976832ed1b3d62f0a8c7d9233` | yes |
| 15 | `036c81ae2e17661a3fe070cb440131e72cbf7f3cbe2f93fbadd68fede7845a48` | `036c81ae2e17661a3fe070cb440131e72cbf7f3cbe2f93fbadd68fede7845a48` | yes |
| 16 | `4745cf37045c6a047c19e84998c2ffb335f9d5e5cddcd05a6aa66786950947e1` | `4745cf37045c6a047c19e84998c2ffb335f9d5e5cddcd05a6aa66786950947e1` | yes |
| 17 | `357289f3d9f0c7d8603ca74d449bdf8218c2e94abc011f545a0d266d598c05a5` | `357289f3d9f0c7d8603ca74d449bdf8218c2e94abc011f545a0d266d598c05a5` | yes |
| 18 | `2606eac022a53c76ca294fdfdb77c429b66791059cf83901ec6d38b99bf3b1d7` | `2606eac022a53c76ca294fdfdb77c429b66791059cf83901ec6d38b99bf3b1d7` | yes |
| 19 | `01cf0deefecbc1574df067fe160f030829e64fad0b068f136276b8c233d245c7` | `01cf0deefecbc1574df067fe160f030829e64fad0b068f136276b8c233d245c7` | yes |
| 20 | `cae0d8992a8b028006b3d338785461a001c9f56e77ce9c9febb7262242050a86` | `cae0d8992a8b028006b3d338785461a001c9f56e77ce9c9febb7262242050a86` | yes |
| 21 | `76313b6ecbfa2a973fb3a4977222add233fe63df1e2241b586722ad117fd5930` | `76313b6ecbfa2a973fb3a4977222add233fe63df1e2241b586722ad117fd5930` | yes |
| 22 | `4827cacbfea50d29313012936dbe25b4806ad814e132dec2e36ba8bd9f545db6` | `4827cacbfea50d29313012936dbe25b4806ad814e132dec2e36ba8bd9f545db6` | yes |
| 23 | `2ec26f1b79d590833b432e629fd3b43125530a0b65f8f030cb26ee2b8b8b5e82` | `2ec26f1b79d590833b432e629fd3b43125530a0b65f8f030cb26ee2b8b8b5e82` | yes |
| 24 | `6147423da41cf2bb6e5772b82a69b99a7ce8a8dfc925b29af58229ce62920fed` | `6147423da41cf2bb6e5772b82a69b99a7ce8a8dfc925b29af58229ce62920fed` | yes |
| 25 | `dfc8a68e7255973a7bc26ea88e4057c35bd711a009abb50a5e9d672b929e9773` | `dfc8a68e7255973a7bc26ea88e4057c35bd711a009abb50a5e9d672b929e9773` | yes |
| 26 | `9647cf254b9341eaeffaf61f81f1f0fde41c717c402734d9922e3e1f787e7484` | `9647cf254b9341eaeffaf61f81f1f0fde41c717c402734d9922e3e1f787e7484` | yes |
| 27 | `8e5246e06868b28bb406426681ed77a5af90c2fa61b25f9375e826c6067c18df` | `8e5246e06868b28bb406426681ed77a5af90c2fa61b25f9375e826c6067c18df` | yes |
| 28 | `d29e30a88c79b0c1f34f61b0abde8df4fa94eb0477ff0a328069bf947823e5da` | `d29e30a88c79b0c1f34f61b0abde8df4fa94eb0477ff0a328069bf947823e5da` | yes |
| 29 | `2c675b39ca10519efc2854a2eb21b87d5303a3c78e58da41e85b4c46b3fbbe30` | `2c675b39ca10519efc2854a2eb21b87d5303a3c78e58da41e85b4c46b3fbbe30` | yes |
| 30 | `a87d59b90e42c23e2470e2a539b2b8efb753ba1d72a391416def486ceb08afe2` | `a87d59b90e42c23e2470e2a539b2b8efb753ba1d72a391416def486ceb08afe2` | yes |
| 31 | `65f346ec63183c6fcf2dfe565a8496148f354c8833c2a5e5a71e15cc8ec69bf6` | `65f346ec63183c6fcf2dfe565a8496148f354c8833c2a5e5a71e15cc8ec69bf6` | yes |
| 32 | `c12213e361c23bab9dd91a74d383a83ee6bf56b85697f6d005291d0c604edcb7` | `c12213e361c23bab9dd91a74d383a83ee6bf56b85697f6d005291d0c604edcb7` | yes |
| 33 | `bf01a84aae7a840fa9f492acfa7b464417728caefdd95af07762845b87098035` | `bf01a84aae7a840fa9f492acfa7b464417728caefdd95af07762845b87098035` | yes |
| 34 | `66d54f0e692d0d6ede6a4067cef6d0660b444627b464b61cc6110f432e05d513` | `66d54f0e692d0d6ede6a4067cef6d0660b444627b464b61cc6110f432e05d513` | yes |
| 35 | `01b368f4645ffc27d81e6395858d88024a0c6e7777d573b0d65a9463f5619c69` | `01b368f4645ffc27d81e6395858d88024a0c6e7777d573b0d65a9463f5619c69` | yes |
| 36 | `1266c404d13d056c432c39cb03994f444d6d45c6468c90547eedead63427d9b7` | `1266c404d13d056c432c39cb03994f444d6d45c6468c90547eedead63427d9b7` | yes |
| 37 | `00c778601f8ea82d8ddb71e121d6bd874930c6ffd951abc5e9b3ddbeeed25461` | `00c778601f8ea82d8ddb71e121d6bd874930c6ffd951abc5e9b3ddbeeed25461` | yes |
| 38 | `ffa3c2f33a8752767ca530519c32560a2a4a604e64380b6f64dd2ddf21dd94cf` | `ffa3c2f33a8752767ca530519c32560a2a4a604e64380b6f64dd2ddf21dd94cf` | yes |
| 39 | `123b4020c93b378f53dac1904f231f2f0868b2ccc7b72031cc1fbc0daca9fb74` | `123b4020c93b378f53dac1904f231f2f0868b2ccc7b72031cc1fbc0daca9fb74` | yes |
| 40 | `33a3b701a1fbb411a666a5995c5a4ab1b7d3ac69f1c401610cd94ecfaffdca23` | `33a3b701a1fbb411a666a5995c5a4ab1b7d3ac69f1c401610cd94ecfaffdca23` | yes |
| 41 | `e68121622e05ffa1734c81b93897f7e6a1f52a1b209473c913a88964fee41a98` | `e68121622e05ffa1734c81b93897f7e6a1f52a1b209473c913a88964fee41a98` | yes |
| 42 | `f57477f948efbabddea0e6980f4319f8a57a9164a4c165f24af0a18816a29453` | `f57477f948efbabddea0e6980f4319f8a57a9164a4c165f24af0a18816a29453` | yes |
| 43 | `649e3eb16d83f3dc106a6bb6182341e8909c7496cb5d33d409fb946dbb476e69` | `649e3eb16d83f3dc106a6bb6182341e8909c7496cb5d33d409fb946dbb476e69` | yes |
| 44 | `b4ff650c4f66ea8cfcf78cd4dc094ef5f861c275fa25034a417d06d00f57f3d7` | `b4ff650c4f66ea8cfcf78cd4dc094ef5f861c275fa25034a417d06d00f57f3d7` | yes |
| 45 | `10d853995d545c90dcacec080b5e03802f4992f4491e6305848a4b242cb6130c` | `10d853995d545c90dcacec080b5e03802f4992f4491e6305848a4b242cb6130c` | yes |

## Provenance

The run's sealed telemetry file has the sha256 `e85f4f36031055fa2ee288c14bf46a1cedd142276c5c2f839a7efeb06fba6e66`. Pelloran keeps it with the run's record. It holds
the full dynamic telemetry, which this report leaves out.
