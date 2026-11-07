# G2 fleet LIVE g2fl60b (live SNAP data)

Real 2x corr (`b8_corr2x.py live`, replay None, DEC 16.2734) on 16 corr nodes -> C sender (frame v2, 16384 B bursts) -> `search_rx` v4 -> `b8_owner2x.py engine` on 4 search nodes. Format 2x-gridded-ns4-cint4, plan 0xae1599d449a35b00, 60.0 min from 2026-11-07T00:13:55Z (block0 None).

**Result: FAILED** -- gates {'corr': True, 'corner_turn': False, 'search': True}; leftovers after teardown: {'n03': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n04': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n05': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n06': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n07': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n08': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n10': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n11': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n12': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n14': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n15': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n16': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n18': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n19': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n21': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n22': ['file /tmp/dsart-x2-corr-fast-live.ready'], 'n01': [], 'n02': [], 'n09': [], 'n13': []}

## Corr (per node; block time from corr_fast's summary, over-budget blocks from the trace, which skips the first 32 blocks)

| node | chg | mean ms | p99 ms | max ms | steady p99 / max ms | GPU block mean / p99 ms | blocks > 134.2 ms | blocks | overruns | late start s | TX Gb/s | TX drops | sent n01/n02/n09/n13 | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n03 | 0 | 74.3 | 91.1 | 104.6 | 91.1 / 104.6 | 47.5 / 53.3 | 0 | 26822 | 0 | 0.00 | 2.76 | 0 | 40232724/40232761/40233287/40232854 | ok |
| n04 | 1 | 73.3 | 85.7 | 105.1 | 85.7 / 105.1 | 47.3 / 51.2 | 0 | 26822 | 0 | 0.00 | 2.77 | 0 | 43182712/43182877/43182917/43182996 | ok |
| n05 | 2 | 76.2 | 90.4 | 109.3 | 90.3 / 109.3 | 47.6 / 55.1 | 0 | 26822 | 0 | 0.00 | 2.79 | 0 | 46133046/46133168/46133116/46132859 | ok |
| n06 | 3 | 75.5 | 88.7 | 110.7 | 88.6 / 110.7 | 48.8 / 51.9 | 0 | 26822 | 0 | 0.00 | 2.78 | 0 | 46132766/46132801/46132803/46133284 | ok |
| n07 | 4 | 75.8 | 90.3 | 107.8 | 90.3 / 107.8 | 47.9 / 54.8 | 0 | 26822 | 0 | 0.00 | 2.75 | 0 | 43181938/43182004/43182346/43182135 | ok |
| n08 | 5 | 78.4 | 91.2 | 164.9 | 91.1 / 111.5 | 47.7 / 50.3 | 0 | 26822 | 0 | 0.00 | 2.77 | 0 | 46132296/46132285/46132603/46132490 | ok |
| n10 | 6 | 77.4 | 92.8 | 108.2 | 92.6 / 108.2 | 48.4 / 53.8 | 0 | 26822 | 0 | 0.00 | 2.78 | 0 | 43181811/43181810/43181813/43181811 | ok |
| n11 | 7 | 74.8 | 88.8 | 164.0 | 88.5 / 149.1 | 47.5 / 49.1 | 1 | 26822 | 0 | 0.00 | 2.82 | 0 | 43182719/43182353/43182922/43182535 | ok |
| n12 | 8 | 72.5 | 84.5 | 107.9 | 84.4 / 107.9 | 47.6 / 50.9 | 0 | 26822 | 0 | 0.00 | 2.69 | 0 | 37281746/37281618/37282194/37282147 | ok |
| n14 | 9 | 78.1 | 92.1 | 111.4 | 92.0 / 111.4 | 48.5 / 53.3 | 0 | 26822 | 0 | 0.00 | 2.82 | 0 | 46136139/46132268/46132521/46132415 | ok |
| n15 | 10 | 76.7 | 90.4 | 155.9 | 90.3 / 116.2 | 47.6 / 52.5 | 0 | 26822 | 0 | 0.00 | 2.75 | 0 | 43185457/43182006/43182191/43182066 | ok |
| n16 | 11 | 74.7 | 86.8 | 103.5 | 86.7 / 102.0 | 47.5 / 51.8 | 0 | 26822 | 0 | 0.00 | 2.77 | 0 | 43184600/43181827/43181940/43181940 | ok |
| n18 | 12 | 78.4 | 92.3 | 149.8 | 92.1 / 104.6 | 48.1 / 51.8 | 0 | 26822 | 0 | 0.00 | 2.77 | 0 | 43184467/43181823/43181912/43181938 | ok |
| n19 | 13 | 75.0 | 88.4 | 106.1 | 88.3 / 106.1 | 47.2 / 51.4 | 0 | 26822 | 0 | 0.00 | 2.78 | 0 | 43182551/43181905/43181832/43181823 | ok |
| n21 | 14 | 75.4 | 88.0 | 102.1 | 87.9 / 100.0 | 48.0 / 52.0 | 0 | 26822 | 0 | 0.00 | 2.78 | 0 | 43182667/43182730/43182084/43181922 | ok |
| n22 | 15 | 73.6 | 84.6 | 98.2 | 84.6 / 98.2 | 48.1 / 52.8 | 0 | 26822 | 0 | 0.00 | 2.78 | 0 | 40232221/40232190/40231989/40231758 | ok |

## Corner turn (per search node; loss on the non-lossy flows vs max(1e-4, PHY CRC fraction over the same window))

| node | sent | loss (excl. lossy) | loss (all) | PHY CRC frac | CRC per datagram | NIC discards | rejected | fan-in windows c/f/d | rows missing | Gb/s | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n01 | 690929860 | 2.11e-04 | 2.11e-04 | 1.45e-04 | 1.45e-04 | 0 | 0 | 107267/19/0 | 4.34e-04 | 11.09 | FAIL loss_le_max_1e-4_crc |
| n02 | 690916426 | 2.09e-04 | 2.09e-04 | 1.44e-04 | 1.44e-04 | 0 | 0 | 107269/17/0 | 4.68e-04 | 11.09 | FAIL loss_le_max_1e-4_crc |
| n09 | 690918470 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 | 0 | 107280/6/0 | 2.91e-04 | 11.09 | ok |
| n13 | 690916973 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0 | 0 | 107279/7/0 | 3.26e-04 | 11.09 | ok |

### Paths with loss > 1e-4 (corr -> search)

| search node | paths (loss) | loss on its other paths | lost datagrams | lost per CRC error |
|---|---|---|---|---|
| n01 | n21 (3.4e-03) | 9.8e-07 | 145562 | 1.45 |
| n02 | n21 (3.3e-03) | 9.3e-07 | 144696 | 1.45 |

## Search (per half)

| half | node | duty | wall p99 ms (gate 181.2) | seeks | chain us/image | cycles | start-up skips | stop | gates |
|---|---|---|---|---|---|---|---|---|---|
| 0 | n01 | 0.566 | 153.0 | 0 | 3.58 | 17881 | 0 | idle | ok |
| 1 | n01 | 0.555 | 153.6 | 0 | 3.55 | 17881 | 0 | idle | ok |
| 2 | n02 | 0.546 | 150.3 | 0 | 3.49 | 17881 | 0 | idle | ok |
| 3 | n02 | 0.548 | 159.5 | 0 | 3.57 | 17881 | 0 | idle | ok |
| 4 | n09 | 0.545 | 148.6 | 0 | 3.55 | 17881 | 0 | idle | ok |
| 5 | n09 | 0.546 | 150.8 | 0 | 3.62 | 17881 | 0 | idle | ok |
| 6 | n13 | 0.522 | 153.3 | 0 | 3.51 | 17881 | 0 | idle | ok |
| 7 | n13 | 0.514 | 156.5 | 0 | 3.50 | 17881 | 0 | idle | ok |
