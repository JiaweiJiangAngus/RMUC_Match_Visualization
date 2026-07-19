# RMUC 2026 区域赛地图地形跨越点

点位几何来自 V1.4.0 规则手册图 5-23，并配准到项目 `web/assets/map.webp`。当前 V2.1.0 的点数、顺序和时限规则未改变，但全国赛新版地图应单独配准。

| ID | 类型 | 方位路线 | 顺序 | 时限 | 中心坐标 (m) | 区域赛事件 |
| --- | --- | --- | --- | ---: | --- | --- |
| `highland_red_low` | 高地 | red | 1:low | 5s | (8.902, 7.548) | 过中央高地 |
| `highland_red_high` | 高地 | red | 2:high | 5s | (10.868, 7.547) | 过中央高地 |
| `highland_blue_low` | 高地 | blue | 1:low | 5s | (19.059, 7.471) | 过中央高地 |
| `highland_blue_high` | 高地 | blue | 2:high | 5s | (17.125, 7.083) | 过中央高地 |
| `road_red_low` | 公路 | red | 1:low | 3s | (8.367, 2.142) | 台阶跨越 |
| `road_red_high` | 公路 | red | 2:high | 3s | (8.365, 1.379) | 台阶跨越 |
| `road_blue_low` | 公路 | blue | 1:low | 3s | (19.611, 12.888) | 台阶跨越 |
| `road_blue_high` | 公路 | blue | 2:high | 3s | (19.611, 13.676) | 台阶跨越 |
| `fly_red_a` | 飞坡 | red | 1:point_a | 10s | (10.609, 0.137) | 飞坡 |
| `fly_red_b` | 飞坡 | red | 2:point_b | 10s | (16.163, 0.088) | 飞坡 |
| `fly_blue_a` | 飞坡 | blue | 1:point_a | 10s | (11.715, 14.917) | 飞坡 |
| `fly_blue_b` | 飞坡 | blue | 2:point_b | 10s | (17.336, 14.871) | 飞坡 |
| `tunnel_red_end_a` | 隧道 | red | 1:end | 3s | (9.456, 1.757) | 无独立标签 |
| `tunnel_red_middle` | 隧道 | red | 2:middle | 3s | (14.897, 1.135) | 无独立标签 |
| `tunnel_red_end_b` | 隧道 | red | 3:end | 3s | (16.410, 3.885) | 无独立标签 |
| `tunnel_blue_end_a` | 隧道 | blue | 1:end | 3s | (11.482, 10.859) | 无独立标签 |
| `tunnel_blue_middle` | 隧道 | blue | 2:middle | 3s | (13.050, 13.881) | 无独立标签 |
| `tunnel_blue_end_b` | 隧道 | blue | 3:end | 3s | (18.569, 13.281) | 无独立标签 |

## 判定口径

- 高地、公路：必须先低点后高点。
- 飞坡：10 秒内检测到同一路线两个点，规则未要求低/高顺序。
- 隧道：一端、中间、另一端；橙色三个轨迹区各覆盖两条并行卡位，共 6 张卡。
- 轨迹坐标存在定位误差，建议调用 `detect_zones(..., padding_m=0.35)`，不要只做中心点距离判断。
- 区域赛 SQLite 可直接验证高地、公路和飞坡；隧道没有独立事件标签，只能依据点位顺序生成弱标签。
