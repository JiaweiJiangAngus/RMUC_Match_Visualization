# RMUC 2026 地形跨越语义（OpenCV 精确边缘 V4）

蓝方语义来自用户 P 图，中央高地边缘在用户描边走廊内由 OpenCV 吸附到原图像素；红方按场地中心 180° 对称生成。

| 编号 | 蓝方 ID | 红方 ID | 地形 | 蓝方中心 (m) | 能力标签 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `blue_fly_ramp` | `red_fly_ramp` | 飞坡 | (14.006, 14.887) | `fly_ramp_capable` |
| 2 | `blue_road_tunnel` | `red_road_tunnel` | 公路隧道 | (18.559, 13.178) | `road_tunnel_fit` |
| 3 | `blue_road_step` | `red_road_step` | 公路台阶 | (19.673, 13.178) | `road_step_capable` |
| 4 | `blue_rough_road` | `red_rough_road` | 起伏路段 | (22.201, 14.317) | `rough_terrain_capable` |
| 5 | `blue_central_highland_step` | `red_central_highland_step` | 中央高地台阶 | (18.355, 7.454) | `central_highland_step_capable` |
| 6 | `blue_highland_tunnel` | `red_highland_tunnel` | 高地隧道 | (14.114, 1.186) | `highland_tunnel_fit` |
| 7 | `blue_slope_43` | `red_slope_43` | 43°坡 | (24.010, 4.605) | `slope_43_capable` |
| 8 | `blue_trapezoid_highland_step` | `red_trapezoid_highland_step` | 梯形高地台阶 | (24.010, 2.525) | `trapezoid_step_capable` |

## 中央高地高差口径

- 常规高地台阶只是编号 5，即中央高地边缘最凸出处。
- B5/R5 轨迹容差区为 190×240 地图像素；纵向宽度来自中央高地内部白色竖线的 Hough 检测长度。
- 橙色虚线所示其余边缘为约 400 mm 高差，默认不可直接通行。
- 精确边缘原始逐行点和算法参数见 `terrain_highland_edges_cv.json`。
- 梯形高地仅是 B7/B8 周围的小平台；左侧长斜坡和道路边界明确排除。
- 该小平台边界作为 `terrain_boundary` 独立存储，不与 43° 坡或梯形高地台阶混合。
- 仅当队伍/机器人已被独立证据标记为 `jump_capable` 时，才把该边缘当作候选跳跃路线。

## 代码接口

- `detect_features(x_m, y_m, padding_m=0.30)`：查询轨迹点附近的跨越接口/高差边缘。
- `classify_transition(prev_x, prev_y, x, y, capabilities=...)`：判定一段移动是否穿过接口或 400 mm 边缘。
- 轨迹容差矩形用于清洗定位噪声，不应解释为设施尺寸。
