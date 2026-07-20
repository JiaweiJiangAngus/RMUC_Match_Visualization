(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.RMUCTerrainRouter = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const FIELD_WIDTH = 28;
  const FIELD_HEIGHT = 15;
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const distance = (left, right) => Math.hypot(left[0] - right[0], left[1] - right[1]);

  function pointInPolygon(point, polygon) {
    let inside = false;
    for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index, index += 1) {
      const [x, y] = polygon[index];
      const [previousX, previousY] = polygon[previous];
      if ((y > point[1]) !== (previousY > point[1])
        && point[0] < (previousX - x) * (point[1] - y) / (previousY - y) + x) inside = !inside;
    }
    return inside;
  }

  function orientation(a, b, c) {
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  }

  function onSegment(a, b, point) {
    return Math.abs(orientation(a, b, point)) < 1e-8
      && point[0] >= Math.min(a[0], b[0]) - 1e-8 && point[0] <= Math.max(a[0], b[0]) + 1e-8
      && point[1] >= Math.min(a[1], b[1]) - 1e-8 && point[1] <= Math.max(a[1], b[1]) + 1e-8;
  }

  function segmentsIntersect(a, b, c, d) {
    const one = orientation(a, b, c);
    const two = orientation(a, b, d);
    const three = orientation(c, d, a);
    const four = orientation(c, d, b);
    if (Math.abs(one) < 1e-8 && onSegment(a, b, c)) return true;
    if (Math.abs(two) < 1e-8 && onSegment(a, b, d)) return true;
    if (Math.abs(three) < 1e-8 && onSegment(c, d, a)) return true;
    if (Math.abs(four) < 1e-8 && onSegment(c, d, b)) return true;
    return (one > 0) !== (two > 0) && (three > 0) !== (four > 0);
  }

  function segmentHitsPolygon(start, end, polygon) {
    if (pointInPolygon(start, polygon) || pointInPolygon(end, polygon)) return true;
    for (let index = 0; index < polygon.length; index += 1) {
      if (segmentsIntersect(start, end, polygon[index], polygon[(index + 1) % polygon.length])) return true;
    }
    return false;
  }

  function pushPoint(route, point) {
    const clean = [clamp(point[0], 0, FIELD_WIDTH), clamp(point[1], 0, FIELD_HEIGHT)];
    if (!route.length || distance(route[route.length - 1], clean) > 1e-4) route.push(clean);
  }

  function routeLength(route) {
    let total = 0;
    for (let index = 1; index < route.length; index += 1) total += distance(route[index - 1], route[index]);
    return total;
  }

  function regionEntries(navigation) {
    return [
      { id: "central_highland", polygon: navigation.regions.central_highland },
      { id: "blue_trapezoid_highland", polygon: navigation.regions.blue_trapezoid_highland },
      { id: "red_trapezoid_highland", polygon: navigation.regions.red_trapezoid_highland },
    ];
  }

  function regionAt(navigation, point) {
    return regionEntries(navigation).find((region) => pointInPolygon(point, region.polygon)) || null;
  }

  function roleCapabilities(navigation, school, role) {
    const data = navigation.teams?.[school]?.[role];
    return { abilities: new Set(data?.abilities || []), reverseFly: Boolean(data?.reverse_fly_ramp?.allowed) };
  }

  function polygonCentroid(polygon) {
    const sum = polygon.reduce((value, point) => [value[0] + point[0], value[1] + point[1]], [0, 0]);
    return [sum[0] / polygon.length, sum[1] / polygon.length];
  }

  function directionVector(direction) {
    return { negative_x: [-1, 0], positive_x: [1, 0], negative_y: [0, -1], positive_y: [0, 1] }[direction] || null;
  }

  function gatePair(gate, region) {
    const center = gate.center;
    const centroid = polygonCentroid(region.polygon);
    const vector = directionVector(gate.high_direction);
    const length = Math.max(0.001, distance(center, centroid));
    const ux = vector?.[0] ?? (centroid[0] - center[0]) / length;
    const uy = vector?.[1] ?? (centroid[1] - center[1]) / length;
    let inside = center;
    let outside = center;
    for (let step = 0.15; step <= 3; step += 0.15) {
      const candidate = [center[0] + ux * step, center[1] + uy * step];
      if (pointInPolygon(candidate, region.polygon)) { inside = candidate; break; }
    }
    for (let step = 0.15; step <= 3; step += 0.15) {
      const candidate = [center[0] - ux * step, center[1] - uy * step];
      if (!pointInPolygon(candidate, region.polygon)) { outside = candidate; break; }
    }
    return { gate, inside, outside };
  }

  function gatesForRegion(navigation, region) {
    if (region.id === "central_highland") {
      return navigation.gates.filter((gate) => gate.category === "central_highland_step").map((gate) => gatePair(gate, region));
    }
    const side = region.id.startsWith("blue") ? "blue" : "red";
    return navigation.gates
      .filter((gate) => gate.side === side && ["slope_43", "trapezoid_highland_step"].includes(gate.category))
      .map((gate) => gatePair(gate, region));
  }

  function gateLabel(gate, direction) {
    const prefix = gate.side === "blue" ? "B" : "R";
    const name = {
      central_highland_step: "中央高地台阶", slope_43: "43°坡",
      trapezoid_highland_step: "梯形高地台阶", road_tunnel: "公路隧道",
      highland_tunnel: "高地隧道", road_step: "公路台阶", rough_road: "起伏路",
      fly_ramp: "飞坡",
    }[gate.category] || gate.category;
    return `${prefix}${gate.gate_index}${direction}${name}`;
  }

  function pushTerrainAction(actions, gate, direction, label, centreOverride) {
    const action = {
      id: gate?.id || `${gate?.category || "terrain"}:${label}`,
      category: gate?.category || "terrain",
      direction,
      label,
      center: [...(centreOverride || gate?.center || [0, 0])],
      polygon: gate?.polygon?.map((point) => [...point]) || null,
    };
    if (!actions.some((item) => item.id === action.id && item.direction === direction)) actions.push(action);
  }

  function ascendingThroughGate(gate, start, end) {
    const vector = directionVector(gate.high_direction);
    if (!vector) return null;
    return (end[0] - start[0]) * vector[0] + (end[1] - start[1]) * vector[1] > 0;
  }

  function heapPush(heap, item) {
    heap.push(item);
    let index = heap.length - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (heap[parent][0] <= item[0]) break;
      heap[index] = heap[parent];
      index = parent;
    }
    heap[index] = item;
  }

  function heapPop(heap) {
    if (!heap.length) return null;
    const root = heap[0];
    const tail = heap.pop();
    if (heap.length) {
      let index = 0;
      while (true) {
        let child = index * 2 + 1;
        if (child >= heap.length) break;
        if (child + 1 < heap.length && heap[child + 1][0] < heap[child][0]) child += 1;
        if (heap[child][0] >= tail[0]) break;
        heap[index] = heap[child];
        index = child;
      }
      heap[index] = tail;
    }
    return root;
  }

  function routeAvoiding(navigation, start, end, extraPolygons = []) {
    const obstacles = [...regionEntries(navigation).map((region) => region.polygon), ...extraPolygons];
    if (!obstacles.some((polygon) => segmentHitsPolygon(start, end, polygon))) return [start, end];
    const step = Number(navigation.routing.grid_m || 0.35);
    const columns = Math.round(FIELD_WIDTH / step) + 1;
    const rows = Math.round(FIELD_HEIGHT / step) + 1;
    const total = columns * rows;
    const nodePoint = (index) => {
      const x = index % columns;
      const y = Math.floor(index / columns);
      return [Math.min(FIELD_WIDTH, x * step), Math.min(FIELD_HEIGHT, y * step)];
    };
    const blocked = (index) => obstacles.some((polygon) => pointInPolygon(nodePoint(index), polygon));
    const nearestIndex = (point) => {
      const baseX = clamp(Math.round(point[0] / step), 0, columns - 1);
      const baseY = clamp(Math.round(point[1] / step), 0, rows - 1);
      const startsBlocked = obstacles.some((polygon) => pointInPolygon(point, polygon));
      for (let radius = 0; radius < 8; radius += 1) {
        for (let dy = -radius; dy <= radius; dy += 1) {
          for (let dx = -radius; dx <= radius; dx += 1) {
            if (Math.max(Math.abs(dx), Math.abs(dy)) !== radius) continue;
            const x = baseX + dx;
            const y = baseY + dy;
            if (x < 0 || x >= columns || y < 0 || y >= rows) continue;
            const index = y * columns + x;
            if (!blocked(index) && (startsBlocked
              || !obstacles.some((polygon) => segmentHitsPolygon(point, nodePoint(index), polygon)))) return index;
          }
        }
      }
      return -1;
    };
    const startIndex = nearestIndex(start);
    const endIndex = nearestIndex(end);
    if (startIndex < 0 || endIndex < 0) return [start];
    const scores = new Float64Array(total);
    scores.fill(Infinity);
    scores[startIndex] = 0;
    const previous = new Int32Array(total);
    previous.fill(-1);
    const closed = new Uint8Array(total);
    const heap = [];
    heapPush(heap, [distance(nodePoint(startIndex), end), startIndex]);
    const moves = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
    while (heap.length) {
      const item = heapPop(heap);
      const index = item[1];
      if (closed[index]) continue;
      if (index === endIndex) break;
      closed[index] = 1;
      const x = index % columns;
      const y = Math.floor(index / columns);
      const from = nodePoint(index);
      for (const [dx, dy] of moves) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= columns || ny < 0 || ny >= rows) continue;
        const next = ny * columns + nx;
        if (closed[next] || blocked(next)) continue;
        const to = nodePoint(next);
        // Check the entire grid edge, not only its midpoint.  Midpoint-only
        // checks allowed diagonal corner cutting through narrow tunnel gates.
        if (obstacles.some((polygon) => segmentHitsPolygon(from, to, polygon))) continue;
        const score = scores[index] + Math.hypot(dx, dy) * step;
        if (score >= scores[next]) continue;
        scores[next] = score;
        previous[next] = index;
        heapPush(heap, [score + distance(to, end), next]);
      }
    }
    if (!Number.isFinite(scores[endIndex])) return [start];
    const reversed = [];
    for (let at = endIndex; at >= 0; at = previous[at]) {
      reversed.push(nodePoint(at));
      if (at === startIndex) break;
    }
    const safeEnd = obstacles.some((polygon) => pointInPolygon(end, polygon)) ? nodePoint(endIndex) : end;
    const raw = [start, ...reversed.reverse(), safeEnd];
    const simplified = [raw[0]];
    let anchor = 0;
    while (anchor < raw.length - 1) {
      let next = anchor + 1;
      for (let candidate = raw.length - 1; candidate > anchor + 1; candidate -= 1) {
        if (!obstacles.some((polygon) => segmentHitsPolygon(raw[anchor], raw[candidate], polygon))) { next = candidate; break; }
      }
      pushPoint(simplified, raw[next]);
      anchor = next;
    }
    return simplified;
  }

  function boundaryCrossing(start, end, polygon) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const hits = [];
    for (let index = 0; index < polygon.length; index += 1) {
      const a = polygon[index];
      const b = polygon[(index + 1) % polygon.length];
      const ex = b[0] - a[0];
      const ey = b[1] - a[1];
      const denominator = dx * ey - dy * ex;
      if (Math.abs(denominator) < 1e-9) continue;
      const t = ((a[0] - start[0]) * ey - (a[1] - start[1]) * ex) / denominator;
      const u = ((a[0] - start[0]) * dy - (a[1] - start[1]) * dx) / denominator;
      if (t >= 0 && t <= 1 && u >= 0 && u <= 1) hits.push({ t, point: [start[0] + dx * t, start[1] + dy * t] });
    }
    hits.sort((left, right) => left.t - right.t);
    return hits[0]?.point || null;
  }

  function crossingPair(start, end, polygon) {
    const crossing = boundaryCrossing(start, end, polygon);
    if (!crossing) return null;
    const length = Math.max(0.001, distance(start, end));
    const ux = (end[0] - start[0]) / length;
    const uy = (end[1] - start[1]) / length;
    return { outside: [crossing[0] - ux * 0.24, crossing[1] - uy * 0.24], inside: [crossing[0] + ux * 0.24, crossing[1] + uy * 0.24] };
  }

  function bestGate(candidates, start, end, abilities, ascending) {
    const allowed = candidates.filter((candidate) => !ascending || abilities.has(candidate.gate.category));
    allowed.sort((left, right) => distance(start, left.outside) + distance(left.inside, end) - distance(start, right.outside) - distance(right.inside, end));
    return allowed[0] || null;
  }

  function applyDirectionalGates(navigation, route, capabilities, passages, actions) {
    const watched = navigation.gates.filter((gate) => [
      "fly_ramp", "road_step", "rough_road", "road_tunnel", "highland_tunnel",
    ].includes(gate.category));
    const output = [route[0]];
    for (let index = 1; index < route.length; index += 1) {
      const start = route[index - 1];
      const end = route[index];
      const blockers = [];
      for (const gate of watched) {
        if (!segmentHitsPolygon(start, end, gate.polygon)) continue;
        if (gate.category === "fly_ramp") {
          const forward = gate.side === "blue" ? end[0] < start[0] : end[0] > start[0];
          const allowed = capabilities.abilities.has("fly_ramp") && (forward || capabilities.reverseFly);
          if (!allowed) blockers.push(gate.polygon);
          else {
            const label = `${gate.side === "blue" ? "B" : "R"}1${forward ? "飞坡" : "反飞坡"}`;
            passages.push(label);
            pushTerrainAction(actions, gate, forward ? "forward" : "reverse", label);
          }
        } else if (gate.category === "road_step") {
          const ascending = ascendingThroughGate(gate, start, end);
          if (ascending !== false && !capabilities.abilities.has("road_step")) blockers.push(gate.polygon);
          else {
            const direction = ascending === false ? "down" : "up";
            const label = `${gate.side === "blue" ? "B" : "R"}${gate.gate_index}${direction === "down" ? "下" : "上"}公路台阶`;
            passages.push(label);
            pushTerrainAction(actions, gate, direction, label);
          }
        } else if (!capabilities.abilities.has(gate.category)) {
          // 隧道不是普通平地：未确认尺寸/机构能力时必须绕行。
          blockers.push(gate.polygon);
        } else {
          const label = gateLabel(gate, "");
          passages.push(label);
          pushTerrainAction(actions, gate, "through", label);
        }
      }
      const segment = blockers.length ? routeAvoiding(navigation, start, end, blockers) : [start, end];
      if (segment.length === 1) pushPoint(output, end);
      else segment.slice(1).forEach((point) => pushPoint(output, point));
    }
    return output;
  }

  function terrainRoute(navigation, start, target, school, role) {
    if (role === "空中") return { route: [start, target], target, passages: ["空中直达"], actions: [], corrected: false };
    const capabilities = roleCapabilities(navigation, school, role);
    const symmetricBlockers = navigation.gates
      .filter((gate) => ["rough_road", "road_tunnel", "highland_tunnel"].includes(gate.category)
        && !capabilities.abilities.has(gate.category))
      .map((gate) => gate.polygon);
    const avoid = (from, to) => routeAvoiding(navigation, from, to, symmetricBlockers);
    const startRegion = regionAt(navigation, start);
    const targetRegion = regionAt(navigation, target);
    const route = [start];
    const passages = [];
    const actions = [];
    let current = start;
    let adjustedTarget = target;
    let corrected = false;
    if (startRegion && targetRegion && startRegion.id === targetRegion.id) {
      pushPoint(route, target);
      return { route, target, passages, actions, corrected };
    }
    if (startRegion) {
      const exit = bestGate(gatesForRegion(navigation, startRegion), start, target, capabilities.abilities, false);
      if (exit) {
        pushPoint(route, exit.inside);
        pushPoint(route, exit.outside);
        current = exit.outside;
        const label = gateLabel(exit.gate, "下");
        passages.push(label);
        pushTerrainAction(actions, exit.gate, "down", label);
      }
    }
    if (targetRegion) {
      const entry = bestGate(gatesForRegion(navigation, targetRegion), current, target, capabilities.abilities, true);
      const jumpPair = targetRegion.id === "central_highland" && capabilities.abilities.has("central_highland_400mm_jump")
        ? crossingPair(current, target, targetRegion.polygon) : null;
      const jumpCentre = jumpPair ? [(jumpPair.outside[0] + jumpPair.inside[0]) / 2, (jumpPair.outside[1] + jumpPair.inside[1]) / 2] : null;
      const crossesStep = jumpCentre && navigation.gates.some((gate) => gate.category === "central_highland_step" && pointInPolygon(jumpCentre, gate.polygon));
      const entryScore = entry ? distance(current, entry.outside) + distance(entry.inside, target) + 0.35 : Infinity;
      const jumpScore = jumpPair && !crossesStep ? distance(current, jumpPair.outside) + distance(jumpPair.inside, target) + 0.65 : Infinity;
      if (jumpScore < entryScore) {
        avoid(current, jumpPair.outside).slice(1).forEach((point) => pushPoint(route, point));
        pushPoint(route, jumpPair.inside);
        pushPoint(route, target);
        passages.push("400mm跳跃上高地");
        pushTerrainAction(actions, { category: "central_highland_400mm_jump" }, "up", "400mm跳跃上高地", jumpCentre);
      } else if (entry) {
        avoid(current, entry.outside).slice(1).forEach((point) => pushPoint(route, point));
        pushPoint(route, entry.inside);
        pushPoint(route, target);
        const label = gateLabel(entry.gate, "上");
        passages.push(label);
        pushTerrainAction(actions, entry.gate, "up", label);
      } else if (jumpPair && !crossesStep) {
        avoid(current, jumpPair.outside).slice(1).forEach((point) => pushPoint(route, point));
        pushPoint(route, jumpPair.inside);
        pushPoint(route, target);
        passages.push("400mm跳跃上高地");
        pushTerrainAction(actions, { category: "central_highland_400mm_jump" }, "up", "400mm跳跃上高地", jumpCentre);
      } else {
        const pair = crossingPair(current, target, targetRegion.polygon);
        adjustedTarget = pair?.outside || current;
        avoid(current, adjustedTarget).slice(1).forEach((point) => pushPoint(route, point));
        corrected = true;
        passages.push("能力不足·停在地形外");
      }
    } else {
      avoid(current, target).slice(1).forEach((point) => pushPoint(route, point));
    }
    const directional = applyDirectionalGates(navigation, route, capabilities, passages, actions);
    return {
      route: directional, target: directional[directional.length - 1] || adjustedTarget,
      passages: [...new Set(passages)], actions, corrected,
    };
  }

  function moveAlongRoute(position, route, metres, stopAtWaypoint = false) {
    if (!route?.length || metres <= 0) return { position: [...position], route: route || [position] };
    const remainingRoute = [[...position], ...route.slice(1)];
    let remaining = metres;
    while (remainingRoute.length > 1) {
      const length = distance(remainingRoute[0], remainingRoute[1]);
      if (length <= remaining + 1e-6) {
        remaining -= length;
        remainingRoute.shift();
        // Ground robots must visibly complete the turn at an avoidance/gate
        // waypoint.  Carrying leftover distance into the next segment made
        // one-second interpolation cut across tunnel corners.
        if (stopAtWaypoint) {
          remaining = 0;
          break;
        }
        continue;
      }
      const ratio = remaining / Math.max(length, 1e-6);
      remainingRoute[0] = [
        remainingRoute[0][0] + (remainingRoute[1][0] - remainingRoute[0][0]) * ratio,
        remainingRoute[0][1] + (remainingRoute[1][1] - remainingRoute[0][1]) * ratio,
      ];
      remaining = 0;
      break;
    }
    return { position: [...remainingRoute[0]], route: remainingRoute };
  }

  function pointSegmentDistance(point, start, end) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const squared = dx * dx + dy * dy;
    if (squared < 1e-9) return distance(point, start);
    const ratio = clamp(((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / squared, 0, 1);
    return distance(point, [start[0] + dx * ratio, start[1] + dy * ratio]);
  }

  function terrainMotion(navigation, position, route, actions, nominalMetres) {
    if (!actions?.length || nominalMetres <= 0 || !route?.length) return { multiplier: 1, action: null };
    const proposed = moveAlongRoute(position, route, nominalMetres, true).position;
    const profiles = navigation.routing?.terrain_speed_multipliers || {};
    const active = actions.filter((action) => {
      if (action.polygon?.length) {
        return pointInPolygon(position, action.polygon)
          || pointInPolygon(proposed, action.polygon)
          || segmentHitsPolygon(position, proposed, action.polygon);
      }
      return pointSegmentDistance(action.center, position, proposed) <= 0.28;
    }).map((action) => {
      const profile = profiles[action.category] || {};
      const multiplier = Number(profile[action.direction] ?? profile.through ?? 1);
      return { action, multiplier: clamp(multiplier, 0.1, 1.25) };
    });
    if (!active.length) return { multiplier: 1, action: null };
    active.sort((left, right) => left.multiplier - right.multiplier);
    return { multiplier: active[0].multiplier, action: active[0].action };
  }

  return {
    distance, pointInPolygon, segmentHitsPolygon, routeLength, regionAt,
    terrainRoute, moveAlongRoute, terrainMotion,
  };
});
