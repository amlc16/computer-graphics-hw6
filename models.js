function createCube() {
  const positions = new Float32Array([
    -1, -1,  1,   1, -1,  1,   1,  1,  1,  -1,  1,  1,   // front
     1, -1, -1,  -1, -1, -1,  -1,  1, -1,   1,  1, -1,   // back
    -1,  1,  1,   1,  1,  1,   1,  1, -1,  -1,  1, -1,   // top
    -1, -1, -1,   1, -1, -1,   1, -1,  1,  -1, -1,  1,   // bottom
     1, -1,  1,   1, -1, -1,   1,  1, -1,   1,  1,  1,   // right
    -1, -1, -1,  -1, -1,  1,  -1,  1,  1,  -1,  1, -1    // left
  ]);

  const faceNormals = [
    [0, 0, 1], [0, 0, -1], [0, 1, 0],
    [0, -1, 0], [1, 0, 0], [-1, 0, 0]
  ];
  const faceTangents = [
    [1, 0, 0], [-1, 0, 0], [1, 0, 0],
    [1, 0, 0], [0, 0, -1], [0, 0, 1]
  ];

  const normals = new Float32Array(
    faceNormals.flatMap(n => [...n, ...n, ...n, ...n])
  );
  const tangents = new Float32Array(
    faceTangents.flatMap(t => [...t, ...t, ...t, ...t])
  );

  // Each cube face gets the full [0,1] x [0,1] texture region.
  const faceUVs = [0, 0, 1, 0, 1, 1, 0, 1];
  const texCoords = new Float32Array(
    Array.from({ length: 6 }, () => faceUVs).flat()
  );

  const indices = new Uint16Array([
     0,  1,  2,   0,  2,  3,   // front
     4,  5,  6,   4,  6,  7,   // back
     8,  9, 10,   8, 10, 11,  // top
    12, 13, 14,  12, 14, 15,  // bottom
    16, 17, 18,  16, 18, 19,  // right
    20, 21, 22,  20, 22, 23   // left
  ]);

  return { positions, normals, tangents, texCoords, indices };
}

// Cylinder axis: Y. Side and cap vertices are separate so each region has
// appropriate normals, tangents, and texture coordinates.
function createCylinder(segments = 32, uRepeat = 1) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];

  // Side surface. U wraps around the cylinder; V runs from top to bottom.
  for (let i = 0; i <= segments; i++) {
    const u = i / segments;
    const angle = 2 * Math.PI * u;
    const x = Math.cos(angle);
    const z = Math.sin(angle);

    for (const y of [-1, 1]) {
      positions.push(x, y, z);
      normals.push(x, 0, z);
      // B = cross(N,T) points toward -Y, matching decreasing V.
      tangents.push(-z, 0, x);
      texCoords.push(u * uRepeat, (1 - y) / 2);
    }
  }

  for (let i = 0; i < segments; i++) {
    const b = i * 2;
    indices.push(b, b + 1, b + 3, b, b + 3, b + 2);
  }

  // Flat-shaded caps with planar XZ texture coordinates.
  for (const y of [-1, 1]) {
    const center = positions.length / 3;
    positions.push(0, y, 0);
    normals.push(0, y, 0);
    tangents.push(1, 0, 0);
    texCoords.push(0.5, 0.5);

    for (let i = 0; i <= segments; i++) {
      const angle = 2 * Math.PI * i / segments;
      const x = Math.cos(angle);
      const z = Math.sin(angle);
      positions.push(x, y, z);
      normals.push(0, y, 0);
      tangents.push(1, 0, 0);
      texCoords.push(0.5 + 0.5 * x, 0.5 - 0.5 * z);
    }

    for (let i = 0; i < segments; i++) {
      const a = center + 1 + i;
      const b = a + 1;
      if (y > 0) indices.push(center, b, a);
      else indices.push(center, a, b);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}


// ------------------------------------------------------------
// SPHERE
// Unit sphere centered at the origin. Y is the polar axis.
// U wraps around longitude; V runs from the north to south pole.
// ------------------------------------------------------------
function createSphere(latitudeBands = 24, longitudeBands = 32, uRepeat = 1, vRepeat = 1) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];

  for (let lat = 0; lat <= latitudeBands; lat++) {
    const v = lat / latitudeBands;
    const phi = v * Math.PI;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);

    for (let lon = 0; lon <= longitudeBands; lon++) {
      const u = lon / longitudeBands;
      const theta = u * Math.PI * 2;
      const cosTheta = Math.cos(theta);
      const sinTheta = Math.sin(theta);

      const x = sinPhi * cosTheta;
      const y = cosPhi;
      const z = sinPhi * sinTheta;

      positions.push(x, y, z);
      normals.push(x, y, z);
      // Tangent follows increasing longitude and is perpendicular to the normal.
      tangents.push(-sinTheta, 0, cosTheta);
      texCoords.push(u * uRepeat, v * vRepeat);
    }
  }

  for (let lat = 0; lat < latitudeBands; lat++) {
    for (let lon = 0; lon < longitudeBands; lon++) {
      const a = lat * (longitudeBands + 1) + lon;
      const b = a + longitudeBands + 1;
      // Winding faces outward.
      indices.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

// ------------------------------------------------------------
// CONE
// Axis: Y. Base radius = 1, height = 2.
// Side vertices are duplicated per segment to provide clean UV seams
// and smooth radial normals. The base cap has separate flat normals.
// ------------------------------------------------------------
function createCone(segments = 32, uRepeat = 1) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];

  // Side triangles: one base vertex on each side of the UV seam and
  // one apex per segment, avoiding a single degenerate apex UV/tangent.
  for (let i = 0; i < segments; i++) {
    const u0 = i / segments;
    const u1 = (i + 1) / segments;
    const midU = (u0 + u1) * 0.5;
    const a0 = u0 * Math.PI * 2;
    const a1 = u1 * Math.PI * 2;
    const am = midU * Math.PI * 2;

    const base0 = [Math.cos(a0), -1, Math.sin(a0)];
    const base1 = [Math.cos(a1), -1, Math.sin(a1)];
    const apex = [0, 1, 0];

    const addSideVertex = (p, angle, u, v) => {
      const nx = Math.cos(angle);
      const nz = Math.sin(angle);
      const invLen = 1 / Math.sqrt(1.25); // slope = radius / height = 1/2
      positions.push(...p);
      normals.push(nx * invLen, 0.5 * invLen, nz * invLen);
      tangents.push(-Math.sin(angle), 0, Math.cos(angle));
      texCoords.push(u * uRepeat, v);
    };

    const first = positions.length / 3;
    addSideVertex(base0, a0, u0, 1);
    addSideVertex(apex, am, midU, 0);
    addSideVertex(base1, a1, u1, 1);
    // Outward-facing side triangle.
    indices.push(first, first + 1, first + 2);
  }

  // Bottom cap, with planar UVs and a constant downward normal.
  const center = positions.length / 3;
  positions.push(0, -1, 0);
  normals.push(0, -1, 0);
  tangents.push(1, 0, 0);
  texCoords.push(0.5, 0.5);

  for (let i = 0; i <= segments; i++) {
    const angle = 2 * Math.PI * i / segments;
    const x = Math.cos(angle);
    const z = Math.sin(angle);
    positions.push(x, -1, z);
    normals.push(0, -1, 0);
    tangents.push(1, 0, 0);
    texCoords.push(0.5 + 0.5 * x, 0.5 + 0.5 * z);
  }

  for (let i = 0; i < segments; i++) {
    // Clockwise from above gives the bottom cap an outward -Y normal.
    indices.push(center, center + 1 + i, center + 1 + i + 1);
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}

// ------------------------------------------------------------
// TORUS
// Standard torus around the Y axis. majorRadius controls the ring
// radius; minorRadius controls the tube radius.
// ------------------------------------------------------------
function createTorus(majorRadius = 0.72, minorRadius = 0.28,
                    majorSegments = 48, minorSegments = 16,
                    uRepeat = 1, vRepeat = 1) {
  const positions = [], normals = [], tangents = [], texCoords = [], indices = [];

  for (let i = 0; i <= majorSegments; i++) {
    const u = i / majorSegments;
    const theta = u * Math.PI * 2;
    const cosTheta = Math.cos(theta);
    const sinTheta = Math.sin(theta);

    for (let j = 0; j <= minorSegments; j++) {
      const v = j / minorSegments;
      const phi = v * Math.PI * 2;
      const cosPhi = Math.cos(phi);
      const sinPhi = Math.sin(phi);

      const ring = majorRadius + minorRadius * cosPhi;
      const x = ring * cosTheta;
      const y = minorRadius * sinPhi;
      const z = ring * sinTheta;

      positions.push(x, y, z);
      normals.push(cosPhi * cosTheta, sinPhi, cosPhi * sinTheta);
      // Tangent follows the major ring direction (increasing U).
      tangents.push(-sinTheta, 0, cosTheta);
      texCoords.push(u * uRepeat, v * vRepeat);
    }
  }

  for (let i = 0; i < majorSegments; i++) {
    for (let j = 0; j < minorSegments; j++) {
      const a = i * (minorSegments + 1) + j;
      const b = (i + 1) * (minorSegments + 1) + j;
      // Outward-facing triangles.
      indices.push(a, a + 1, b, a + 1, b + 1, b);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    tangents: new Float32Array(tangents),
    texCoords: new Float32Array(texCoords),
    indices: new Uint16Array(indices)
  };
}
